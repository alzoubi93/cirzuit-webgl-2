import React, { useEffect, useRef, useMemo } from "react";
import {
  PcbDoc,
  PcbFootprint,
  PcbLayerId,
  isCopperLayer,
  getCopperLayerStandardColor,
} from "@/lib/pcb";
import {
  FootprintSceneGraph,
  FootprintParentNode,
  FootprintChildPad,
  FootprintChildGraphic,
} from "@/lib/pcbFootprintSceneGraph";

export interface PcbWebGLFootprintsProps {
  pcb: PcbDoc;
  pan: { x: number; y: number };
  zoom: number;
  boardRotation: number;
  activeLayer: PcbLayerId;
  dimInactiveLayers: boolean;
  selectedId: string | null;
  selection: any;
  groupSelected: { footprints: string[]; tracks: string[]; vias: string[]; pads: string[] } | null;
  highlightedNetIds?: number[];
  containerWidth: number;
  containerHeight: number;
}

// ============================================================================
// GLSL Vertex Shader: Instanced Footprint Pads via Parent Transformation Matrix
// ============================================================================
const PAD_VERTEX_SHADER = `
attribute vec2 a_quad_pos; // [-1, 1] unit quad

// Parent Transformation Matrix (Passed per-instance from CPU Scene Graph)
// Col 0 & 1: (a, b, c, d), Col 2: (tx, ty)
attribute vec4 a_parent_mat_col01; // [a, b, c, d]
attribute vec2 a_parent_mat_col2;  // [tx, ty]
attribute vec2 a_parent_state;     // (isSelected, isFlipped)

// Child Pad Local Attributes (in Footprint local space, mm)
attribute vec2 a_pad_local_pos;    // (lx, ly) relative to footprint origin
attribute vec4 a_pad_dim_rot;      // (width, height, localRotRad, padShapeId)
attribute vec4 a_pad_drill_props;  // (drillDia, drillOffX, drillOffY, roundrectRatio)
attribute vec4 a_pad_color;        // Base Pad RGBA
attribute vec4 a_pad_drill_color;  // Drill hole RGBA
attribute vec4 a_pad_params;       // (layerId, isSelected, netHighlight, dimAlpha)

// Global Camera Uniforms
uniform vec2 u_resolution; // (w * dpr, h * dpr)
uniform vec2 u_pan;        // pan in CSS pixels (x, y)
uniform float u_zoom;      // zoom factor
uniform float u_rotation;  // board rotation in radians
uniform float u_dpr;       // device pixel ratio

// Varyings to Fragment Shader
varying vec2 v_local_pad_pos;      // Local coords in mm inside pad's own bounding box
varying vec4 v_pad_dim_shape;      // (width, height, shapeId, roundrectRatio)
varying vec4 v_drill_props;        // (drillDia, drillOffX, drillOffY, hasDrill)
varying vec4 v_pad_color;
varying vec4 v_drill_color;
varying vec4 v_params;             // (layerId, isSelected, netHighlight, dimAlpha)

void main() {
  v_pad_dim_shape = vec4(a_pad_dim_rot.x, a_pad_dim_rot.y, a_pad_dim_rot.w, a_pad_drill_props.w);
  v_drill_props = vec4(a_pad_drill_props.xyz, a_pad_drill_props.x > 0.0 ? 1.0 : 0.0);
  v_pad_color = a_pad_color;
  v_drill_color = a_pad_drill_color;
  v_params = a_pad_params;

  float halfW = a_pad_dim_rot.x * 0.5;
  float halfH = a_pad_dim_rot.y * 0.5;
  float selState = max(a_parent_state.x, a_pad_params.y);
  float haloPad = selState > 0.5 ? 0.6 : 0.25;

  // Quad half-size in pad's own coordinate space (with AA/halo padding)
  float quadHalfW = halfW + haloPad + max(0.15, 3.0 / max(u_zoom * u_dpr, 0.001));
  float quadHalfH = halfH + haloPad + max(0.15, 3.0 / max(u_zoom * u_dpr, 0.001));

  vec2 pad_box_pos = vec2(a_quad_pos.x * quadHalfW, a_quad_pos.y * quadHalfH);
  v_local_pad_pos = pad_box_pos;

  // 1. Rotate vertex by child pad's local rotation
  float padRot = a_pad_dim_rot.z;
  float cos_p = cos(padRot);
  float sin_p = sin(padRot);
  vec2 rotated_pad_vertex = vec2(
    cos_p * pad_box_pos.x - sin_p * pad_box_pos.y,
    sin_p * pad_box_pos.x + cos_p * pad_box_pos.y
  );

  // 2. Add child local offset to get footprint-space coordinate
  vec2 fp_local_pos = a_pad_local_pos + rotated_pad_vertex;

  // 3. Apply Parent Transformation Matrix (M_parent * p_local) completely on GPU!
  // a_parent_mat_col01 = (a, b, c, d), a_parent_mat_col2 = (tx, ty)
  vec2 world_pos = vec2(
    a_parent_mat_col01.x * fp_local_pos.x + a_parent_mat_col01.z * fp_local_pos.y + a_parent_mat_col2.x,
    a_parent_mat_col01.y * fp_local_pos.x + a_parent_mat_col01.w * fp_local_pos.y + a_parent_mat_col2.y
  );

  // 4. Apply Board Global Transform (Rotation + Pan + Zoom)
  float cos_b = cos(u_rotation);
  float sin_b = sin(u_rotation);
  vec2 rotated_board_pos = vec2(
    cos_b * world_pos.x - sin_b * world_pos.y,
    sin_b * world_pos.x + cos_b * world_pos.y
  );

  vec2 screen_coord = u_pan + u_zoom * rotated_board_pos;

  vec2 clip_pos = vec2(
    (screen_coord.x * u_dpr / u_resolution.x) * 2.0 - 1.0,
    1.0 - (screen_coord.y * u_dpr / u_resolution.y) * 2.0
  );

  gl_Position = vec4(clip_pos, 0.0, 1.0);
}
`;

// ============================================================================
// GLSL Fragment Shader: Instanced SDF Pad Rendering
// ============================================================================
const PAD_FRAGMENT_SHADER = `
precision highp float;

uniform float u_zoom;
uniform float u_dpr;

varying vec2 v_local_pad_pos;
varying vec4 v_pad_dim_shape;  // (width, height, shapeId, roundrectRatio)
varying vec4 v_drill_props;    // (drillDia, drillOffX, drillOffY, hasDrill)
varying vec4 v_pad_color;
varying vec4 v_drill_color;
varying vec4 v_params;         // (layerId, isSelected, netHighlight, dimAlpha)

// 2D Signed Distance to Rounded Box
float sdRoundBox(vec2 p, vec2 b, float r) {
  vec2 q = abs(p) - b + r;
  return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

// 2D Signed Distance to Circle
float sdCircle(vec2 p, float r) {
  return length(p) - r;
}

void main() {
  float pixelSize = 1.0 / max(u_zoom * u_dpr, 0.001);
  float aa = pixelSize * 1.5;

  float width = v_pad_dim_shape.x;
  float height = v_pad_dim_shape.y;
  float shapeId = v_pad_dim_shape.z; // 0=rect, 1=circle, 2=roundrect, 3=oval
  float roundRatio = v_pad_dim_shape.w;

  float halfW = width * 0.5;
  float halfH = height * 0.5;

  float d_pad = 0.0;
  if (shapeId > 0.5 && shapeId < 1.5) {
    // Circle
    d_pad = sdCircle(v_local_pad_pos, min(halfW, halfH));
  } else if (shapeId > 1.5 && shapeId < 2.5) {
    // Roundrect
    float r = min(halfW, halfH) * max(roundRatio, 0.25);
    d_pad = sdRoundBox(v_local_pad_pos, vec2(halfW, halfH), r);
  } else if (shapeId > 2.5) {
    // Oval
    float r = min(halfW, halfH);
    vec2 b = vec2(halfW, halfH);
    d_pad = sdRoundBox(v_local_pad_pos, b, r);
  } else {
    // Sharp rectangle with microscopic corner rounding for pristine AA
    d_pad = sdRoundBox(v_local_pad_pos, vec2(halfW, halfH), 0.03);
  }

  // Check selection halo
  float isSel = v_params.y;
  float isNetHigh = v_params.z;
  float dimAlpha = v_params.w;

  float haloWidth = (isSel > 0.5 || isNetHigh > 0.5) ? 0.35 : 0.0;
  vec4 haloColor = isSel > 0.5 ? vec4(0.23, 0.51, 0.96, 0.85) : vec4(0.96, 0.62, 0.07, 0.85);

  float padAlpha = 1.0 - smoothstep(-aa, aa, d_pad);
  float haloAlpha = (haloWidth > 0.0) ? (1.0 - smoothstep(haloWidth - aa, haloWidth + aa, d_pad)) : 0.0;

  if (padAlpha <= 0.0 && haloAlpha <= 0.0) {
    discard;
  }

  vec4 col = v_pad_color;
  col.a *= dimAlpha;

  if (haloAlpha > 0.0 && padAlpha < 1.0) {
    col = mix(haloColor, col, padAlpha);
  }

  // Handle Drill hole if Through-Hole pad
  if (v_drill_props.w > 0.5) {
    float drillR = v_drill_props.x * 0.5;
    vec2 drillPos = v_local_pad_pos - v_drill_props.yz;
    float d_drill = sdCircle(drillPos, drillR);
    float drillAlpha = 1.0 - smoothstep(-aa, aa, d_drill);
    
    if (drillAlpha > 0.0) {
      // Internal hole shading for mechanical realism
      vec4 holeCol = v_drill_color;
      col = mix(col, holeCol, drillAlpha);
    }
  }

  gl_FragColor = col;
}
`;

// ============================================================================
// GLSL Vertex & Fragment Shaders for Silkscreen Lines via Parent Transform
// ============================================================================
const SILK_VERTEX_SHADER = `
attribute vec2 a_quad_pos; // [-1, 1]

// Parent Matrix
attribute vec4 a_parent_mat_col01; // [a, b, c, d]
attribute vec2 a_parent_mat_col2;  // [tx, ty]

// Child Local Segment
attribute vec2 a_silk_p0;          // Local start point (lx, ly)
attribute vec2 a_silk_p1;          // Local end point (lx, ly)
attribute vec4 a_silk_props;       // (strokeWidth, isSelected, dimAlpha, unused)
attribute vec4 a_silk_color;       // RGBA

uniform vec2 u_resolution;
uniform vec2 u_pan;
uniform float u_zoom;
uniform float u_rotation;
uniform float u_dpr;

varying vec2 v_local_silk_p;
varying float v_seg_len;
varying float v_radius;
varying vec4 v_silk_color;
varying float v_dim_alpha;

void main() {
  v_radius = a_silk_props.x * 0.5;
  v_silk_color = a_silk_color;
  v_dim_alpha = a_silk_props.z;

  // 1. Transform p0 and p1 from local footprint space to world space on GPU
  vec2 w_p0 = vec2(
    a_parent_mat_col01.x * a_silk_p0.x + a_parent_mat_col01.z * a_silk_p0.y + a_parent_mat_col2.x,
    a_parent_mat_col01.y * a_silk_p0.x + a_parent_mat_col01.w * a_silk_p0.y + a_parent_mat_col2.y
  );

  vec2 w_p1 = vec2(
    a_parent_mat_col01.x * a_silk_p1.x + a_parent_mat_col01.z * a_silk_p1.y + a_parent_mat_col2.x,
    a_parent_mat_col01.y * a_silk_p1.x + a_parent_mat_col01.w * a_silk_p1.y + a_parent_mat_col2.y
  );

  vec2 dir = w_p1 - w_p0;
  float len = length(dir);
  v_seg_len = len;
  vec2 u = len > 0.0001 ? (dir / len) : vec2(1.0, 0.0);
  vec2 n = vec2(-u.y, u.x);

  float pixelSize = 1.0 / max(u_zoom * u_dpr, 0.001);
  float totalR = v_radius + max(pixelSize * 2.0, 0.05);

  vec2 center = (w_p0 + w_p1) * 0.5;
  float halfLen = len * 0.5 + totalR;
  float halfWidth = totalR;

  vec2 world_pos = center + u * (a_quad_pos.x * halfLen) + n * (a_quad_pos.y * halfWidth);
  v_local_silk_p = vec2(a_quad_pos.x * halfLen, a_quad_pos.y * halfWidth);

  // Apply Board Rotation
  float cos_b = cos(u_rotation);
  float sin_b = sin(u_rotation);
  vec2 rot_pos = vec2(
    cos_b * world_pos.x - sin_b * world_pos.y,
    sin_b * world_pos.x + cos_b * world_pos.y
  );

  vec2 screen_coord = u_pan + u_zoom * rot_pos;
  vec2 clip_pos = vec2(
    (screen_coord.x * u_dpr / u_resolution.x) * 2.0 - 1.0,
    1.0 - (screen_coord.y * u_dpr / u_resolution.y) * 2.0
  );

  gl_Position = vec4(clip_pos, 0.0, 1.0);
}
`;

const SILK_FRAGMENT_SHADER = `
precision highp float;

uniform float u_zoom;
uniform float u_dpr;

varying vec2 v_local_silk_p;
varying float v_seg_len;
varying float v_radius;
varying vec4 v_silk_color;
varying float v_dim_alpha;

float sdCapsule(vec2 p, vec2 a, vec2 b, float r) {
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

void main() {
  float pixelSize = 1.0 / max(u_zoom * u_dpr, 0.001);
  float aa = pixelSize * 1.5;

  vec2 p0 = vec2(-v_seg_len * 0.5, 0.0);
  vec2 p1 = vec2(v_seg_len * 0.5, 0.0);

  float d = sdCapsule(v_local_silk_p, p0, p1, v_radius);
  float alpha = 1.0 - smoothstep(-aa, aa, d);

  if (alpha <= 0.0) discard;

  vec4 col = v_silk_color;
  col.a *= (alpha * v_dim_alpha);
  gl_FragColor = col;
}
`;

export const PcbWebGLFootprints: React.FC<PcbWebGLFootprintsProps> = ({
  pcb,
  pan,
  zoom,
  boardRotation,
  activeLayer,
  dimInactiveLayers,
  selectedId,
  selection,
  groupSelected,
  highlightedNetIds = [],
  containerWidth,
  containerHeight,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);

  // Instanced extension & shader programs
  const extInstancingRef = useRef<ANGLE_instanced_arrays | null>(null);
  const padProgramRef = useRef<WebGLProgram | null>(null);
  const silkProgramRef = useRef<WebGLProgram | null>(null);

  // Buffers
  const quadBufferRef = useRef<WebGLBuffer | null>(null);
  const padInstanceBufferRef = useRef<WebGLBuffer | null>(null);
  const silkInstanceBufferRef = useRef<WebGLBuffer | null>(null);

  // Persistent Scene Graph
  const sceneGraph = useMemo(() => new FootprintSceneGraph(), []);

  // Synchronize Footprint Scene Graph whenever pcb footprints or selection state changes
  useEffect(() => {
    const groupFps = groupSelected?.footprints || [];
    sceneGraph.syncFromFootprints(pcb.footprints || [], selectedId, groupFps);
  }, [pcb.footprints, selectedId, groupSelected?.footprints?.join(","), sceneGraph]);

  // Initialize WebGL context, shaders, and geometry buffers
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl", {
      alpha: true,
      antialias: true,
      premultipliedAlpha: true,
    });

    if (!gl) {
      console.warn("WebGL not supported for PCB Footprints rendering");
      return;
    }
    glRef.current = gl;

    const ext = gl.getExtension("ANGLE_instanced_arrays");
    if (!ext) {
      console.warn("ANGLE_instanced_arrays WebGL extension not supported");
      return;
    }
    extInstancingRef.current = ext;

    // Helper to compile shader
    const compileShader = (type: number, source: string): WebGLShader | null => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error("Footprints WebGL shader compile error:", gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const createProg = (vsSrc: string, fsSrc: string): WebGLProgram | null => {
      const vs = compileShader(gl.VERTEX_SHADER, vsSrc);
      const fs = compileShader(gl.FRAGMENT_SHADER, fsSrc);
      if (!vs || !fs) return null;

      const prog = gl.createProgram();
      if (!prog) return null;
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.error("Footprints WebGL program link error:", gl.getProgramInfoLog(prog));
        return null;
      }
      return prog;
    };

    padProgramRef.current = createProg(PAD_VERTEX_SHADER, PAD_FRAGMENT_SHADER);
    silkProgramRef.current = createProg(SILK_VERTEX_SHADER, SILK_FRAGMENT_SHADER);

    // Quad geometry [-1, -1] to [1, 1]
    const quadVertices = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1,
    ]);

    const qBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, qBuf);
    gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);
    quadBufferRef.current = qBuf;

    padInstanceBufferRef.current = gl.createBuffer();
    silkInstanceBufferRef.current = gl.createBuffer();

    return () => {
      if (qBuf) gl.deleteBuffer(qBuf);
      if (padInstanceBufferRef.current) gl.deleteBuffer(padInstanceBufferRef.current);
      if (silkInstanceBufferRef.current) gl.deleteBuffer(silkInstanceBufferRef.current);
      if (padProgramRef.current) gl.deleteProgram(padProgramRef.current);
      if (silkProgramRef.current) gl.deleteProgram(silkProgramRef.current);
    };
  }, []);

  // Main Render Pass: Draw all footprint child elements transformed via Parent matrices
  useEffect(() => {
    const gl = glRef.current;
    const ext = extInstancingRef.current;
    const canvas = canvasRef.current;
    const padProg = padProgramRef.current;
    const silkProg = silkProgramRef.current;

    if (!gl || !ext || !canvas || !padProg || !silkProg) return;

    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(containerWidth * dpr);
    const h = Math.floor(containerHeight * dpr);

    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const parents = sceneGraph.getParents();
    if (parents.length === 0) return;

    const layerMap = new Map((pcb.layers || []).map((l) => [l.id, l]));

    // ------------------------------------------------------------------------
    // 1. Pack Instanced Pad Attributes from Scene Graph
    // ------------------------------------------------------------------------
    // Each pad instance takes:
    // a_parent_mat_col01 (4 floats), a_parent_mat_col2 (2 floats), a_parent_state (2 floats)
    // a_pad_local_pos (2 floats), a_pad_dim_rot (4 floats), a_pad_drill_props (4 floats)
    // a_pad_color (4 floats), a_pad_drill_color (4 floats), a_pad_params (4 floats)
    // Total: 30 floats per instance
    const FLOATS_PER_PAD = 30;
    let totalPads = 0;
    parents.forEach((p) => { totalPads += (p.pads?.length || 0); });

    if (totalPads > 0) {
      const padData = new Float32Array(totalPads * FLOATS_PER_PAD);
      let offset = 0;

      for (const parent of parents) {
        const mat = parent.matrix; // [a, b, c, d, tx, ty]
        const parentSel = parent.selected ? 1.0 : 0.0;
        const parentFlipped = parent.isFlippedBottom ? 1.0 : 0.0;

        for (const pad of (parent.pads || [])) {
          const padLayer = pad.layer === "multi_layer" 
            ? activeLayer 
            : (pad.layer || (parent.isFlippedBottom ? "bottom_copper" : "top_copper"));

          const isLayerVisible = pad.layer === "multi_layer" || layerMap.get(padLayer)?.visible !== false;
          if (!isLayerVisible) continue;

          const isPadActive = !dimInactiveLayers || pad.layer === "multi_layer" || padLayer === activeLayer;
          const dimAlpha = isPadActive ? 1.0 : 0.25;

          const isPadSel = selection?.kind === "pad" && selection.id === pad.id ? 1.0 : 0.0;
          const isNetHigh = pad.netId !== undefined && highlightedNetIds.includes(pad.netId) ? 1.0 : 0.0;

          // Resolve color
          const hexCol = getCopperLayerStandardColor(padLayer);
          const r = parseInt(hexCol.slice(1, 3), 16) / 255;
          const g = parseInt(hexCol.slice(3, 5), 16) / 255;
          const b = parseInt(hexCol.slice(5, 7), 16) / 255;

          // Pad Shape ID
          let shapeId = 0.0;
          if (pad.shape === "circle") shapeId = 1.0;
          else if (pad.shape === "roundrect") shapeId = 2.0;
          else if (pad.shape === "oval") shapeId = 3.0;

          // 1. Parent Transform Matrix & State
          padData[offset++] = mat[0];
          padData[offset++] = mat[1];
          padData[offset++] = mat[2];
          padData[offset++] = mat[3];
          padData[offset++] = mat[4];
          padData[offset++] = mat[5];
          padData[offset++] = parentSel;
          padData[offset++] = parentFlipped;

          // 2. Child Local Coordinates & Shapes
          padData[offset++] = pad.localX;
          padData[offset++] = pad.localY;
          padData[offset++] = pad.width;
          padData[offset++] = pad.height;
          padData[offset++] = (pad.localRotationDeg * Math.PI) / 180;
          padData[offset++] = shapeId;

          // 3. Child Drill Properties
          padData[offset++] = pad.drill || 0.0;
          padData[offset++] = pad.drillX || 0.0;
          padData[offset++] = pad.drillY || 0.0;
          padData[offset++] = pad.roundrectRatio || 0.25;

          // 4. Colors
          padData[offset++] = r;
          padData[offset++] = g;
          padData[offset++] = b;
          padData[offset++] = 1.0;

          // Drill hole color (Dark charcoal interior with metallic ring)
          padData[offset++] = 0.07;
          padData[offset++] = 0.09;
          padData[offset++] = 0.12;
          padData[offset++] = 1.0;

          // 5. Parameters
          padData[offset++] = padLayer === "top_copper" ? 0.0 : 1.0;
          padData[offset++] = isPadSel;
          padData[offset++] = isNetHigh;
          padData[offset++] = dimAlpha;
        }
      }

      const activePadCount = offset / FLOATS_PER_PAD;
      if (activePadCount > 0) {
        gl.useProgram(padProg);

        // Upload buffer
        gl.bindBuffer(gl.ARRAY_BUFFER, padInstanceBufferRef.current);
        gl.bufferData(gl.ARRAY_BUFFER, padData.subarray(0, offset), gl.DYNAMIC_DRAW);

        // Set Uniforms
        gl.uniform2f(gl.getUniformLocation(padProg, "u_resolution"), w, h);
        gl.uniform2f(gl.getUniformLocation(padProg, "u_pan"), pan.x, pan.y);
        gl.uniform1f(gl.getUniformLocation(padProg, "u_zoom"), zoom);
        gl.uniform1f(gl.getUniformLocation(padProg, "u_rotation"), (boardRotation * Math.PI) / 180);
        gl.uniform1f(gl.getUniformLocation(padProg, "u_dpr"), dpr);

        // Bind Quad
        gl.bindBuffer(gl.ARRAY_BUFFER, quadBufferRef.current);
        const locQuad = gl.getAttribLocation(padProg, "a_quad_pos");
        gl.enableVertexAttribArray(locQuad);
        gl.vertexAttribPointer(locQuad, 2, gl.FLOAT, false, 0, 0);

        // Bind Instanced Attributes
        gl.bindBuffer(gl.ARRAY_BUFFER, padInstanceBufferRef.current);
        const stride = FLOATS_PER_PAD * 4;

        const locMat01 = gl.getAttribLocation(padProg, "a_parent_mat_col01");
        const locMat2 = gl.getAttribLocation(padProg, "a_parent_mat_col2");
        const locPState = gl.getAttribLocation(padProg, "a_parent_state");
        const locLocPos = gl.getAttribLocation(padProg, "a_pad_local_pos");
        const locDimRot = gl.getAttribLocation(padProg, "a_pad_dim_rot");
        const locDrill = gl.getAttribLocation(padProg, "a_pad_drill_props");
        const locColor = gl.getAttribLocation(padProg, "a_pad_color");
        const locDrillCol = gl.getAttribLocation(padProg, "a_pad_drill_color");
        const locParams = gl.getAttribLocation(padProg, "a_pad_params");

        const setupInstancedAttr = (loc: number, size: number, floatOff: number) => {
          if (loc >= 0) {
            gl.enableVertexAttribArray(loc);
            gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, floatOff * 4);
            ext.vertexAttribDivisorANGLE(loc, 1);
          }
        };

        setupInstancedAttr(locMat01, 4, 0);
        setupInstancedAttr(locMat2, 2, 4);
        setupInstancedAttr(locPState, 2, 6);
        setupInstancedAttr(locLocPos, 2, 8);
        setupInstancedAttr(locDimRot, 4, 10);
        setupInstancedAttr(locDrill, 4, 14);
        setupInstancedAttr(locColor, 4, 18);
        setupInstancedAttr(locDrillCol, 4, 22);
        setupInstancedAttr(locParams, 4, 26);

        // Render All Instanced Pads
        ext.drawArraysInstancedANGLE(gl.TRIANGLES, 0, 6, activePadCount);

        // Reset Divisors
        [locMat01, locMat2, locPState, locLocPos, locDimRot, locDrill, locColor, locDrillCol, locParams].forEach((loc) => {
          if (loc >= 0) ext.vertexAttribDivisorANGLE(loc, 0);
        });
      }
    }

    // ------------------------------------------------------------------------
    // 2. Pack Instanced Silkscreen Graphics from Scene Graph
    // ------------------------------------------------------------------------
    // a_parent_mat_col01 (4 floats), a_parent_mat_col2 (2 floats)
    // a_silk_p0 (2 floats), a_silk_p1 (2 floats), a_silk_props (4 floats), a_silk_color (4 floats)
    // Total: 18 floats per instance
    const FLOATS_PER_SILK = 18;
    let totalSilk = 0;
    parents.forEach((p) => { totalSilk += (p.silkscreenGraphics?.length || 0); });

    if (totalSilk > 0) {
      const silkData = new Float32Array(totalSilk * FLOATS_PER_SILK);
      let sOffset = 0;

      for (const parent of parents) {
        const mat = parent.matrix;
        const parentSel = parent.selected ? 1.0 : 0.0;

        for (const g of (parent.silkscreenGraphics || [])) {
          const isSilkVis = layerMap.get(g.layer)?.visible !== false;
          if (!isSilkVis) continue;

          const isSilkActive = !dimInactiveLayers || g.layer === activeLayer;
          const dimAlpha = isSilkActive ? 1.0 : 0.3;

          silkData[sOffset++] = mat[0];
          silkData[sOffset++] = mat[1];
          silkData[sOffset++] = mat[2];
          silkData[sOffset++] = mat[3];
          silkData[sOffset++] = mat[4];
          silkData[sOffset++] = mat[5];

          const p0x = g.p0?.x ?? 0;
          const p0y = g.p0?.y ?? 0;
          const p1x = g.p1?.x ?? 0;
          const p1y = g.p1?.y ?? 0;

          silkData[sOffset++] = p0x;
          silkData[sOffset++] = p0y;
          silkData[sOffset++] = p1x;
          silkData[sOffset++] = p1y;

          silkData[sOffset++] = g.strokeWidth;
          silkData[sOffset++] = parentSel;
          silkData[sOffset++] = dimAlpha;
          silkData[sOffset++] = 0.0;

          // Silkscreen Yellow (#fde047)
          silkData[sOffset++] = 0.99;
          silkData[sOffset++] = 0.88;
          silkData[sOffset++] = 0.28;
          silkData[sOffset++] = 0.95;
        }
      }

      const activeSilkCount = sOffset / FLOATS_PER_SILK;
      if (activeSilkCount > 0) {
        gl.useProgram(silkProg);

        gl.bindBuffer(gl.ARRAY_BUFFER, silkInstanceBufferRef.current);
        gl.bufferData(gl.ARRAY_BUFFER, silkData.subarray(0, sOffset), gl.DYNAMIC_DRAW);

        gl.uniform2f(gl.getUniformLocation(silkProg, "u_resolution"), w, h);
        gl.uniform2f(gl.getUniformLocation(silkProg, "u_pan"), pan.x, pan.y);
        gl.uniform1f(gl.getUniformLocation(silkProg, "u_zoom"), zoom);
        gl.uniform1f(gl.getUniformLocation(silkProg, "u_rotation"), (boardRotation * Math.PI) / 180);
        gl.uniform1f(gl.getUniformLocation(silkProg, "u_dpr"), dpr);

        gl.bindBuffer(gl.ARRAY_BUFFER, quadBufferRef.current);
        const locQuad = gl.getAttribLocation(silkProg, "a_quad_pos");
        gl.enableVertexAttribArray(locQuad);
        gl.vertexAttribPointer(locQuad, 2, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, silkInstanceBufferRef.current);
        const sStride = FLOATS_PER_SILK * 4;

        const locSMat01 = gl.getAttribLocation(silkProg, "a_parent_mat_col01");
        const locSMat2 = gl.getAttribLocation(silkProg, "a_parent_mat_col2");
        const locSP0 = gl.getAttribLocation(silkProg, "a_silk_p0");
        const locSP1 = gl.getAttribLocation(silkProg, "a_silk_p1");
        const locSProps = gl.getAttribLocation(silkProg, "a_silk_props");
        const locSCol = gl.getAttribLocation(silkProg, "a_silk_color");

        const setupSilkAttr = (loc: number, size: number, floatOff: number) => {
          if (loc >= 0) {
            gl.enableVertexAttribArray(loc);
            gl.vertexAttribPointer(loc, size, gl.FLOAT, false, sStride, floatOff * 4);
            ext.vertexAttribDivisorANGLE(loc, 1);
          }
        };

        setupSilkAttr(locSMat01, 4, 0);
        setupSilkAttr(locSMat2, 2, 4);
        setupSilkAttr(locSP0, 2, 6);
        setupSilkAttr(locSP1, 2, 8);
        setupSilkAttr(locSProps, 4, 10);
        setupSilkAttr(locSCol, 4, 14);

        ext.drawArraysInstancedANGLE(gl.TRIANGLES, 0, 6, activeSilkCount);

        [locSMat01, locSMat2, locSP0, locSP1, locSProps, locSCol].forEach((loc) => {
          if (loc >= 0) ext.vertexAttribDivisorANGLE(loc, 0);
        });
      }
    }
  }, [
    pcb.layers,
    pan.x,
    pan.y,
    zoom,
    boardRotation,
    activeLayer,
    dimInactiveLayers,
    selectedId,
    selection,
    groupSelected,
    highlightedNetIds,
    containerWidth,
    containerHeight,
    sceneGraph,
  ]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none select-none z-[1]"
      style={{
        width: containerWidth,
        height: containerHeight,
      }}
    />
  );
};
