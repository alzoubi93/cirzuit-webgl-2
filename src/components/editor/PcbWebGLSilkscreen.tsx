import React, { useEffect, useRef, useMemo } from "react";
import {
  PcbDoc,
  PcbLayerId,
  PcbFootprint,
} from "@/lib/pcb";
import {
  FootprintSceneGraph,
  AffineMatrix2D,
  flattenGraphicToSegments,
  GraphicLineSegment,
} from "@/lib/pcbFootprintSceneGraph";
import { generateHersheyTextStrokes } from "@/lib/kicadHersheyFont";

export interface PcbWebGLSilkscreenProps {
  pcb: PcbDoc;
  pan: { x: number; y: number };
  zoom: number;
  boardRotation: number;
  activeLayer: PcbLayerId;
  dimInactiveLayers: boolean;
  selectedId: string | null;
  selection: any;
  groupSelected: { footprints: string[]; tracks: string[]; vias: string[]; pads: string[]; texts?: string[] } | null;
  containerWidth: number;
  containerHeight: number;
  sceneGraph?: FootprintSceneGraph;
  textFontMode?: "vector" | "msdf";
}

// ============================================================================
// GLSL Shaders: Instanced Line Segments with Analytical SDF Round Caps & Matrix Transform
// ============================================================================
const SILK_VERTEX_SHADER = `
attribute vec2 a_quad_pos; // [-1, 1] Master Unit Quad

// Parent Matrix (Instanced)
attribute vec4 a_parent_mat_col01; // [a, b, c, d]
attribute vec2 a_parent_mat_col2;  // [tx, ty]

// Local Segment Attributes (Instanced)
attribute vec2 a_seg_p0;          // Local start point (lx, ly)
attribute vec2 a_seg_p1;          // Local end point (lx, ly)
attribute vec4 a_seg_props;       // (width_mm, halo_mm, dimAlpha, isSelected)
attribute vec4 a_seg_color;       // RGBA color
attribute vec4 a_seg_halo_color;  // RGBA halo color

// Global Camera Uniforms
uniform vec2 u_resolution; // (w * dpr, h * dpr)
uniform vec2 u_pan;        // pan in CSS pixels (x, y)
uniform float u_zoom;      // zoom factor
uniform float u_rotation;  // board rotation in radians
uniform float u_dpr;       // device pixel ratio

// Varyings to Fragment Shader
varying vec2 v_world_pos;
varying vec2 v_world_p0;
varying vec2 v_world_p1;
varying vec4 v_props;
varying vec4 v_color;
varying vec4 v_halo_color;

void main() {
  v_props = a_seg_props;
  v_color = a_seg_color;
  v_halo_color = a_seg_halo_color;

  // 1. Transform p0 and p1 from local footprint space to world mm on GPU
  vec2 w_p0 = vec2(
    a_parent_mat_col01.x * a_seg_p0.x + a_parent_mat_col01.z * a_seg_p0.y + a_parent_mat_col2.x,
    a_parent_mat_col01.y * a_seg_p0.x + a_parent_mat_col01.w * a_seg_p0.y + a_parent_mat_col2.y
  );

  vec2 w_p1 = vec2(
    a_parent_mat_col01.x * a_seg_p1.x + a_parent_mat_col01.z * a_seg_p1.y + a_parent_mat_col2.x,
    a_parent_mat_col01.y * a_seg_p1.x + a_parent_mat_col01.w * a_seg_p1.y + a_parent_mat_col2.y
  );

  v_world_p0 = w_p0;
  v_world_p1 = w_p1;

  float width = a_seg_props.x;
  float haloWidth = a_seg_props.y;
  float radius = width * 0.5;

  // Subpixel anti-aliasing padding in mm
  float pixelSizeMm = 1.0 / max(u_zoom * u_dpr, 0.001);
  float aaPadding = pixelSizeMm * 1.5;
  float totalR = radius + haloWidth + aaPadding;

  vec2 dir = w_p1 - w_p0;
  float segLen = length(dir);
  vec2 u = segLen > 0.00001 ? (dir / segLen) : vec2(1.0, 0.0);
  vec2 n = vec2(-u.y, u.x);

  vec2 center = (w_p0 + w_p1) * 0.5;
  float halfLen = segLen * 0.5 + totalR;
  float halfWidth = totalR;

  vec2 world_pos = center + u * (a_quad_pos.x * halfLen) + n * (a_quad_pos.y * halfWidth);
  v_world_pos = world_pos;

  // Apply Board Rotation
  float cos_r = cos(u_rotation);
  float sin_r = sin(u_rotation);
  vec2 rotated_pos = vec2(
    cos_r * world_pos.x - sin_r * world_pos.y,
    sin_r * world_pos.x + cos_r * world_pos.y
  );

  // Screen coordinates
  vec2 screen_coord = u_pan + u_zoom * rotated_pos;

  // Convert to WebGL NDC [-1, 1]
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

varying vec2 v_world_pos;
varying vec2 v_world_p0;
varying vec2 v_world_p1;
varying vec4 v_props;      // (width, haloWidth, dimAlpha, isSelected)
varying vec4 v_color;
varying vec4 v_halo_color;

float sdCapsule(vec2 p, vec2 a, vec2 b, float r) {
  vec2 pa = p - a, ba = b - a;
  float l2 = dot(ba, ba);
  float h = l2 > 0.000001 ? clamp(dot(pa, ba) / l2, 0.0, 1.0) : 0.0;
  return length(pa - ba * h) - r;
}

void main() {
  float width = v_props.x;
  float haloWidth = v_props.y;
  float dimAlpha = v_props.z;
  float isSelected = v_props.w;
  float radius = width * 0.5;

  float dist = sdCapsule(v_world_pos, v_world_p0, v_world_p1, 0.0);

  // Subpixel AA in mm
  float pixelSizeMm = 1.0 / max(u_zoom * u_dpr, 0.001);
  float halfPx = 0.5 * pixelSizeMm;

  float d_body = dist - radius;
  float d_total = dist - (radius + haloWidth);

  if (d_total > halfPx * 1.5) {
    discard;
  }

  float alpha_total = 1.0 - smoothstep(-halfPx, halfPx, d_total);
  float alpha_body = 1.0 - smoothstep(-halfPx, halfPx, d_body);

  vec4 finalColor;
  if (isSelected > 0.5 && haloWidth > 0.0) {
    vec4 mixedColor = mix(v_halo_color, v_color, alpha_body);
    finalColor = vec4(mixedColor.rgb, alpha_total * dimAlpha * mixedColor.a);
  } else {
    finalColor = vec4(v_color.rgb, alpha_body * dimAlpha * v_color.a);
  }

  if (finalColor.a <= 0.001) discard;
  gl_FragColor = finalColor;
}
`;

// 22 Floats per Line Segment Instance:
// [0..3]: Parent Matrix col01 [a, b, c, d]
// [4..5]: Parent Matrix col2  [tx, ty]
// [6..7]: Local P0 [x0, y0]
// [8..9]: Local P1 [x1, y1]
// [10..13]: Props [width, haloWidth, dimAlpha, isSelected]
// [14..17]: Color RGBA [r, g, b, a]
// [18..21]: Halo Color RGBA [r, g, b, a]
const FLOATS_PER_SEGMENT_INSTANCE = 22;

/**
 * Parses Hex or RGBA string into normalized [r, g, b, a]
 */
function parseColorToRgba(colorStr: string, defaultAlpha = 1.0): [number, number, number, number] {
  if (!colorStr) return [1, 1, 1, defaultAlpha];

  if (colorStr.startsWith("#")) {
    let hex = colorStr.slice(1);
    if (hex.length === 3) {
      hex = hex.split("").map((c) => c + c).join("");
    }
    if (hex.length === 6) {
      const num = parseInt(hex, 16);
      return [((num >> 16) & 255) / 255, ((num >> 8) & 255) / 255, (num & 255) / 255, defaultAlpha];
    }
    if (hex.length === 8) {
      const num = parseInt(hex, 16);
      return [
        ((num >> 24) & 255) / 255,
        ((num >> 16) & 255) / 255,
        ((num >> 8) & 255) / 255,
        (num & 255) / 255,
      ];
    }
  }

  if (colorStr.startsWith("rgba")) {
    const parts = colorStr.match(/[\d.]+/g);
    if (parts && parts.length >= 4) {
      return [parseFloat(parts[0]) / 255, parseFloat(parts[1]) / 255, parseFloat(parts[2]) / 255, parseFloat(parts[3])];
    }
  }

  return [1.0, 0.95, 0.3, defaultAlpha];
}

export const PcbWebGLSilkscreen: React.FC<PcbWebGLSilkscreenProps> = ({
  pcb,
  pan,
  zoom,
  boardRotation,
  activeLayer,
  dimInactiveLayers,
  selectedId,
  selection,
  groupSelected,
  containerWidth,
  containerHeight,
  sceneGraph,
  textFontMode = "vector",
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const progRef = useRef<WebGLProgram | null>(null);
  const extInstancingRef = useRef<ANGLE_instanced_arrays | null>(null);

  const quadVboRef = useRef<WebGLBuffer | null>(null);
  const instanceVboRef = useRef<WebGLBuffer | null>(null);

  const localSceneGraphRef = useRef<FootprintSceneGraph>(new FootprintSceneGraph());
  const activeSceneGraph = sceneGraph || localSceneGraphRef.current;

  // Stabilized dependency string for group footprints
  const groupFpKey = useMemo(() => {
    return groupSelected?.footprints?.slice().sort().join(",") || "";
  }, [groupSelected?.footprints]);

  // Selected footprint ID
  const activeSelectedFpId = useMemo(() => {
    if (selectedId) return selectedId;
    if (selection?.kind === "footprint" && selection.id) return selection.id;
    return null;
  }, [selectedId, selection]);

  // Sync Scene Graph
  useEffect(() => {
    const groupList = groupFpKey ? groupFpKey.split(",") : [];
    activeSceneGraph.syncFromFootprints(pcb.footprints || [], activeSelectedFpId, groupList);
  }, [pcb.footprints, activeSelectedFpId, groupFpKey, activeSceneGraph]);

  // Initialize WebGL Context and Shaders
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl", {
      alpha: true,
      antialias: true,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
    });

    if (!gl) return;
    glRef.current = gl;

    const ext = gl.getExtension("ANGLE_instanced_arrays");
    if (!ext) {
      console.warn("WebGL ANGLE_instanced_arrays extension not supported");
      return;
    }
    extInstancingRef.current = ext;

    // Compile Shaders
    const createShader = (type: number, src: string) => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, src);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error("Silkscreen Shader Error:", gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const vs = createShader(gl.VERTEX_SHADER, SILK_VERTEX_SHADER);
    const fs = createShader(gl.FRAGMENT_SHADER, SILK_FRAGMENT_SHADER);
    if (!vs || !fs) return;

    const prog = gl.createProgram();
    if (!prog) return;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error("Silkscreen Link Error:", gl.getProgramInfoLog(prog));
      return;
    }
    progRef.current = prog;

    // Master Quad Buffer [-1, 1]
    const quadVertices = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1,
    ]);
    const quadVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadVbo);
    gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);
    quadVboRef.current = quadVbo;

    // Instanced VBO
    const instanceVbo = gl.createBuffer();
    instanceVboRef.current = instanceVbo;

    return () => {
      if (quadVboRef.current) gl.deleteBuffer(quadVboRef.current);
      if (instanceVboRef.current) gl.deleteBuffer(instanceVboRef.current);
      if (progRef.current) gl.deleteProgram(progRef.current);
    };
  }, []);

  // Main Render Loop & Buffer Upload
  useEffect(() => {
    const canvas = canvasRef.current;
    const gl = glRef.current;
    const prog = progRef.current;
    const ext = extInstancingRef.current;
    if (!canvas || !gl || !prog || !ext) return;

    const dpr = window.devicePixelRatio || 1;
    const w = containerWidth;
    const h = containerHeight;

    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    }
    gl.viewport(0, 0, canvas.width, canvas.height);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Layer Visibility & Colors map
    const layerMap = new Map<string, { visible: boolean; color: string }>();
    (pcb.layers || []).forEach((l) => {
      layerMap.set(l.id, { visible: l.visible !== false, color: l.color });
    });

    const isLayerVisible = (layerId: PcbLayerId) => {
      if (layerMap.has(layerId)) {
        return layerMap.get(layerId)!.visible;
      }
      return true;
    };

    const getLayerColor = (layerId: PcbLayerId, defaultHex = "#fde047") => {
      return layerMap.get(layerId)?.color || defaultHex;
    };

    // Collect all Line Segments from Footprints & Board Graphics
    const parents = activeSceneGraph.getParents();
    const segmentInstances: number[] = [];

    // 1. Process Footprint Silkscreen & Assembly Outlines (Instanced with Parent Matrix)
    for (const parent of parents) {
      const isSel = parent.selected;
      const mat = parent.matrix; // [a, b, c, d, tx, ty]
      const haloWidth = isSel ? 0.35 : 0.0;
      const haloColor = isSel ? [0.23, 0.51, 0.96, 0.85] : [0.96, 0.62, 0.07, 0.85];

      // Combine Silkscreen and Fab Graphics
      const allGraphics = [...parent.silkscreenGraphics, ...parent.fabGraphics];

      for (const g of allGraphics) {
        if (!isLayerVisible(g.layer)) continue;

        const isLayerActive = activeLayer === g.layer;
        const dimAlpha = dimInactiveLayers && !isLayerActive && !isSel ? 0.25 : 1.0;
        const defaultCol = g.layer === "bottom_silkscreen" ? "#fef08a" : g.layer === "outline" ? "#94a3b8" : "#fde047";
        const [r, gCol, b, a] = parseColorToRgba(getLayerColor(g.layer, defaultCol));

        const segments = flattenGraphicToSegments(g);

        for (const seg of segments) {
          if (!seg) continue;
          const p0x = seg.p0?.x ?? 0;
          const p0y = seg.p0?.y ?? 0;
          const p1x = seg.p1?.x ?? 0;
          const p1y = seg.p1?.y ?? 0;
          // [0..3]: Parent Matrix col01 [a, b, c, d]
          segmentInstances.push(mat[0], mat[1], mat[2], mat[3]);
          // [4..5]: Parent Matrix col2  [tx, ty]
          segmentInstances.push(mat[4], mat[5]);
          // [6..7]: Local P0 [x0, y0]
          segmentInstances.push(p0x, p0y);
          // [8..9]: Local P1 [x1, y1]
          segmentInstances.push(p1x, p1y);
          // [10..13]: Props [width, haloWidth, dimAlpha, isSelected]
          segmentInstances.push(seg.strokeWidth || 0.15, haloWidth, dimAlpha, isSel ? 1.0 : 0.0);
          // [14..17]: Color RGBA
          segmentInstances.push(r, gCol, b, a);
          // [18..21]: Halo Color RGBA
          segmentInstances.push(haloColor[0], haloColor[1], haloColor[2], haloColor[3]);
        }
      }
    }

    // 2. Process Standalone Board Drawings & Outline (Identity Matrix)
    const identityMat: AffineMatrix2D = [1, 0, 0, 1, 0, 0];

    // Board Edge Cuts Outline
    if (isLayerVisible("outline")) {
      const [r, gCol, b, a] = parseColorToRgba(getLayerColor("outline", "#eab308"));
      const bW = pcb.width || 100;
      const bH = pcb.height || 80;
      const outlineCorners = [
        { p0: { x: 0, y: 0 }, p1: { x: bW, y: 0 } },
        { p0: { x: bW, y: 0 }, p1: { x: bW, y: bH } },
        { p0: { x: bW, y: bH }, p1: { x: 0, y: bH } },
        { p0: { x: 0, y: bH }, p1: { x: 0, y: 0 } },
      ];

      for (const seg of outlineCorners) {
        if (!seg) continue;
        const p0x = seg.p0?.x ?? 0;
        const p0y = seg.p0?.y ?? 0;
        const p1x = seg.p1?.x ?? 0;
        const p1y = seg.p1?.y ?? 0;
        segmentInstances.push(identityMat[0], identityMat[1], identityMat[2], identityMat[3]);
        segmentInstances.push(identityMat[4], identityMat[5]);
        segmentInstances.push(p0x, p0y);
        segmentInstances.push(p1x, p1y);
        segmentInstances.push(0.25, 0.0, 1.0, 0.0);
        segmentInstances.push(r, gCol, b, a);
        segmentInstances.push(0, 0, 0, 0);
      }
    }

    // Custom Board Drawings
    if (Array.isArray(pcb.drawings)) {
      for (const d of pcb.drawings) {
        if (!isLayerVisible(d.layer)) continue;
        const [r, gCol, b, a] = parseColorToRgba(getLayerColor(d.layer, "#fde047"));
        const strokeW = d.strokeWidth || 0.15;

        let segs: GraphicLineSegment[] = [];
        if (d.type === "line" && d.x2 !== undefined && d.y2 !== undefined) {
          segs = [{ p0: { x: d.x1, y: d.y1 }, p1: { x: d.x2, y: d.y2 }, strokeWidth: strokeW, layer: d.layer }];
        } else if (d.type === "rect" && d.width !== undefined && d.height !== undefined) {
          const x2 = d.x1 + d.width;
          const y2 = d.y1 + d.height;
          segs = [
            { p0: { x: d.x1, y: d.y1 }, p1: { x: x2, y: d.y1 }, strokeWidth: strokeW, layer: d.layer },
            { p0: { x: x2, y: d.y1 }, p1: { x: x2, y: y2 }, strokeWidth: strokeW, layer: d.layer },
            { p0: { x: x2, y: y2 }, p1: { x: d.x1, y: y2 }, strokeWidth: strokeW, layer: d.layer },
            { p0: { x: d.x1, y: y2 }, p1: { x: d.x1, y: d.y1 }, strokeWidth: strokeW, layer: d.layer },
          ];
        }

        for (const seg of segs) {
          if (!seg) continue;
          const p0x = seg.p0?.x ?? 0;
          const p0y = seg.p0?.y ?? 0;
          const p1x = seg.p1?.x ?? 0;
          const p1y = seg.p1?.y ?? 0;
          segmentInstances.push(identityMat[0], identityMat[1], identityMat[2], identityMat[3]);
          segmentInstances.push(identityMat[4], identityMat[5]);
          segmentInstances.push(p0x, p0y);
          segmentInstances.push(p1x, p1y);
          segmentInstances.push(seg.strokeWidth || 0.15, 0.0, 1.0, 0.0);
          segmentInstances.push(r, gCol, b, a);
          segmentInstances.push(0, 0, 0, 0);
        }
      }
    }

    // Custom Board Graphics (KiCad PCB gr_line, gr_arc, gr_circle, gr_rect, gr_poly)
    if (Array.isArray(pcb.graphics)) {
      for (const g of pcb.graphics) {
        if (!g || !isLayerVisible(g.layer as PcbLayerId)) continue;
        const [r, gCol, b, a] = parseColorToRgba(getLayerColor(g.layer as PcbLayerId, "#fde047"));
        const strokeW = g.width || 0.15;

        let segs: GraphicLineSegment[] = [];
        if (g.kind === "line" && g.start && g.end) {
          segs = [{ p0: { x: g.start.x, y: g.start.y }, p1: { x: g.end.x, y: g.end.y }, strokeWidth: strokeW, layer: g.layer }];
        } else if (g.kind === "rect" && g.start && g.end) {
          const x1 = Math.min(g.start.x, g.end.x);
          const x2 = Math.max(g.start.x, g.end.x);
          const y1 = Math.min(g.start.y, g.end.y);
          const y2 = Math.max(g.start.y, g.end.y);
          segs = [
            { p0: { x: x1, y: y1 }, p1: { x: x2, y: y1 }, strokeWidth: strokeW, layer: g.layer },
            { p0: { x: x2, y: y1 }, p1: { x: x2, y: y2 }, strokeWidth: strokeW, layer: g.layer },
            { p0: { x: x2, y: y2 }, p1: { x: x1, y: y2 }, strokeWidth: strokeW, layer: g.layer },
            { p0: { x: x1, y: y2 }, p1: { x: x1, y: y1 }, strokeWidth: strokeW, layer: g.layer },
          ];
        } else if ((g.kind === "poly" || g.kind === "curve") && Array.isArray(g.points) && g.points.length > 1) {
          for (let i = 0; i < g.points.length - 1; i++) {
            const pt0 = g.points[i];
            const pt1 = g.points[i + 1];
            if (pt0 && pt1 && typeof pt0.x === "number" && typeof pt1.x === "number") {
              segs.push({ p0: { x: pt0.x, y: pt0.y }, p1: { x: pt1.x, y: pt1.y }, strokeWidth: strokeW, layer: g.layer });
            }
          }
        } else if (g.kind === "arc" && Array.isArray(g.points) && g.points.length > 1) {
          for (let i = 0; i < g.points.length - 1; i++) {
            const pt0 = g.points[i];
            const pt1 = g.points[i + 1];
            if (pt0 && pt1 && typeof pt0.x === "number" && typeof pt1.x === "number") {
              segs.push({ p0: { x: pt0.x, y: pt0.y }, p1: { x: pt1.x, y: pt1.y }, strokeWidth: strokeW, layer: g.layer });
            }
          }
        } else if (g.kind === "circle" && g.center && g.end) {
          const radius = Math.hypot(g.end.x - g.center.x, g.end.y - g.center.y);
          const steps = 32;
          for (let i = 0; i < steps; i++) {
            const a0 = (i / steps) * Math.PI * 2;
            const a1 = ((i + 1) / steps) * Math.PI * 2;
            segs.push({
              p0: { x: g.center.x + Math.cos(a0) * radius, y: g.center.y + Math.sin(a0) * radius },
              p1: { x: g.center.x + Math.cos(a1) * radius, y: g.center.y + Math.sin(a1) * radius },
              strokeWidth: strokeW,
              layer: g.layer,
            });
          }
        }

        for (const seg of segs) {
          if (!seg || !seg.p0 || !seg.p1) continue;
          const p0x = seg.p0.x ?? 0;
          const p0y = seg.p0.y ?? 0;
          const p1x = seg.p1.x ?? 0;
          const p1y = seg.p1.y ?? 0;
          segmentInstances.push(identityMat[0], identityMat[1], identityMat[2], identityMat[3]);
          segmentInstances.push(identityMat[4], identityMat[5]);
          segmentInstances.push(p0x, p0y);
          segmentInstances.push(p1x, p1y);
          segmentInstances.push(seg.strokeWidth || 0.15, 0.0, 1.0, 0.0);
          segmentInstances.push(r, gCol, b, a);
          segmentInstances.push(0, 0, 0, 0);
        }
      }
    }

    // KiCad Hershey Vector Stroke Font for PCB Texts and Footprint Labels
    if (textFontMode !== "msdf") {
      // 1. Standalone PCB Texts (pcb.texts)
      if (Array.isArray(pcb.texts)) {
        for (const t of pcb.texts) {
          if (!t || !isLayerVisible(t.layer)) continue;
          const isLayerActive = activeLayer === t.layer;
          const isSel = selection?.kind === "text" && selection.id === t.id;
          const isGroupSel = Boolean(groupSelected?.texts?.includes(t.id));
          const dimAlpha = dimInactiveLayers && !isLayerActive && !isSel && !isGroupSel ? 0.25 : 1.0;

          const defaultCol = t.layer === "bottom_silkscreen" ? "#fef08a" : t.layer.includes("copper") ? "#22c55e" : "#fde047";
          const baseColor = isGroupSel ? "#f59e0b" : isSel ? "#3b82f6" : getLayerColor(t.layer, defaultCol);
          const [r, gCol, b, a] = parseColorToRgba(baseColor);
          const finalAlpha = a * dimAlpha;

          const strokes = generateHersheyTextStrokes({
            text: t.text,
            x: t.x,
            y: t.y,
            size: t.size || 1.2,
            thickness: Math.max(0.1, (t.size || 1.2) * 0.12),
            rotation: t.rotation || 0,
            layer: t.layer,
            justify: (t.justify?.[0] as any) || "center",
            verticalAlign: "middle",
            bold: t.bold,
            italic: t.italic,
            mirror: t.mirror || t.layer.includes("bottom"),
          });

          for (const seg of strokes) {
            segmentInstances.push(identityMat[0], identityMat[1], identityMat[2], identityMat[3]);
            segmentInstances.push(identityMat[4], identityMat[5]);
            segmentInstances.push(seg.p0.x, seg.p0.y);
            segmentInstances.push(seg.p1.x, seg.p1.y);
            segmentInstances.push(seg.strokeWidth, 0.0, 1.0, 0.0);
            segmentInstances.push(r, gCol, b, finalAlpha);
            segmentInstances.push(0, 0, 0, 0);
          }
        }
      }

      // 2. Footprint Reference & Value Texts
      if (Array.isArray(pcb.footprints)) {
        for (const fp of pcb.footprints) {
          if (!fp) continue;
          const isBottom = fp.nativeKicadFootprint?.layer === "B.Cu" || false;
          const silkLayerId: PcbLayerId = isBottom ? "bottom_silkscreen" : "silkscreen";
          if (!isLayerVisible(silkLayerId)) continue;

          const isFpActive = !dimInactiveLayers || (isBottom ? activeLayer.includes("bottom") : !activeLayer.includes("bottom"));
          const isFpSel = (selectedId === fp.id) || Boolean(groupSelected?.footprints.includes(fp.id));
          const dimAlpha = isFpActive || isFpSel ? 1.0 : 0.25;

          const [r, gCol, b, a] = parseColorToRgba(getLayerColor(silkLayerId, "#fde047"));
          const finalAlpha = a * dimAlpha;

          // Footprint Reference
          const refStr = fp.reference || fp.nativeKicadFootprint?.properties?.Reference;
          if (refStr && !fp.nativeKicadFootprint) {
            const refStrokes = generateHersheyTextStrokes({
              text: refStr,
              x: fp.x,
              y: fp.y - 1.6,
              size: 1.0,
              thickness: 0.14,
              rotation: fp.rotation || 0,
              layer: silkLayerId,
              justify: "center",
              verticalAlign: "middle",
              bold: false,
              mirror: isBottom,
            });
            for (const seg of refStrokes) {
              segmentInstances.push(identityMat[0], identityMat[1], identityMat[2], identityMat[3]);
              segmentInstances.push(identityMat[4], identityMat[5]);
              segmentInstances.push(seg.p0.x, seg.p0.y);
              segmentInstances.push(seg.p1.x, seg.p1.y);
              segmentInstances.push(seg.strokeWidth, 0.0, 1.0, 0.0);
              segmentInstances.push(r, gCol, b, finalAlpha);
              segmentInstances.push(0, 0, 0, 0);
            }
          }

          // Footprint Value
          const valStr = fp.value || fp.nativeKicadFootprint?.properties?.Value;
          if (valStr && valStr !== refStr && valStr !== "~" && !fp.nativeKicadFootprint) {
            const valStrokes = generateHersheyTextStrokes({
              text: valStr,
              x: fp.x,
              y: fp.y + 1.6,
              size: 0.85,
              thickness: 0.12,
              rotation: fp.rotation || 0,
              layer: silkLayerId,
              justify: "center",
              verticalAlign: "middle",
              mirror: isBottom,
            });
            for (const seg of valStrokes) {
              segmentInstances.push(identityMat[0], identityMat[1], identityMat[2], identityMat[3]);
              segmentInstances.push(identityMat[4], identityMat[5]);
              segmentInstances.push(seg.p0.x, seg.p0.y);
              segmentInstances.push(seg.p1.x, seg.p1.y);
              segmentInstances.push(seg.strokeWidth, 0.0, 1.0, 0.0);
              segmentInstances.push(r, gCol, b, finalAlpha * 0.85);
              segmentInstances.push(0, 0, 0, 0);
            }
          }

          // Native KiCad Footprint Text Graphics
          if (fp.nativeKicadFootprint?.graphics) {
            for (const g of fp.nativeKicadFootprint.graphics) {
              if (g.type === "text" && g.text) {
                const gLayer: PcbLayerId = g.layer?.includes("SilkS") 
                  ? (isBottom ? "bottom_silkscreen" : "silkscreen") 
                  : "silkscreen";
                if (!isLayerVisible(gLayer)) continue;

                const gPos = g.position || { x: 0, y: 0 };
                const rotRad = ((fp.rotation || 0) * Math.PI) / 180;
                const wx = fp.x + (gPos.x * Math.cos(rotRad) - gPos.y * Math.sin(rotRad));
                const wy = fp.y + (gPos.x * Math.sin(rotRad) + gPos.y * Math.cos(rotRad));

                const gStrokes = generateHersheyTextStrokes({
                  text: g.text,
                  x: wx,
                  y: wy,
                  size: g.size?.y || 0.9,
                  thickness: g.stroke?.width || 0.12,
                  rotation: (fp.rotation || 0) + (g.rotation || 0),
                  layer: gLayer,
                  justify: "center",
                  verticalAlign: "middle",
                  bold: g.bold,
                  italic: g.italic,
                  mirror: isBottom || g.mirror,
                });
                for (const seg of gStrokes) {
                  segmentInstances.push(identityMat[0], identityMat[1], identityMat[2], identityMat[3]);
                  segmentInstances.push(identityMat[4], identityMat[5]);
                  segmentInstances.push(seg.p0.x, seg.p0.y);
                  segmentInstances.push(seg.p1.x, seg.p1.y);
                  segmentInstances.push(seg.strokeWidth, 0.0, 1.0, 0.0);
                  segmentInstances.push(r, gCol, b, finalAlpha);
                  segmentInstances.push(0, 0, 0, 0);
                }
              }
            }
          }
        }
      }
    }

    const instanceCount = segmentInstances.length / FLOATS_PER_SEGMENT_INSTANCE;
    if (instanceCount === 0) return;

    // Upload to GPU Instance Buffer
    const instanceData = new Float32Array(segmentInstances);
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceVboRef.current);
    gl.bufferData(gl.ARRAY_BUFFER, instanceData, gl.DYNAMIC_DRAW);

    gl.useProgram(prog);

    // Bind Uniforms
    const uResLoc = gl.getUniformLocation(prog, "u_resolution");
    const uPanLoc = gl.getUniformLocation(prog, "u_pan");
    const uZoomLoc = gl.getUniformLocation(prog, "u_zoom");
    const uRotLoc = gl.getUniformLocation(prog, "u_rotation");
    const uDprLoc = gl.getUniformLocation(prog, "u_dpr");

    gl.uniform2f(uResLoc, canvas.width, canvas.height);
    gl.uniform2f(uPanLoc, pan.x, pan.y);
    gl.uniform1f(uZoomLoc, zoom);
    const rotRad = (boardRotation * Math.PI) / 180;
    gl.uniform1f(uRotLoc, rotRad);
    gl.uniform1f(uDprLoc, dpr);

    // Bind Quad Template VBO
    gl.bindBuffer(gl.ARRAY_BUFFER, quadVboRef.current);
    const aQuadPosLoc = gl.getAttribLocation(prog, "a_quad_pos");
    gl.enableVertexAttribArray(aQuadPosLoc);
    gl.vertexAttribPointer(aQuadPosLoc, 2, gl.FLOAT, false, 2 * 4, 0);
    ext.vertexAttribDivisorANGLE(aQuadPosLoc, 0);

    // Bind Instance Attributes
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceVboRef.current);
    const stride = FLOATS_PER_SEGMENT_INSTANCE * 4;

    const aParentCol01Loc = gl.getAttribLocation(prog, "a_parent_mat_col01");
    const aParentCol2Loc = gl.getAttribLocation(prog, "a_parent_mat_col2");
    const aSegP0Loc = gl.getAttribLocation(prog, "a_seg_p0");
    const aSegP1Loc = gl.getAttribLocation(prog, "a_seg_p1");
    const aSegPropsLoc = gl.getAttribLocation(prog, "a_seg_props");
    const aSegColorLoc = gl.getAttribLocation(prog, "a_seg_color");
    const aSegHaloColorLoc = gl.getAttribLocation(prog, "a_seg_halo_color");

    if (aParentCol01Loc >= 0) {
      gl.enableVertexAttribArray(aParentCol01Loc);
      gl.vertexAttribPointer(aParentCol01Loc, 4, gl.FLOAT, false, stride, 0 * 4);
      ext.vertexAttribDivisorANGLE(aParentCol01Loc, 1);
    }

    if (aParentCol2Loc >= 0) {
      gl.enableVertexAttribArray(aParentCol2Loc);
      gl.vertexAttribPointer(aParentCol2Loc, 2, gl.FLOAT, false, stride, 4 * 4);
      ext.vertexAttribDivisorANGLE(aParentCol2Loc, 1);
    }

    if (aSegP0Loc >= 0) {
      gl.enableVertexAttribArray(aSegP0Loc);
      gl.vertexAttribPointer(aSegP0Loc, 2, gl.FLOAT, false, stride, 6 * 4);
      ext.vertexAttribDivisorANGLE(aSegP0Loc, 1);
    }

    if (aSegP1Loc >= 0) {
      gl.enableVertexAttribArray(aSegP1Loc);
      gl.vertexAttribPointer(aSegP1Loc, 2, gl.FLOAT, false, stride, 8 * 4);
      ext.vertexAttribDivisorANGLE(aSegP1Loc, 1);
    }

    if (aSegPropsLoc >= 0) {
      gl.enableVertexAttribArray(aSegPropsLoc);
      gl.vertexAttribPointer(aSegPropsLoc, 4, gl.FLOAT, false, stride, 10 * 4);
      ext.vertexAttribDivisorANGLE(aSegPropsLoc, 1);
    }

    if (aSegColorLoc >= 0) {
      gl.enableVertexAttribArray(aSegColorLoc);
      gl.vertexAttribPointer(aSegColorLoc, 4, gl.FLOAT, false, stride, 14 * 4);
      ext.vertexAttribDivisorANGLE(aSegColorLoc, 1);
    }

    if (aSegHaloColorLoc >= 0) {
      gl.enableVertexAttribArray(aSegHaloColorLoc);
      gl.vertexAttribPointer(aSegHaloColorLoc, 4, gl.FLOAT, false, stride, 18 * 4);
      ext.vertexAttribDivisorANGLE(aSegHaloColorLoc, 1);
    }

    // Execute Instanced Draw
    ext.drawArraysInstancedANGLE(gl.TRIANGLES, 0, 6, instanceCount);

    // Reset Divisors
    if (aParentCol01Loc >= 0) ext.vertexAttribDivisorANGLE(aParentCol01Loc, 0);
    if (aParentCol2Loc >= 0) ext.vertexAttribDivisorANGLE(aParentCol2Loc, 0);
    if (aSegP0Loc >= 0) ext.vertexAttribDivisorANGLE(aSegP0Loc, 0);
    if (aSegP1Loc >= 0) ext.vertexAttribDivisorANGLE(aSegP1Loc, 0);
    if (aSegPropsLoc >= 0) ext.vertexAttribDivisorANGLE(aSegPropsLoc, 0);
    if (aSegColorLoc >= 0) ext.vertexAttribDivisorANGLE(aSegColorLoc, 0);
    if (aSegHaloColorLoc >= 0) ext.vertexAttribDivisorANGLE(aSegHaloColorLoc, 0);
  }, [
    pcb,
    pan.x,
    pan.y,
    zoom,
    boardRotation,
    activeLayer,
    dimInactiveLayers,
    activeSelectedFpId,
    groupFpKey,
    containerWidth,
    containerHeight,
    activeSceneGraph,
    textFontMode,
    selection,
    groupSelected,
  ]);

  return (
    <canvas
      ref={canvasRef}
      id="pcb-webgl-silkscreen-canvas"
      className="absolute top-0 left-0 w-full h-full pointer-events-none"
      style={{ width: containerWidth, height: containerHeight }}
    />
  );
};
