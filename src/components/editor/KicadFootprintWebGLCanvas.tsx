import React, { useEffect, useRef } from "react";
import type { KicadFootprintModel } from "@/lib/kicad/footprint";
import { KicadFootprintRuntime } from "@/lib/kicad/footprint/kicadFootprintRuntime";
import { kicadGeometryEngine, type KicadGeometryItem } from "@/lib/kicad/footprint/geometry";
import { getKicadLayerColor } from "./KicadFootprintRenderer";
import { resolveKicadDisplayLayer, isKicadLayerVisible } from "@/lib/kicad/footprint/kicadLayerAdapter";
import { generateHersheyTextStrokes } from "@/lib/kicadHersheyFont";
import earcut from "earcut";

export interface WebGLBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

interface Props {
  footprint: KicadFootprintModel | KicadFootprintRuntime;
  bounds?: WebGLBounds;
  layerColors?: Record<string, string>;
  layerVisibility?: Record<string, boolean>;
  activeLayer?: string;
  dimInactiveLayers?: boolean;
  reference?: string;
  value?: string;
  className?: string;
  style?: React.CSSProperties;
}

// -----------------------------------------------------------------------------
// 1. FILL SHADERS (Component Body Fill, Pads, Drills, Solid Geometry)
// -----------------------------------------------------------------------------
const FILL_VS_SOURCE = `
attribute vec2 a_pos;
attribute vec4 a_color;

uniform vec4 u_bounds;     // (minX, minY, width, height)
uniform vec2 u_canvas_res; // (width_px, height_px)

varying vec4 v_color;

void main() {
  float minX = u_bounds.x;
  float minY = u_bounds.y;
  float bWidth = max(u_bounds.z, 0.0001);
  float bHeight = max(u_bounds.w, 0.0001);

  float normX = (a_pos.x - minX) / bWidth;
  float normY = (a_pos.y - minY) / bHeight;

  float clipX = normX * 2.0 - 1.0;
  float clipY = 1.0 - normY * 2.0; // Invert Y for display coords (top-down)

  gl_Position = vec4(clipX, clipY, 0.0, 1.0);
  v_color = a_color;
}
`;

const FILL_FS_SOURCE = `
precision highp float;

varying vec4 v_color;

void main() {
  if (v_color.a <= 0.001) discard;
  gl_FragColor = v_color;
}
`;

// -----------------------------------------------------------------------------
// 2. STROKE & TEXT SHADERS (Silkscreen Lines, Outlines, Hershey Vector Text)
// -----------------------------------------------------------------------------
const STROKE_VS_SOURCE = `
attribute vec2 a_quad_pos; // [-1, 1] unit quad corner
attribute vec2 a_seg_p0;   // Start point in footprint world space (mm)
attribute vec2 a_seg_p1;   // End point in footprint world space (mm)
attribute vec4 a_seg_props; // (strokeWidth, dimAlpha, isDashed, startLen)
attribute vec4 a_seg_color; // RGBA color

uniform vec4 u_bounds;     // (minX, minY, width, height)
uniform vec2 u_canvas_res; // (width_px, height_px)

varying vec2 v_world_pos;
varying vec2 v_world_p0;
varying vec2 v_world_p1;
varying vec4 v_props;
varying vec4 v_color;
varying float v_arcLength;

void main() {
  v_world_p0 = a_seg_p0;
  v_world_p1 = a_seg_p1;
  v_props = a_seg_props;
  v_color = a_seg_color;

  float width = a_seg_props.x;
  float radius = width * 0.5;

  float pixelSizeMm = u_bounds.z / max(u_canvas_res.x, 1.0);
  float aaPadding = pixelSizeMm * 1.5;
  float totalR = radius + aaPadding;

  vec2 dir = a_seg_p1 - a_seg_p0;
  float segLen = length(dir);
  vec2 u = segLen > 0.00001 ? (dir / segLen) : vec2(1.0, 0.0);
  vec2 n = vec2(-u.y, u.x);

  vec2 center = (a_seg_p0 + a_seg_p1) * 0.5;
  float halfLen = segLen * 0.5 + totalR;
  float halfWidth = totalR;

  vec2 world_pos = center + u * (a_quad_pos.x * halfLen) + n * (a_quad_pos.y * halfWidth);
  v_world_pos = world_pos;

  float localArcLen = dot(world_pos - a_seg_p0, u);
  v_arcLength = a_seg_props.w + localArcLen;

  // Transform world_pos (mm) to WebGL NDC [-1, 1]
  float minX = u_bounds.x;
  float minY = u_bounds.y;
  float bWidth = max(u_bounds.z, 0.0001);
  float bHeight = max(u_bounds.w, 0.0001);

  float normX = (world_pos.x - minX) / bWidth;
  float normY = (world_pos.y - minY) / bHeight;

  float clipX = normX * 2.0 - 1.0;
  float clipY = 1.0 - normY * 2.0;

  gl_Position = vec4(clipX, clipY, 0.0, 1.0);
}
`;

const STROKE_FS_SOURCE = `
precision highp float;

uniform vec4 u_bounds;
uniform vec2 u_canvas_res;
uniform float u_dashSize;  // Dash pattern cycle size in mm
uniform float u_dashRatio; // Dash stroke ratio (0.0 to 1.0)

varying vec2 v_world_pos;
varying vec2 v_world_p0;
varying vec2 v_world_p1;
varying vec4 v_props;
varying vec4 v_color;
varying float v_arcLength;

float sdCapsule(vec2 p, vec2 a, vec2 b, float r) {
  vec2 pa = p - a, ba = b - a;
  float l2 = dot(ba, ba);
  float h = l2 > 0.000001 ? clamp(dot(pa, ba) / l2, 0.0, 1.0) : 0.0;
  return length(pa - ba * h) - r;
}

void main() {
  float strokeWidth = v_props.x;
  float dimAlpha = v_props.y;
  float isDashed = v_props.z;
  float radius = strokeWidth * 0.5;

  // Procedural Dash Fragment Shader for Courtyard Boundaries (F.CrtYd)
  if (isDashed > 0.5) {
    float dashSize = u_dashSize > 0.001 ? u_dashSize : 0.5;
    float dashRatio = u_dashRatio > 0.001 ? u_dashRatio : 0.6;
    float pattern = fract(v_arcLength / dashSize);
    if (pattern > dashRatio) {
      discard; // Gap between dashes
    }
  }

  float dist = sdCapsule(v_world_pos, v_world_p0, v_world_p1, 0.0);

  float pixelSizeMm = u_bounds.z / max(u_canvas_res.x, 1.0);
  float halfPx = 0.5 * pixelSizeMm;

  float d_body = dist - radius;
  if (d_body > halfPx * 1.5) {
    discard;
  }

  float alpha = 1.0 - smoothstep(-halfPx, halfPx, d_body);
  if (alpha <= 0.001) discard;

  gl_FragColor = vec4(v_color.rgb, alpha * v_color.a * dimAlpha);
}
`;

function parseHexColor(hex: string): [number, number, number, number] {
  if (!hex) return [0.99, 0.88, 0.28, 1.0];
  if (hex.startsWith("rgba")) {
    const match = hex.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([\d.]+))?\s*\)/);
    if (match) {
      return [
        parseInt(match[1]) / 255,
        parseInt(match[2]) / 255,
        parseInt(match[3]) / 255,
        match[4] !== undefined ? parseFloat(match[4]) : 1.0,
      ];
    }
  }
  let c = hex.replace("#", "");
  if (c.length === 3) {
    c = c.split("").map((ch) => ch + ch).join("");
  }
  const num = parseInt(c, 16);
  if (isNaN(num)) return [0.99, 0.88, 0.28, 1.0];
  return [((num >> 16) & 255) / 255, ((num >> 8) & 255) / 255, (num & 255) / 255, 1.0];
}

interface SegmentQuadData {
  p0x: number;
  p0y: number;
  p1x: number;
  p1y: number;
  strokeWidth: number;
  dimAlpha: number;
  isDashed: number;
  startLen: number;
  r: number;
  g: number;
  b: number;
  a: number;
}

interface FillVertexData {
  x: number;
  y: number;
  r: number;
  g: number;
  b: number;
  a: number;
}

/**
 * Reconstruct a closed polygon boundary from a list of connected lines and arcs
 */
function extractClosedBoundaryFromFab(items: KicadGeometryItem[]): { x: number; y: number }[] | null {
  const segments: { start: { x: number; y: number }; end: { x: number; y: number }; arcPts?: { x: number; y: number }[] }[] = [];

  for (const item of items) {
    if (item.layer !== "F.Fab" && item.layer !== "top_fab") continue;
    const p = item.primitive;
    if (p.kind === "line") {
      segments.push({ start: p.start, end: p.end });
    } else if (p.kind === "arc") {
      // Sample arc into 6 points
      const pts: { x: number; y: number }[] = [];
      const steps = 6;
      const a0 = Math.atan2(p.start.y - p.center.y, p.start.x - p.center.x);
      const sweep = p.sweepRadians || 0;
      for (let i = 0; i <= steps; i++) {
        const a = a0 + (sweep * i) / steps;
        pts.push({ x: p.center.x + p.radius * Math.cos(a), y: p.center.y + p.radius * Math.sin(a) });
      }
      segments.push({ start: p.start, end: p.end, arcPts: pts });
    }
  }

  if (segments.length < 3) return null;

  // Chain segments into an ordered loop
  const polygon: { x: number; y: number }[] = [];
  const remaining = [...segments];
  const currentSeg = remaining.shift()!;
  if (currentSeg.arcPts) {
    polygon.push(...currentSeg.arcPts);
  } else {
    polygon.push(currentSeg.start, currentSeg.end);
  }

  const eps = 0.05;
  const maxIters = 60;
  let iters = 0;

  while (remaining.length > 0 && iters++ < maxIters) {
    const lastPt = polygon[polygon.length - 1];
    let nextIdx = -1;
    let reverse = false;

    for (let i = 0; i < remaining.length; i++) {
      const seg = remaining[i];
      if (Math.hypot(seg.start.x - lastPt.x, seg.start.y - lastPt.y) <= eps) {
        nextIdx = i;
        reverse = false;
        break;
      }
      if (Math.hypot(seg.end.x - lastPt.x, seg.end.y - lastPt.y) <= eps) {
        nextIdx = i;
        reverse = true;
        break;
      }
    }

    if (nextIdx === -1) break;

    const nextSeg = remaining.splice(nextIdx, 1)[0];
    if (nextSeg.arcPts) {
      const pts = reverse ? [...nextSeg.arcPts].reverse() : nextSeg.arcPts;
      polygon.push(...pts.slice(1));
    } else {
      polygon.push(reverse ? nextSeg.start : nextSeg.end);
    }
  }

  return polygon.length >= 3 ? polygon : null;
}

export function buildWebGLGeometryForFootprint(
  items: KicadGeometryItem[],
  reference: string,
  value: string,
  activeLayer: string,
  layerColors: Record<string, string>,
  layerVisibility: Record<string, boolean>,
  dimInactiveLayers: boolean
): {
  bodyFillVertices: FillVertexData[];
  padFillVertices: FillVertexData[];
  segments: SegmentQuadData[];
} {
  const bodyFillVertices: FillVertexData[] = [];
  const padFillVertices: FillVertexData[] = [];
  const segments: SegmentQuadData[] = [];

  let renderedRef = false;
  let renderedVal = false;

  const isSideLayer = (l: string) => l.startsWith("F.") || l.startsWith("B.");

  // Helper: Add triangle for filled geometry
  const addFillTriangle = (
    targetList: FillVertexData[],
    p0: { x: number; y: number },
    p1: { x: number; y: number },
    p2: { x: number; y: number },
    fr: number,
    fg: number,
    fb: number,
    fa: number
  ) => {
    targetList.push({ x: p0.x, y: p0.y, r: fr, g: fg, b: fb, a: fa });
    targetList.push({ x: p1.x, y: p1.y, r: fr, g: fg, b: fb, a: fa });
    targetList.push({ x: p2.x, y: p2.y, r: fr, g: fg, b: fb, a: fa });
  };

  // Helper: Add stroke segment
  const addSeg = (
    p0: { x: number; y: number },
    p1: { x: number; y: number },
    width = 0.12,
    startLen = 0,
    segR = 1.0,
    segG = 1.0,
    segB = 1.0,
    segA = 1.0,
    isDashed = 0.0,
    dimAlpha = 1.0
  ) => {
    segments.push({
      p0x: p0.x,
      p0y: p0.y,
      p1x: p1.x,
      p1y: p1.y,
      strokeWidth: width,
      dimAlpha,
      isDashed,
      startLen,
      r: segR,
      g: segG,
      b: segB,
      a: segA,
    });
    return Math.hypot(p1.x - p0.x, p1.y - p0.y);
  };

  // =========================================================================
  // 1. COMPONENT BODY FILL (Drawn in background layer of footprint)
  // =========================================================================
  const allowBodyFill =
    layerVisibility["body fill"] !== false &&
    layerVisibility["body_fill"] !== false &&
    layerVisibility["F.fab"] !== false;

  if (allowBodyFill) {
    // Elegant Dark Slate/Charcoal IC Package body color: rgba(24, 34, 52, 0.75)
    const bodyR = 0.10;
    const bodyG = 0.14;
    const bodyB = 0.22;
    const bodyA = 0.75;

    let bodyFilled = false;

    // Strategy 1: Check for explicit closed polygons or solid shapes on F.Fab
    for (const item of items) {
      if (item.layer !== "F.Fab" && item.layer !== "top_fab") continue;
      const p = item.primitive;
      if (p.kind === "polygon" && p.points && p.points.length >= 3) {
        const flatCoords: number[] = [];
        for (const pt of p.points) flatCoords.push(pt.x, pt.y);
        const triIndices = earcut(flatCoords);
        for (let i = 0; i < triIndices.length; i += 3) {
          addFillTriangle(
            bodyFillVertices,
            p.points[triIndices[i]],
            p.points[triIndices[i + 1]],
            p.points[triIndices[i + 2]],
            bodyR,
            bodyG,
            bodyB,
            bodyA
          );
        }
        bodyFilled = true;
      } else if (p.kind === "rect") {
        const x1 = Math.min(p.start.x, p.end.x);
        const x2 = Math.max(p.start.x, p.end.x);
        const y1 = Math.min(p.start.y, p.end.y);
        const y2 = Math.max(p.start.y, p.end.y);
        let pts = [
          { x: x1, y: y1 },
          { x: x2, y: y1 },
          { x: x2, y: y2 },
          { x: x1, y: y2 },
        ];
        if (p.rotation) {
          const cx = (x1 + x2) / 2;
          const cy = (y1 + y2) / 2;
          const rad = (p.rotation * Math.PI) / 180;
          const cos = Math.cos(rad);
          const sin = Math.sin(rad);
          pts = pts.map((pt) => {
            const dx = pt.x - cx;
            const dy = pt.y - cy;
            return { x: cx + dx * cos - dy * sin, y: cy + dx * sin + dy * cos };
          });
        }
        addFillTriangle(bodyFillVertices, pts[0], pts[1], pts[2], bodyR, bodyG, bodyB, bodyA);
        addFillTriangle(bodyFillVertices, pts[0], pts[2], pts[3], bodyR, bodyG, bodyB, bodyA);
        bodyFilled = true;
      } else if (p.kind === "circle") {
        const steps = 32;
        const cx = p.center.x;
        const cy = p.center.y;
        const rad = p.radius;
        const cCenter = { x: cx, y: cy };
        for (let i = 0; i < steps; i++) {
          const a0 = (i / steps) * Math.PI * 2;
          const a1 = ((i + 1) / steps) * Math.PI * 2;
          addFillTriangle(
            bodyFillVertices,
            cCenter,
            { x: cx + Math.cos(a0) * rad, y: cy + Math.sin(a0) * rad },
            { x: cx + Math.cos(a1) * rad, y: cy + Math.sin(a1) * rad },
            bodyR,
            bodyG,
            bodyB,
            bodyA
          );
        }
        bodyFilled = true;
      }
    }

    // Strategy 2: If F.Fab is defined as a series of connected lines/arcs (e.g. DIP, SOIC, QFP, Resistors)
    if (!bodyFilled) {
      const fabPolygon = extractClosedBoundaryFromFab(items);
      if (fabPolygon && fabPolygon.length >= 3) {
        const flatCoords: number[] = [];
        for (const pt of fabPolygon) flatCoords.push(pt.x, pt.y);
        const triIndices = earcut(flatCoords);
        for (let i = 0; i < triIndices.length; i += 3) {
          addFillTriangle(
            bodyFillVertices,
            fabPolygon[triIndices[i]],
            fabPolygon[triIndices[i + 1]],
            fabPolygon[triIndices[i + 2]],
            bodyR,
            bodyG,
            bodyB,
            bodyA
          );
        }
        bodyFilled = true;
      }
    }

    // Strategy 3: Synthesize body fill between pads if no Fab outline was closed
    if (!bodyFilled) {
      const padItemsList = items.filter((i) => i.source === "pad" || i.source === "custom-pad");
      if (padItemsList.length >= 2) {
        const padBounds = kicadGeometryEngine.bounds(padItemsList);
        const w = padBounds.maxX - padBounds.minX;
        const h = padBounds.maxY - padBounds.minY;
        const insetX = Math.min(0.2, w * 0.1);
        const insetY = Math.min(0.2, h * 0.1);
        const bx1 = padBounds.minX + insetX;
        const by1 = padBounds.minY + insetY;
        const bx2 = padBounds.maxX - insetX;
        const by2 = padBounds.maxY - insetY;
        if (bx2 > bx1 && by2 > by1) {
          addFillTriangle(bodyFillVertices, { x: bx1, y: by1 }, { x: bx2, y: by1 }, { x: bx2, y: by2 }, bodyR, bodyG, bodyB, bodyA);
          addFillTriangle(bodyFillVertices, { x: bx1, y: by1 }, { x: bx2, y: by2 }, { x: bx1, y: by2 }, bodyR, bodyG, bodyB, bodyA);
        }
      }
    }
  }

  // =========================================================================
  // 2. COPPER PADS, DRILLS, SILKSCREEN & VECTOR TEXT
  // =========================================================================
  for (const item of items) {
    const displayLayer = resolveKicadDisplayLayer(item.layer, activeLayer);

    const isFab = displayLayer === "top_fab" || displayLayer === "bottom_fab" || item.layer === "F.Fab" || item.layer === "B.Fab";
    const isSilkscreen = displayLayer === "silkscreen" || displayLayer === "bottom_silkscreen" || item.layer === "F.SilkS" || item.layer === "B.SilkS";
    const isCourtyard = displayLayer === "top_courtyard" || displayLayer === "bottom_courtyard" || item.layer === "F.CrtYd" || item.layer === "B.CrtYd" || item.layer === "courtyard" || item.layer === "F.courtyard";
    const isCopper = displayLayer === "top_copper" || displayLayer === "bottom_copper" || item.layer === "F.Cu" || item.layer === "B.Cu" || item.layer === "*.Cu";
    const isDrill = item.layer === "drill" || item.source === "drill";

    // Visibility layer filtering
    if (isSilkscreen && (layerVisibility["F.silkscreen"] === false || layerVisibility["silkscreen"] === false || layerVisibility["F.SilkS"] === false)) continue;
    if (isFab && (layerVisibility["F.fab"] === false || layerVisibility["top_fab"] === false || layerVisibility["F.Fab"] === false)) continue;
    if (isCourtyard && (layerVisibility["F.courtyard"] === false || layerVisibility["top_courtyard"] === false || layerVisibility["F.CrtYd"] === false)) continue;
    if (isCopper && (layerVisibility["F.Cu"] === false || layerVisibility["top_copper"] === false)) continue;

    const inactiveSide = isSideLayer(item.layer) && ((item.layer.startsWith("F.") && activeLayer !== "top_copper") || (item.layer.startsWith("B.") && activeLayer !== "bottom_copper"));
    const dimAlpha = dimInactiveLayers && inactiveSide ? 0.35 : 1.0;

    const colorHex = getKicadLayerColor(displayLayer, layerColors);
    const [r, g, b, a] = parseHexColor(colorHex);

    const p = item.primitive;
    const defaultWidth = isSilkscreen ? 0.12 : isFab ? 0.10 : isCourtyard ? 0.05 : 0.12;
    const strokeW = p.stroke?.width ? Math.max(0.05, p.stroke.width) : defaultWidth;
    const isDashed = isCourtyard ? 1.0 : 0.0;

    // -------------------------------------------------------------------------
    // A. COPPER PADS & DRILL HOLES (Middle Layer)
    // -------------------------------------------------------------------------
    if (isCopper || isDrill) {
      const copperA = (isDrill ? 1.0 : 0.95) * dimAlpha;
      const copperR = isDrill ? 0.04 : r;
      const copperG = isDrill ? 0.06 : g;
      const copperB = isDrill ? 0.12 : b;

      switch (p.kind) {
        case "rect": {
          const x1 = Math.min(p.start.x, p.end.x);
          const x2 = Math.max(p.start.x, p.end.x);
          const y1 = Math.min(p.start.y, p.end.y);
          const y2 = Math.max(p.start.y, p.end.y);
          let pts = [
            { x: x1, y: y1 },
            { x: x2, y: y1 },
            { x: x2, y: y2 },
            { x: x1, y: y2 },
          ];
          if (p.rotation) {
            const cx = (x1 + x2) / 2;
            const cy = (y1 + y2) / 2;
            const rad = (p.rotation * Math.PI) / 180;
            const cos = Math.cos(rad);
            const sin = Math.sin(rad);
            pts = pts.map((pt) => {
              const dx = pt.x - cx;
              const dy = pt.y - cy;
              return { x: cx + dx * cos - dy * sin, y: cy + dx * sin + dy * cos };
            });
          }
          addFillTriangle(padFillVertices, pts[0], pts[1], pts[2], copperR, copperG, copperB, copperA);
          addFillTriangle(padFillVertices, pts[0], pts[2], pts[3], copperR, copperG, copperB, copperA);
          break;
        }
        case "roundrect": {
          const cx = p.center.x, cy = p.center.y;
          const hx = p.size.x / 2, hy = p.size.y / 2;
          const rad = p.radii?.topLeft || Math.min(hx, hy) * 0.25;
          const steps = 6;
          const corners = [
            { cx: cx + hx - rad, cy: cy - hy + rad, startA: -Math.PI / 2, endA: 0 },
            { cx: cx + hx - rad, cy: cy + hy - rad, startA: 0, endA: Math.PI / 2 },
            { cx: cx - hx + rad, cy: cy + hy - rad, startA: Math.PI / 2, endA: Math.PI },
            { cx: cx - hx + rad, cy: cy - hy + rad, startA: Math.PI, endA: (3 * Math.PI) / 2 },
          ];
          let allPts: { x: number; y: number }[] = [];
          for (const c of corners) {
            for (let i = 0; i <= steps; i++) {
              const ang = c.startA + (i / steps) * (c.endA - c.startA);
              allPts.push({ x: c.cx + Math.cos(ang) * rad, y: c.cy + Math.sin(ang) * rad });
            }
          }
          if (p.rotation) {
            const radRot = (p.rotation * Math.PI) / 180;
            const cosR = Math.cos(radRot);
            const sinR = Math.sin(radRot);
            allPts = allPts.map((pt) => {
              const dx = pt.x - cx;
              const dy = pt.y - cy;
              return { x: cx + dx * cosR - dy * sinR, y: cy + dx * sinR + dy * cosR };
            });
          }
          const cCenter = { x: cx, y: cy };
          for (let i = 0; i < allPts.length; i++) {
            addFillTriangle(padFillVertices, cCenter, allPts[i], allPts[(i + 1) % allPts.length], copperR, copperG, copperB, copperA);
          }
          break;
        }
        case "chamferrect": {
          const cx = p.center.x, cy = p.center.y;
          const hx = p.size.x / 2, hy = p.size.y / 2;
          const c = p.chamfers || {};
          let pts = [
            { x: cx - hx + (c.topLeft || 0), y: cy - hy },
            { x: cx + hx - (c.topRight || 0), y: cy - hy },
            { x: cx + hx, y: cy - hy + (c.topRight || 0) },
            { x: cx + hx, y: cy + hy - (c.bottomRight || 0) },
            { x: cx + hx - (c.bottomRight || 0), y: cy + hy },
            { x: cx - hx + (c.bottomLeft || 0), y: cy + hy },
            { x: cx - hx, y: cy + hy - (c.bottomLeft || 0) },
            { x: cx - hx, y: cy - hy + (c.topLeft || 0) },
          ];
          if (p.rotation) {
            const radRot = (p.rotation * Math.PI) / 180;
            const cosR = Math.cos(radRot);
            const sinR = Math.sin(radRot);
            pts = pts.map((pt) => {
              const dx = pt.x - cx;
              const dy = pt.y - cy;
              return { x: cx + dx * cosR - dy * sinR, y: cy + dx * sinR + dy * cosR };
            });
          }
          const cCenter = { x: cx, y: cy };
          for (let i = 0; i < pts.length; i++) {
            addFillTriangle(padFillVertices, cCenter, pts[i], pts[(i + 1) % pts.length], copperR, copperG, copperB, copperA);
          }
          break;
        }
        case "circle": {
          const steps = 32;
          const cx = p.center.x, cy = p.center.y, rad = p.radius;
          const cCenter = { x: cx, y: cy };
          for (let i = 0; i < steps; i++) {
            const a0 = (i / steps) * Math.PI * 2;
            const a1 = ((i + 1) / steps) * Math.PI * 2;
            addFillTriangle(
              padFillVertices,
              cCenter,
              { x: cx + Math.cos(a0) * rad, y: cy + Math.sin(a0) * rad },
              { x: cx + Math.cos(a1) * rad, y: cy + Math.sin(a1) * rad },
              copperR,
              copperG,
              copperB,
              copperA
            );
          }
          break;
        }
        case "hole": {
          const cx = p.center.x, cy = p.center.y;
          const hx = p.size.x / 2, hy = p.size.y / 2;
          const isSlot = Math.abs(hx - hy) > 0.001;
          const rad = Math.min(hx, hy);
          const cCenter = { x: cx, y: cy };

          if (isSlot) {
            const halfStraight = Math.max(0, Math.max(hx, hy) - rad);
            const horizontal = hx >= hy;
            let p0 = horizontal ? { x: cx - halfStraight, y: cy } : { x: cx, y: cy - halfStraight };
            let p1 = horizontal ? { x: cx + halfStraight, y: cy } : { x: cx, y: cy + halfStraight };
            if (p.rotation) {
              const radRot = (p.rotation * Math.PI) / 180;
              const cosR = Math.cos(radRot), sinR = Math.sin(radRot);
              const rot = (pt: { x: number; y: number }) => ({
                x: cx + (pt.x - cx) * cosR - (pt.y - cy) * sinR,
                y: cy + (pt.x - cx) * sinR + (pt.y - cy) * cosR,
              });
              p0 = rot(p0);
              p1 = rot(p1);
            }
            const dir = { x: p1.x - p0.x, y: p1.y - p0.y };
            const len = Math.hypot(dir.x, dir.y);
            const u = len > 0.0001 ? { x: dir.x / len, y: dir.y / len } : { x: 1, y: 0 };
            const n = { x: -u.y, y: u.x };

            const quad0 = { x: p0.x + n.x * rad, y: p0.y + n.y * rad };
            const quad1 = { x: p1.x + n.x * rad, y: p1.y + n.y * rad };
            const quad2 = { x: p1.x - n.x * rad, y: p1.y - n.y * rad };
            const quad3 = { x: p0.x - n.x * rad, y: p0.y - n.y * rad };

            addFillTriangle(padFillVertices, quad0, quad1, quad2, 0.04, 0.06, 0.12, 1.0);
            addFillTriangle(padFillVertices, quad0, quad2, quad3, 0.04, 0.06, 0.12, 1.0);

            const steps = 12;
            for (let i = 0; i < steps; i++) {
              const ang0 = -Math.PI / 2 + (i / steps) * Math.PI;
              const ang1 = -Math.PI / 2 + ((i + 1) / steps) * Math.PI;
              addFillTriangle(
                padFillVertices,
                p1,
                { x: p1.x + u.x * Math.cos(ang0) * rad + n.x * Math.sin(ang0) * rad, y: p1.y + u.y * Math.cos(ang0) * rad + n.y * Math.sin(ang0) * rad },
                { x: p1.x + u.x * Math.cos(ang1) * rad + n.x * Math.sin(ang1) * rad, y: p1.y + u.y * Math.cos(ang1) * rad + n.y * Math.sin(ang1) * rad },
                0.04,
                0.06,
                0.12,
                1.0
              );
              addFillTriangle(
                padFillVertices,
                p0,
                { x: p0.x - u.x * Math.cos(ang0) * rad + n.x * Math.sin(ang0) * rad, y: p0.y - u.y * Math.cos(ang0) * rad + n.y * Math.sin(ang0) * rad },
                { x: p0.x - u.x * Math.cos(ang1) * rad + n.x * Math.sin(ang1) * rad, y: p0.y - u.y * Math.cos(ang1) * rad + n.y * Math.sin(ang1) * rad },
                0.04,
                0.06,
                0.12,
                1.0
              );
            }
          } else {
            const steps = 32;
            for (let i = 0; i < steps; i++) {
              const a0 = (i / steps) * Math.PI * 2;
              const a1 = ((i + 1) / steps) * Math.PI * 2;
              addFillTriangle(
                padFillVertices,
                cCenter,
                { x: cx + Math.cos(a0) * rad, y: cy + Math.sin(a0) * rad },
                { x: cx + Math.cos(a1) * rad, y: cy + Math.sin(a1) * rad },
                0.04,
                0.06,
                0.12,
                1.0
              );
            }
          }
          break;
        }
        case "capsule": {
          const rad = p.radius;
          const dir = { x: p.end.x - p.start.x, y: p.end.y - p.start.y };
          const len = Math.hypot(dir.x, dir.y);
          if (len < 0.0001) {
            // Circular pad (equal width & height)
            const steps = 32;
            const cx = p.start.x;
            const cy = p.start.y;
            const cCenter = { x: cx, y: cy };
            for (let i = 0; i < steps; i++) {
              const a0 = (i / steps) * Math.PI * 2;
              const a1 = ((i + 1) / steps) * Math.PI * 2;
              addFillTriangle(
                padFillVertices,
                cCenter,
                { x: cx + Math.cos(a0) * rad, y: cy + Math.sin(a0) * rad },
                { x: cx + Math.cos(a1) * rad, y: cy + Math.sin(a1) * rad },
                copperR,
                copperG,
                copperB,
                copperA
              );
            }
          } else {
            // Elongated oval / capsule pad
            const u = { x: dir.x / len, y: dir.y / len };
            const n = { x: -u.y, y: u.x };

            const p0 = { x: p.start.x + n.x * rad, y: p.start.y + n.y * rad };
            const p1 = { x: p.end.x + n.x * rad, y: p.end.y + n.y * rad };
            const p2 = { x: p.end.x - n.x * rad, y: p.end.y - n.y * rad };
            const p3 = { x: p.start.x - n.x * rad, y: p.start.y - n.y * rad };

            addFillTriangle(padFillVertices, p0, p1, p2, copperR, copperG, copperB, copperA);
            addFillTriangle(padFillVertices, p0, p2, p3, copperR, copperG, copperB, copperA);

            const steps = 16;
            for (let i = 0; i < steps; i++) {
              const ang0 = -Math.PI / 2 + (i / steps) * Math.PI;
              const ang1 = -Math.PI / 2 + ((i + 1) / steps) * Math.PI;
              // End cap semicircle
              addFillTriangle(
                padFillVertices,
                p.end,
                { x: p.end.x + u.x * Math.cos(ang0) * rad + n.x * Math.sin(ang0) * rad, y: p.end.y + u.y * Math.cos(ang0) * rad + n.y * Math.sin(ang0) * rad },
                { x: p.end.x + u.x * Math.cos(ang1) * rad + n.x * Math.sin(ang1) * rad, y: p.end.y + u.y * Math.cos(ang1) * rad + n.y * Math.sin(ang1) * rad },
                copperR,
                copperG,
                copperB,
                copperA
              );
              // Start cap semicircle
              addFillTriangle(
                padFillVertices,
                p.start,
                { x: p.start.x - u.x * Math.cos(ang0) * rad + n.x * Math.sin(ang0) * rad, y: p.start.y - u.y * Math.cos(ang0) * rad + n.y * Math.sin(ang0) * rad },
                { x: p.start.x - u.x * Math.cos(ang1) * rad + n.x * Math.sin(ang1) * rad, y: p.start.y - u.y * Math.cos(ang1) * rad + n.y * Math.sin(ang1) * rad },
                copperR,
                copperG,
                copperB,
                copperA
              );
            }
          }
          break;
        }
        case "polygon": {
          if (p.points && p.points.length >= 3) {
            const flatCoords: number[] = [];
            for (const pt of p.points) flatCoords.push(pt.x, pt.y);
            const triIndices = earcut(flatCoords);
            for (let i = 0; i < triIndices.length; i += 3) {
              addFillTriangle(
                padFillVertices,
                p.points[triIndices[i]],
                p.points[triIndices[i + 1]],
                p.points[triIndices[i + 2]],
                copperR,
                copperG,
                copperB,
                copperA
              );
            }
          }
          break;
        }
      }

      // Render crisp vector pad numbers on pads
      if (
        (item.source === "pad" || item.source === "custom-pad") &&
        item.metadata?.padNumber &&
        layerVisibility["pad_numbers"] !== false &&
        layerVisibility["padNumbers"] !== false
      ) {
        const padNumStr = String(item.metadata.padNumber);
        let padCx = 0,
          padCy = 0,
          padDim = 1.0;
        if ("center" in p && p.center) {
          padCx = p.center.x;
          padCy = p.center.y;
          if ("size" in p && p.size) padDim = Math.min(p.size.x, p.size.y);
          else if ("radius" in p && p.radius) padDim = p.radius * 2;
        } else if ("start" in p && "end" in p) {
          padCx = (p.start.x + p.end.x) / 2;
          padCy = (p.start.y + p.end.y) / 2;
          const len = Math.hypot(p.end.x - p.start.x, p.end.y - p.start.y);
          const rad = "radius" in p && p.radius ? p.radius : 0.4;
          padDim = Math.min(Math.max(len, rad * 2), rad * 2);
        } else if ("points" in p && p.points && p.points.length > 0) {
          padCx = p.points.reduce((acc, pt) => acc + pt.x, 0) / p.points.length;
          padCy = p.points.reduce((acc, pt) => acc + pt.y, 0) / p.points.length;
        }

        const numSize = Math.max(0.35, Math.min(0.85, padDim * 0.45));
        const numThickness = Math.max(0.06, numSize * 0.16);

        const fontStrokes = generateHersheyTextStrokes({
          text: padNumStr,
          x: padCx,
          y: padCy,
          size: numSize,
          thickness: numThickness,
          rotation: 0,
          layer: "drill",
          justify: "center",
          verticalAlign: "middle",
        });
        for (const stroke of fontStrokes) {
          addSeg(stroke.p0, stroke.p1, stroke.strokeWidth || numThickness, 0, 1.0, 1.0, 1.0, 0.95, 0.0, dimAlpha);
        }
      }
    }

    // -------------------------------------------------------------------------
    // B. SILKSCREEN, FAB OUTLINES & PROCEDURAL COURTYARD (Strokes)
    // -------------------------------------------------------------------------
    if (isSilkscreen || isFab || isCourtyard) {
      switch (p.kind) {
        case "line": {
          addSeg(p.start, p.end, strokeW, 0, r, g, b, a, isDashed, dimAlpha);
          break;
        }
        case "rect": {
          const x1 = Math.min(p.start.x, p.end.x);
          const x2 = Math.max(p.start.x, p.end.x);
          const y1 = Math.min(p.start.y, p.end.y);
          const y2 = Math.max(p.start.y, p.end.y);
          let pts = [
            { x: x1, y: y1 },
            { x: x2, y: y1 },
            { x: x2, y: y2 },
            { x: x1, y: y2 },
          ];
          if (p.rotation) {
            const cx = (x1 + x2) / 2;
            const cy = (y1 + y2) / 2;
            const rad = (p.rotation * Math.PI) / 180;
            const cos = Math.cos(rad);
            const sin = Math.sin(rad);
            pts = pts.map((pt) => {
              const dx = pt.x - cx;
              const dy = pt.y - cy;
              return { x: cx + dx * cos - dy * sin, y: cy + dx * sin + dy * cos };
            });
          }
          let len = 0;
          len += addSeg(pts[0], pts[1], strokeW, len, r, g, b, a, isDashed, dimAlpha);
          len += addSeg(pts[1], pts[2], strokeW, len, r, g, b, a, isDashed, dimAlpha);
          len += addSeg(pts[2], pts[3], strokeW, len, r, g, b, a, isDashed, dimAlpha);
          addSeg(pts[3], pts[0], strokeW, len, r, g, b, a, isDashed, dimAlpha);
          break;
        }
        case "circle": {
          const steps = 36;
          const cx = p.center.x, cy = p.center.y, rad = p.radius;
          let len = 0;
          for (let i = 0; i < steps; i++) {
            const a0 = (i / steps) * Math.PI * 2;
            const a1 = ((i + 1) / steps) * Math.PI * 2;
            const p0 = { x: cx + Math.cos(a0) * rad, y: cy + Math.sin(a0) * rad };
            const p1 = { x: cx + Math.cos(a1) * rad, y: cy + Math.sin(a1) * rad };
            len += addSeg(p0, p1, strokeW, len, r, g, b, a, isDashed, dimAlpha);
          }
          break;
        }
        case "arc": {
          const steps = 24;
          const cx = p.center.x, cy = p.center.y, rad = p.radius;
          const startA = Math.atan2(p.start.y - cy, p.start.x - cx);
          const sweep = p.sweepRadians || 0;
          let len = 0;
          for (let i = 0; i < steps; i++) {
            const a0 = startA + (i / steps) * sweep;
            const a1 = startA + ((i + 1) / steps) * sweep;
            const p0 = { x: cx + Math.cos(a0) * rad, y: cy + Math.sin(a0) * rad };
            const p1 = { x: cx + Math.cos(a1) * rad, y: cy + Math.sin(a1) * rad };
            len += addSeg(p0, p1, strokeW, len, r, g, b, a, isDashed, dimAlpha);
          }
          break;
        }
        case "polygon": {
          if (p.points && p.points.length > 1) {
            let len = 0;
            for (let i = 0; i < p.points.length; i++) {
              const p0 = p.points[i];
              const p1 = p.points[(i + 1) % p.points.length];
              len += addSeg(p0, p1, strokeW, len, r, g, b, a, isDashed, dimAlpha);
            }
          }
          break;
        }
        case "text": {
          const isRef = p.role === "reference" || p.text === "REF**" || p.text === "${REFERENCE}" || p.text === "%R";
          const isVal = p.role === "value" || p.text === "VAL**" || p.text === "${VALUE}" || p.text === "%V" || p.text === value;

          if (isRef && layerVisibility["Reference"] === false) break;
          if (isVal && layerVisibility["Value"] === false) break;

          let displayStr = p.text || "";
          if (isRef) {
            if (renderedRef) break;
            displayStr = reference || "REF**";
            renderedRef = true;
          } else if (isVal) {
            if (renderedVal) break;
            displayStr = value || "VAL**";
            renderedVal = true;
          }

          const fontStrokes = generateHersheyTextStrokes({
            text: displayStr,
            x: p.position.x,
            y: p.position.y,
            size: p.size?.y || 1.0,
            thickness: p.thickness || 0.15,
            rotation: p.rotation || 0,
            layer: item.layer,
            justify: p.anchor === "start" ? "left" : p.anchor === "end" ? "right" : "center",
            verticalAlign: "middle",
          });

          let textR = r, textG = g, textB = b;
          if (isRef) {
            const silkColorHex = getKicadLayerColor("silkscreen", layerColors);
            const [sr, sg, sb] = parseHexColor(silkColorHex);
            textR = sr; textG = sg; textB = sb;
          } else if (isVal) {
            const fabColorHex = getKicadLayerColor("top_fab", layerColors);
            const [fr, fg, fb] = parseHexColor(fabColorHex);
            textR = fr; textG = fg; textB = fb;
          }

          for (const stroke of fontStrokes) {
            addSeg(stroke.p0, stroke.p1, stroke.strokeWidth || p.thickness || 0.15, 0, textR, textG, textB, 1.0, 0.0, dimAlpha);
          }
          break;
        }
      }
    }
  }

  return {
    bodyFillVertices,
    padFillVertices,
    segments,
  };
}

export function KicadFootprintWebGLCanvas({
  footprint,
  bounds,
  layerColors = {},
  layerVisibility = {},
  activeLayer = "top_copper",
  dimInactiveLayers = false,
  reference = "REF**",
  value = footprint?.name || "",
  className = "w-full h-full",
  style,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !footprint) return;

    const gl = canvas.getContext("webgl", { alpha: true, antialias: true, premultipliedAlpha: false });
    if (!gl) return;

    const createShader = (type: number, src: string) => {
      const shader = gl.createShader(type)!;
      gl.shaderSource(shader, src);
      gl.compileShader(shader);
      return shader;
    };

    const strokeVert = createShader(gl.VERTEX_SHADER, STROKE_VS_SOURCE);
    const strokeFrag = createShader(gl.FRAGMENT_SHADER, STROKE_FS_SOURCE);
    const strokeProgram = gl.createProgram()!;
    gl.attachShader(strokeProgram, strokeVert);
    gl.attachShader(strokeProgram, strokeFrag);
    gl.linkProgram(strokeProgram);

    const fillVert = createShader(gl.VERTEX_SHADER, FILL_VS_SOURCE);
    const fillFrag = createShader(gl.FRAGMENT_SHADER, FILL_FS_SOURCE);
    const fillProgram = gl.createProgram()!;
    gl.attachShader(fillProgram, fillVert);
    gl.attachShader(fillProgram, fillFrag);
    gl.linkProgram(fillProgram);

    let items: KicadGeometryItem[] = [];
    if (typeof (footprint as any).GetWorldGeometry === "function") {
      items = (footprint as any).GetWorldGeometry() || [];
    } else {
      const local = kicadGeometryEngine.buildFootprint(footprint.graphics || [], footprint.pads || []);
      items = kicadGeometryEngine.transformed(local, {
        position: footprint.position || { x: 0, y: 0 },
        rotation: footprint.rotation || 0,
        scaleX: 1,
        scaleY: 1,
        flipped: footprint.layer === "B.Cu",
      });
    }

    const { bodyFillVertices, padFillVertices, segments } = buildWebGLGeometryForFootprint(
      items,
      reference,
      value,
      activeLayer,
      layerColors,
      layerVisibility,
      dimInactiveLayers
    );

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const widthPx = Math.max(1, Math.floor(rect.width * dpr));
    const heightPx = Math.max(1, Math.floor(rect.height * dpr));
    canvas.width = widthPx;
    canvas.height = heightPx;

    gl.viewport(0, 0, widthPx, heightPx);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Compute aspect-ratio corrected isometric bounds
    const canvasAspect = widthPx / heightPx;
    let b = bounds;
    if (!b) {
      const computed = kicadGeometryEngine.bounds(items);
      const w = Math.max(1.8, computed.maxX - computed.minX);
      const h = Math.max(1.8, computed.maxY - computed.minY);
      const pad = Math.max(0.8, Math.max(w, h) * 0.12);
      b = {
        minX: computed.minX - pad,
        minY: computed.minY - pad,
        maxX: computed.maxX + pad,
        maxY: computed.maxY + pad,
      };
    }

    const bWidthRaw = Math.max(b.maxX - b.minX, 0.001);
    const bHeightRaw = Math.max(b.maxY - b.minY, 0.001);
    const bCenterX = (b.minX + b.maxX) / 2;
    const bCenterY = (b.minY + b.maxY) / 2;
    const boundsAspect = bWidthRaw / bHeightRaw;

    let bWidth = bWidthRaw;
    let bHeight = bHeightRaw;
    let minX = b.minX;
    let minY = b.minY;

    if (canvasAspect > boundsAspect) {
      bWidth = bHeightRaw * canvasAspect;
      minX = bCenterX - bWidth / 2;
    } else {
      bHeight = bWidthRaw / canvasAspect;
      minY = bCenterY - bHeight / 2;
    }

    // =========================================================================
    // 1. DRAW BODY FILL & COPPER PADS (TRIANGLES)
    // =========================================================================
    const maxAttribs = gl.getParameter(gl.MAX_VERTEX_ATTRIBS) || 16;
    const disableAllAttribs = () => {
      for (let i = 0; i < maxAttribs; i++) {
        gl.disableVertexAttribArray(i);
      }
    };

    // Drawing order: Body Fill (Background) first, then Pads
    const allFillVertices = [...bodyFillVertices, ...padFillVertices];
    let fillBuffer: WebGLBuffer | null = null;

    if (allFillVertices.length > 0 && gl.getProgramParameter(fillProgram, gl.LINK_STATUS)) {
      gl.useProgram(fillProgram);
      disableAllAttribs();

      const fillFloatsPerVertex = 6;
      const fillData = new Float32Array(allFillVertices.length * fillFloatsPerVertex);
      let fOffset = 0;
      for (const v of allFillVertices) {
        fillData[fOffset++] = v.x;
        fillData[fOffset++] = v.y;
        fillData[fOffset++] = v.r;
        fillData[fOffset++] = v.g;
        fillData[fOffset++] = v.b;
        fillData[fOffset++] = v.a;
      }

      fillBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, fillBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, fillData, gl.STATIC_DRAW);

      const aPosLoc = gl.getAttribLocation(fillProgram, "a_pos");
      const aColorLoc = gl.getAttribLocation(fillProgram, "a_color");
      const uBoundsLocF = gl.getUniformLocation(fillProgram, "u_bounds");
      const uResLocF = gl.getUniformLocation(fillProgram, "u_canvas_res");

      if (aPosLoc >= 0) {
        gl.enableVertexAttribArray(aPosLoc);
        gl.vertexAttribPointer(aPosLoc, 2, gl.FLOAT, false, fillFloatsPerVertex * 4, 0);
      }
      if (aColorLoc >= 0) {
        gl.enableVertexAttribArray(aColorLoc);
        gl.vertexAttribPointer(aColorLoc, 4, gl.FLOAT, false, fillFloatsPerVertex * 4, 2 * 4);
      }

      gl.uniform4f(uBoundsLocF, minX, minY, bWidth, bHeight);
      gl.uniform2f(uResLocF, widthPx, heightPx);

      gl.drawArrays(gl.TRIANGLES, 0, allFillVertices.length);
      disableAllAttribs();
    }

    // =========================================================================
    // 2. DRAW SILKSCREEN, OUTLINES & HERSHEY TEXT STROKES (QUADS)
    // =========================================================================
    let strokeBuffer: WebGLBuffer | null = null;
    if (segments.length > 0 && gl.getProgramParameter(strokeProgram, gl.LINK_STATUS)) {
      gl.useProgram(strokeProgram);
      disableAllAttribs();

      const unitQuad = [
        [-1, -1],
        [1, -1],
        [-1, 1],
        [-1, 1],
        [1, -1],
        [1, 1],
      ];

      const floatsPerVertex = 14;
      const vertexData = new Float32Array(segments.length * 6 * floatsPerVertex);

      let offset = 0;
      for (const seg of segments) {
        for (const q of unitQuad) {
          vertexData[offset++] = q[0];
          vertexData[offset++] = q[1];
          vertexData[offset++] = seg.p0x;
          vertexData[offset++] = seg.p0y;
          vertexData[offset++] = seg.p1x;
          vertexData[offset++] = seg.p1y;
          vertexData[offset++] = seg.strokeWidth;
          vertexData[offset++] = seg.dimAlpha;
          vertexData[offset++] = seg.isDashed;
          vertexData[offset++] = seg.startLen;
          vertexData[offset++] = seg.r;
          vertexData[offset++] = seg.g;
          vertexData[offset++] = seg.b;
          vertexData[offset++] = seg.a;
        }
      }

      strokeBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, strokeBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, vertexData, gl.STATIC_DRAW);

      const stride = floatsPerVertex * 4;

      const setupAttr = (name: string, size: number, attrOffset: number) => {
        const loc = gl.getAttribLocation(strokeProgram, name);
        if (loc >= 0) {
          gl.enableVertexAttribArray(loc);
          gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, attrOffset * 4);
        }
      };

      setupAttr("a_quad_pos", 2, 0);
      setupAttr("a_seg_p0", 2, 2);
      setupAttr("a_seg_p1", 2, 4);
      setupAttr("a_seg_props", 4, 6);
      setupAttr("a_seg_color", 4, 10);

      const uBoundsLoc = gl.getUniformLocation(strokeProgram, "u_bounds");
      const uResLoc = gl.getUniformLocation(strokeProgram, "u_canvas_res");
      const uDashSizeLoc = gl.getUniformLocation(strokeProgram, "u_dashSize");
      const uDashRatioLoc = gl.getUniformLocation(strokeProgram, "u_dashRatio");

      gl.uniform4f(uBoundsLoc, minX, minY, bWidth, bHeight);
      gl.uniform2f(uResLoc, widthPx, heightPx);
      if (uDashSizeLoc) gl.uniform1f(uDashSizeLoc, 0.5);
      if (uDashRatioLoc) gl.uniform1f(uDashRatioLoc, 0.6);

      gl.drawArrays(gl.TRIANGLES, 0, segments.length * 6);
      disableAllAttribs();
    }

    return () => {
      if (fillBuffer) gl.deleteBuffer(fillBuffer);
      if (strokeBuffer) gl.deleteBuffer(strokeBuffer);
      gl.deleteProgram(fillProgram);
      gl.deleteProgram(strokeProgram);
      gl.deleteShader(fillVert);
      gl.deleteShader(fillFrag);
      gl.deleteShader(strokeVert);
      gl.deleteShader(strokeFrag);
    };
  }, [footprint, bounds, layerColors, layerVisibility, activeLayer, dimInactiveLayers, reference, value]);

  return <canvas ref={canvasRef} className={className} style={style} />;
}
