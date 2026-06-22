// Shared colored-vertex scaffolding for the simple forward renderers (overlay lines/grid + explosion
// particles). RENDERER_PLAN.md §7. The vertex is `pos vec3 (12B) + rgba unorm8x4 (4B)` = 16-byte stride,
// drawn with a passthrough shader. Each renderer keeps its own accumulator/`col` helpers (their signatures
// differ), but the byte layout, shader, packing, and sequential index buffer are ONE definition here so
// they can't drift (the clamp policy in particular — `packColorVerts` always clamps to [0,1]).

import { clamp01 } from "../types";

/** Colored vertex stride: pos vec3 (12B) + rgba unorm8x4 (4B). */
export const COLOR_VERTEX_STRIDE = 16;

/** Passthrough shader: `clip = viewProj·pos`, color forwarded. One uniform (viewProj at binding 0). */
export const COLOR_VERTEX_WGSL = /* wgsl */ `
struct U { viewProj : mat4x4<f32> };
@group(0) @binding(0) var<uniform> u : U;
struct VsIn { @location(0) pos : vec3<f32>, @location(1) color : vec4<f32> };
struct VsOut { @builtin(position) clip : vec4<f32>, @location(0) color : vec4<f32> };
@vertex fn vs(in : VsIn) -> VsOut {
  var o : VsOut;
  o.clip = u.viewProj * vec4<f32>(in.pos, 1.0);
  o.color = in.color;
  return o;
}
@fragment fn fs(in : VsOut) -> @location(0) vec4<f32> { return in.color; }
`;

/** Pack a flat `[x,y,z, r,g,b,a, …]` stream (7 floats/vertex, colors in [0,1]) into the 16B color-vertex
 *  buffer. Colors are clamped to [0,1] then rounded to unorm8 (the single clamp policy). */
export function packColorVerts(flat: number[]): Uint8Array {
  const verts = flat.length / 7;
  const buf = new ArrayBuffer(verts * COLOR_VERTEX_STRIDE);
  const dv = new DataView(buf);
  for (let v = 0; v < verts; v++) {
    const o = v * COLOR_VERTEX_STRIDE;
    const f = v * 7;
    dv.setFloat32(o, flat[f], true);
    dv.setFloat32(o + 4, flat[f + 1], true);
    dv.setFloat32(o + 8, flat[f + 2], true);
    dv.setUint8(o + 12, Math.round(clamp01(flat[f + 3]) * 255));
    dv.setUint8(o + 13, Math.round(clamp01(flat[f + 4]) * 255));
    dv.setUint8(o + 14, Math.round(clamp01(flat[f + 5]) * 255));
    dv.setUint8(o + 15, Math.round(clamp01(flat[f + 6]) * 255));
  }
  return new Uint8Array(buf);
}

/** A sequential `[0,1,2,…,n-1]` Uint32 index buffer (line-list pairs / triangle-list / grid). */
export function sequentialIndexBuffer(n: number): Uint32Array {
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  return idx;
}
