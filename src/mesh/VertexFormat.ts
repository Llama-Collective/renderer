// Packed terrain vertex. RENDERER_PLAN.md §10, §24.10.
//
// 20-byte layout. Position is FIXED-POINT (3× u16, section-local block coords):
// models live on the 1/16 grid (fences/stairs) and rotated elements are off-grid, but a 16-bit
// per-axis encoding over a 32-block range gives 1/2048-block precision (≈0.0005) — ~30× finer than
// the model grid and ample for the cos(22.5°) rotated-element offsets. So we get arbitrary model
// geometry at 6 position bytes, NOT the 12 a float32x3 would cost (the trap I fell into first).
// `uint16x4` is the tightest valid WebGPU format (no `uint16x3`), so the 4th component is free —
// it carries the packed (normal, material). The vertex shader decodes pos and adds the section
// origin (it already translated by the origin, so this is one extra multiply-add).
//
//   off 0   pos    uint16x4   x,y,z fixed-point + w = normal | (material<<8)
//   off 8   color  unorm8x4   baked tint RGBA → vec4f (one color path, TRAP 7.D)
//   off 12  uv     unorm16x2  atlas UV → vec2f[0,1]
//   off 16  light  uint8x4    packed lightmap at location 3 (P6/LM-4): blockCoord, skyCoord, ao, flags.
//                   LUT-sampled LEVELS, never pre-multiplied color (TRAP 6.B). See mesh/Lighting.ts.
//
// The byte layout lives ONLY here; the WGSL `VsIn` must agree, and the round-trip is unit-tested (§22).

import type { VertexLayout } from "../core/GraphicsDevice";
import { VertexScalarKind } from "../core/GraphicsDevice";
import { clamp01 } from "../types";
import { FULLBRIGHT_LIGHT, type PackedLight } from "./Lighting";
import { FACING_COUNT } from "./ModelQuadFacing";

export const VERTEX_STRIDE_BYTES = 20;
/** normalIndex value meaning "no directional shade" (element shade:false). */
export const NO_SHADE = 6;

// Section-local fixed-point position encoding (constants). A u16 maps a 32-block range
// centered so coords in [-8, +24) are representable — covers a 16³ section plus model/fluid overhang
// and the cullface apron. Precision = 1/2048 block. Keep in sync with terrainShader.ts.
export const POS_ORIGIN_BLOCKS = 8;
export const POS_RANGE_BLOCKS = 32;
/** Multiply a block coord by this (after +origin) to get the stored u16. */
export const POS_ENCODE_SCALE = 65536 / POS_RANGE_BLOCKS; // 2048
/** Multiply a stored u16 by this (then -origin) to recover the block coord. */
export const POS_DECODE_SCALE = 1 / POS_ENCODE_SCALE;

export interface RawVertex {
  /** Section-local block coords (float; quantized to 1/2048 on encode). */
  x: number;
  y: number;
  z: number;
  /** Atlas UV in [0,1]. */
  u: number;
  v: number;
  /** Facing index 0..5 (Direction order) for directional shade, or NO_SHADE. */
  normal: number;
  /** Packed tint 0xRRGGBBAA (resolved at bake — TRAP 7.D). */
  colorRGBA: number;
  /** Material bits (e.g. alpha-cutout flag); 0 for solid. */
  material: number;
  /** Packed lightmap word (P6/LM-4 — block/sky LEVEL coords + ao + flags). Optional: omit ⇒ full-bright,
   *  so callers that predate lighting (entities, transient merges) stay byte-identical via the LUT. */
  light?: PackedLight;
}

function encodePos(v: number): number {
  const e = Math.round((v + POS_ORIGIN_BLOCKS) * POS_ENCODE_SCALE);
  return e < 0 ? 0 : e > 0xffff ? 0xffff : e;
}
function decodePos(u: number): number {
  return u * POS_DECODE_SCALE - POS_ORIGIN_BLOCKS;
}

/**
 * Encode a vertex straight from primitive scalars — no `RawVertex` object. This is the single-pass
 * hot-path encoder (AUDIT #1): the mesher pushes vertices through `PackedVertexSink.push` into a
 * growable buffer in the cull loop, eliminating the per-vertex object churn + the second `packVerts`
 * pass. Byte-for-byte identical to `encodeVertex` (which now delegates here).
 */
export function encodeVertexInline(
  out: DataView,
  offset: number,
  x: number,
  y: number,
  z: number,
  u: number,
  v: number,
  normal: number,
  colorRGBA: number,
  material: number,
  light: PackedLight = FULLBRIGHT_LIGHT,
): void {
  out.setUint16(offset + 0, encodePos(x), true);
  out.setUint16(offset + 2, encodePos(y), true);
  out.setUint16(offset + 4, encodePos(z), true);
  out.setUint16(offset + 6, ((material & 0xff) << 8) | (normal & 0xff), true);
  out.setUint8(offset + 8, (colorRGBA >>> 24) & 0xff);
  out.setUint8(offset + 9, (colorRGBA >>> 16) & 0xff);
  out.setUint8(offset + 10, (colorRGBA >>> 8) & 0xff);
  out.setUint8(offset + 11, colorRGBA & 0xff);
  out.setUint16(offset + 12, Math.round(clamp01(u) * 65535), true);
  out.setUint16(offset + 14, Math.round(clamp01(v) * 65535), true);
  // bytes 16..19: packed lightmap (location 3, P6/LM-4). LE word: block | sky | ao | flags.
  out.setUint32(offset + 16, light >>> 0, true);
}

export function encodeVertex(v: RawVertex, out: DataView, offset: number): void {
  encodeVertexInline(out, offset, v.x, v.y, v.z, v.u, v.v, v.normal, v.colorRGBA, v.material, v.light ?? FULLBRIGHT_LIGHT);
}

export function decodeVertex(data: DataView, offset: number): RawVertex {
  const w = data.getUint16(offset + 6, true);
  const r = data.getUint8(offset + 8);
  const g = data.getUint8(offset + 9);
  const b = data.getUint8(offset + 10);
  const a = data.getUint8(offset + 11);
  return {
    x: decodePos(data.getUint16(offset + 0, true)),
    y: decodePos(data.getUint16(offset + 2, true)),
    z: decodePos(data.getUint16(offset + 4, true)),
    normal: w & 0xff,
    material: (w >>> 8) & 0xff,
    colorRGBA: ((r << 24) | (g << 16) | (b << 8) | a) >>> 0,
    u: data.getUint16(offset + 12, true) / 65535,
    v: data.getUint16(offset + 14, true) / 65535,
    light: data.getUint32(offset + 16, true) >>> 0,
  };
}

/** The layout handed to GraphicsDevice.createPipeline. Position is integer-uploaded (`asInt`). */
export function terrainVertexLayout(): VertexLayout {
  return {
    strideBytes: VERTEX_STRIDE_BYTES,
    attributes: [
      // pos.xyz fixed-point + pos.w = packed(normal, material). uint16x4 → vec4<u32> in WGSL.
      { location: 0, kind: VertexScalarKind.Uint16, components: 4, offsetBytes: 0, asInt: true },
      { location: 1, kind: VertexScalarKind.Unorm8, components: 4, offsetBytes: 8, asInt: false },
      { location: 2, kind: VertexScalarKind.Unorm16, components: 2, offsetBytes: 12, asInt: false },
      // P6/LM-4: packed lightmap (blockCoord, skyCoord, ao, flags) → vec4<u32> in WGSL (integer-uploaded).
      { location: 3, kind: VertexScalarKind.Uint8, components: 4, offsetBytes: 16, asInt: true },
    ],
  };
}

/** Pack a vertex list into a tightly-strided ArrayBuffer (20-byte stride). */
export function packVertices(verts: readonly RawVertex[]): ArrayBuffer {
  const buf = new ArrayBuffer(verts.length * VERTEX_STRIDE_BYTES);
  const dv = new DataView(buf);
  for (let i = 0; i < verts.length; i++) encodeVertex(verts[i], dv, i * VERTEX_STRIDE_BYTES);
  return buf;
}

/**
 * A growable packed-vertex buffer (AUDIT #1): the mesher encodes each vertex inline as it culls,
 * with no intermediate `RawVertex[]` and no second packing pass. Grows by doubling; `finish()`
 * returns a tightly-sized ArrayBuffer byte-identical to what `packVertices` would have produced for
 * the same vertices in the same order.
 */
export class PackedVertexSink {
  private ab: ArrayBuffer;
  private buf: Uint8Array;
  private dv: DataView;
  private byteLen = 0;

  constructor(initialQuads = 64) {
    this.ab = new ArrayBuffer(Math.max(1, initialQuads) * 4 * VERTEX_STRIDE_BYTES);
    this.buf = new Uint8Array(this.ab);
    this.dv = new DataView(this.ab);
  }

  /** Append one vertex (4 per quad). `light` defaults to full-bright (P6/LM-4). */
  push(x: number, y: number, z: number, u: number, v: number, normal: number, colorRGBA: number, material: number, light: PackedLight = FULLBRIGHT_LIGHT): void {
    if (this.byteLen + VERTEX_STRIDE_BYTES > this.ab.byteLength) this.grow();
    encodeVertexInline(this.dv, this.byteLen, x, y, z, u, v, normal, colorRGBA, material, light);
    this.byteLen += VERTEX_STRIDE_BYTES;
  }

  private grow(): void {
    const next = new ArrayBuffer(this.ab.byteLength * 2);
    new Uint8Array(next).set(this.buf);
    this.ab = next;
    this.buf = new Uint8Array(next);
    this.dv = new DataView(next);
  }

  get quadCount(): number {
    return this.byteLen / (4 * VERTEX_STRIDE_BYTES);
  }

  get byteLength(): number {
    return this.byteLen;
  }

  isEmpty(): boolean {
    return this.byteLen === 0;
  }

  /** Rewind to empty, KEEPING the backing buffer — for a sink reused across per-frame bakes (A4). */
  reset(): void {
    this.byteLen = 0;
  }

  /** A VIEW of the encoded bytes (no copy) — valid until the next `push`/`reset`. Safe to hand to
   *  `device.writeBuffer`, which copies synchronously (so a reused sink can be refilled afterwards). */
  bytes(): Uint8Array {
    return this.buf.subarray(0, this.byteLen);
  }

  /** A tightly-sized copy of the encoded bytes (the transferable vertex buffer). */
  finish(): ArrayBuffer {
    return this.ab.slice(0, this.byteLen);
  }
}

/**
 * FS-2: an opaque/cutout sink partitioned into 7 facing runs (`ModelQuadFacing` slices). Each
 * sub-sink collects whole quads of one facing; `finish()` concatenates them in canonical facing order
 * (0..6) so the draw layer can emit one drawIndexed per VISIBLE facing run (FS-4) and skip the camera-away
 * slices the GPU already back-face-culls. Each per-facing run is byte-identical to the corresponding subset
 * of an unpartitioned `PackedVertexSink`; only the WHOLE-pass order changes (reordered, not re-encoded), so
 * the rendered image is unchanged (FS-5 proves 0-pixel-diff). Default-off ⇒ the plain `PackedVertexSink`.
 */
export class FacingPartitionedSink {
  private readonly sinks: PackedVertexSink[] = [];

  constructor(initialQuadsPerFacing = 8) {
    for (let i = 0; i < FACING_COUNT; i++) this.sinks.push(new PackedVertexSink(initialQuadsPerFacing));
  }

  /** Append one vertex into the sub-sink for `facing` (0..6). All 4 vertices of a quad share one facing. */
  pushFacing(
    facing: number,
    x: number, y: number, z: number,
    u: number, v: number,
    normal: number, colorRGBA: number, material: number,
    light: PackedLight = FULLBRIGHT_LIGHT,
  ): void {
    this.sinks[facing].push(x, y, z, u, v, normal, colorRGBA, material, light);
  }

  isEmpty(): boolean {
    for (const s of this.sinks) if (!s.isEmpty()) return false;
    return true;
  }

  get quadCount(): number {
    let q = 0;
    for (const s of this.sinks) q += s.quadCount;
    return q;
  }

  /** Per-facing VERTEX counts (4 per quad), canonical order 0..6 — the run lengths in `finish()`'s buffer.
   *  Vertex-relative (NOT absolute offsets) so they survive an arena relocation (FS-3). */
  facingVertexCounts(): Uint32Array {
    const counts = new Uint32Array(FACING_COUNT);
    for (let i = 0; i < FACING_COUNT; i++) counts[i] = this.sinks[i].quadCount * 4;
    return counts;
  }

  /** Concatenate the 7 facing runs (canonical order) into one tightly-sized buffer. The k-th run's bytes
   *  are byte-identical to those quads encoded in order by a plain `PackedVertexSink`. */
  finish(): ArrayBuffer {
    let total = 0;
    for (const s of this.sinks) total += s.byteLength;
    const out = new Uint8Array(total);
    let off = 0;
    for (const s of this.sinks) {
      out.set(s.bytes(), off);
      off += s.byteLength;
    }
    return out.buffer;
  }
}

/**
 * Indices for `quadCount` quads sharing the standard 4-vertex / 2-triangle winding (0,1,2, 0,2,3).
 * Uint32 because a dense section can exceed 65 535 vertices.
 */
/** Write ONE quad's 6-index triangle winding (0,1,2, 0,2,3 relative to `base`) at `out[o..o+5]`. The
 *  single home of the quad winding — shared by `quadIndices` (identity order) and the translucent
 *  `expandQuadOrder` (arbitrary sorted order) so the two can't drift. */
export function writeQuadWinding(out: Uint32Array, o: number, base: number): void {
  out[o] = base;
  out[o + 1] = base + 1;
  out[o + 2] = base + 2;
  out[o + 3] = base;
  out[o + 4] = base + 2;
  out[o + 5] = base + 3;
}

export function quadIndices(quadCount: number): Uint32Array {
  const idx = new Uint32Array(quadCount * 6);
  for (let q = 0; q < quadCount; q++) writeQuadWinding(idx, q * 6, q * 4);
  return idx;
}
