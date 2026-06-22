// Packed-vertex round-trip + index generation. RENDERER_PLAN.md §10, §24.10, §22.

import { describe, it, expect } from "vitest";
import { decodeVertex, encodeVertex, PackedVertexSink, packVertices, quadIndices, terrainVertexLayout, VERTEX_STRIDE_BYTES, type RawVertex } from "./VertexFormat";
import { FULLBRIGHT_LIGHT, packLight } from "./Lighting";

function roundTrip(v: RawVertex): RawVertex {
  const buf = new ArrayBuffer(VERTEX_STRIDE_BYTES);
  const dv = new DataView(buf);
  encodeVertex(v, dv, 0);
  return decodeVertex(dv, 0);
}

describe("VertexFormat", () => {
  it("round-trips position (1/16-grid values exact in 1/2048 fixed-point), color, normal, material", () => {
    // 5.375 = 11008/2048 and 12.0625 = 41088/2048 — integer numerators, so exact after quantization.
    const out = roundTrip({ x: 5.375, y: 0, z: 12.0625, u: 0, v: 1, normal: 5, colorRGBA: 0x12345678, material: 0xab });
    expect(out.x).toBeCloseTo(5.375, 6);
    expect(out.y).toBe(0);
    expect(out.z).toBeCloseTo(12.0625, 6);
    expect(out.normal).toBe(5);
    expect(out.material).toBe(0xab);
    expect(out.colorRGBA >>> 0).toBe(0x12345678);
  });

  it("round-trips the packed lightmap word (P6/LM-4); omitting light defaults to full-bright", () => {
    const lit = packLight(7, 12, 200, false);
    const out = roundTrip({ x: 0, y: 0, z: 0, u: 0, v: 0, normal: 0, colorRGBA: 0, material: 0, light: lit });
    expect(out.light).toBe(lit);
    const def = roundTrip({ x: 0, y: 0, z: 0, u: 0, v: 0, normal: 0, colorRGBA: 0, material: 0 });
    expect(def.light).toBe(FULLBRIGHT_LIGHT);
  });

  it("quantizes off-grid position to within 1/2048 block", () => {
    const out = roundTrip({ x: 1.2944, y: -0.7071, z: 17.3, u: 0, v: 0, normal: 0, colorRGBA: 0, material: 0 });
    const tol = 1 / 2048;
    expect(Math.abs(out.x - 1.2944)).toBeLessThanOrEqual(tol);
    expect(Math.abs(out.y - -0.7071)).toBeLessThanOrEqual(tol);
    expect(Math.abs(out.z - 17.3)).toBeLessThanOrEqual(tol);
  });

  it("round-trips UV within 16-bit quantization", () => {
    const out = roundTrip({ x: 0, y: 0, z: 0, u: 0.3333, v: 0.75, normal: 0, colorRGBA: 0xffffffff, material: 0 });
    expect(out.u).toBeCloseTo(0.3333, 4);
    expect(out.v).toBeCloseTo(0.75, 4);
  });

  it("packs vertices contiguously at the 20-byte stride", () => {
    const buf = new ArrayBuffer(VERTEX_STRIDE_BYTES * 3);
    const dv = new DataView(buf);
    for (let i = 0; i < 3; i++) {
      encodeVertex({ x: i + 0.5, y: i, z: i * 2, u: 0, v: 0, normal: i, colorRGBA: 0, material: 0 }, dv, i * VERTEX_STRIDE_BYTES);
    }
    for (let i = 0; i < 3; i++) {
      const out = decodeVertex(dv, i * VERTEX_STRIDE_BYTES);
      expect(out.x).toBeCloseTo(i + 0.5, 6);
      expect(out.normal).toBe(i);
    }
  });

  it("PackedVertexSink (single-pass #1) is byte-identical to packVertices, incl. across a grow", () => {
    const verts: RawVertex[] = Array.from({ length: 10 }, (_, i) => ({
      x: i + 0.5, y: i * 0.25, z: 12.0625 - i, u: (i % 4) / 4, v: 0.75, normal: i % 7, colorRGBA: (0x11223344 * (i + 1)) >>> 0, material: i & 0xff,
    }));
    const sink = new PackedVertexSink(1); // capacity 4 verts → forces a grow at vertex 5 and 9
    for (const v of verts) sink.push(v.x, v.y, v.z, v.u, v.v, v.normal, v.colorRGBA, v.material);
    expect(sink.quadCount).toBe(verts.length / 4);
    expect(sink.byteLength).toBe(verts.length * VERTEX_STRIDE_BYTES);
    expect(Array.from(new Uint8Array(sink.finish()))).toEqual(Array.from(new Uint8Array(packVertices(verts))));
  });

  it("generates shared quad indices (0,1,2,0,2,3 per quad)", () => {
    expect(Array.from(quadIndices(1))).toEqual([0, 1, 2, 0, 2, 3]);
    expect(Array.from(quadIndices(2).slice(6))).toEqual([4, 5, 6, 4, 6, 7]);
    expect(quadIndices(1000).length).toBe(6000);
  });

  it("layout matches the 20-byte stride and attribute offsets (incl. the location-3 lightmap)", () => {
    const layout = terrainVertexLayout();
    expect(layout.strideBytes).toBe(20);
    expect(layout.attributes.map((a) => a.offsetBytes)).toEqual([0, 8, 12, 16]);
    expect(layout.attributes.map((a) => a.location)).toEqual([0, 1, 2, 3]);
    // position (loc 0) AND the lightmap (loc 3) are integer-uploaded (uint16x4 / uint8x4).
    expect(layout.attributes.filter((a) => a.asInt).map((a) => a.location)).toEqual([0, 3]);
  });
});
