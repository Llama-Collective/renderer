// MEM-1 — RegionKey: regionOf / packRegionCoord round-trip + no-overflow parity with packSectionCoord.
//
// The region key reuses the SAME bias/span scheme as OcclusionCuller.packSectionCoord; since region coords
// are >>3/>>2 of section coords (strictly smaller magnitude), they sit inside the existing ±16384 bias
// headroom with no overflow. These tests pin the 8×4×8 mapping + the lossless pack across the full domain.

import { describe, it, expect } from "vitest";
import { regionOf, packRegionCoord, packRegionOfSection, unpackRegionCoord, REGION_SHIFT_X, REGION_SHIFT_Y, REGION_SHIFT_Z } from "./RegionKey";
import { packSectionCoord } from "./OcclusionCuller";

describe("RegionKey.regionOf — 8×4×8 section→region mapping", () => {
  it("maps the 8×4×8 block of sections to one region (origin region)", () => {
    for (let sx = 0; sx < 8; sx++)
      for (let sy = 0; sy < 4; sy++)
        for (let sz = 0; sz < 8; sz++)
          expect(regionOf(sx, sy, sz)).toEqual([0, 0, 0]);
  });

  it("crosses to the next region exactly at the 8/4/8 span boundary on each axis", () => {
    expect(regionOf(7, 3, 7)).toEqual([0, 0, 0]);
    expect(regionOf(8, 0, 0)).toEqual([1, 0, 0]); // x span = 8
    expect(regionOf(0, 4, 0)).toEqual([0, 1, 0]); // y span = 4
    expect(regionOf(0, 0, 8)).toEqual([0, 0, 1]); // z span = 8
  });

  it("uses arithmetic shift (floor-divide) so negative section coords map to the region below", () => {
    expect(regionOf(-1, -1, -1)).toEqual([-1, -1, -1]); // −1 >> 3 === −1, −1 >> 2 === −1
    expect(regionOf(-8, -4, -8)).toEqual([-1, -1, -1]);
    expect(regionOf(-9, -5, -9)).toEqual([-2, -2, -2]);
  });

  it("the span shifts are the documented 8×4×8", () => {
    expect([REGION_SHIFT_X, REGION_SHIFT_Y, REGION_SHIFT_Z]).toEqual([3, 2, 3]);
  });
});

describe("RegionKey.packRegionCoord — round-trip + no overflow", () => {
  it("round-trips region coords across a wide range (lossless pack/unpack)", () => {
    for (const rx of [-1000, -8, -1, 0, 1, 8, 1000])
      for (const ry of [-1000, -4, -1, 0, 1, 4, 1000])
        for (const rz of [-1000, -8, -1, 0, 1, 8, 1000])
          expect(unpackRegionCoord(packRegionCoord(rx, ry, rz))).toEqual([rx, ry, rz]);
  });

  it("packRegionOfSection == packRegionCoord(regionOf(section))", () => {
    for (const [sx, sy, sz] of [[0, 0, 0], [7, 3, 7], [8, 4, 8], [-9, -5, -9], [123, 45, -67]] as const) {
      const [rx, ry, rz] = regionOf(sx, sy, sz);
      expect(packRegionOfSection(sx, sy, sz)).toBe(packRegionCoord(rx, ry, rz));
    }
  });

  it("distinct region coords pack to distinct keys (no collision in the working domain)", () => {
    const seen = new Set<number>();
    for (let rx = -50; rx <= 50; rx++)
      for (let rz = -50; rz <= 50; rz++) {
        const k = packRegionCoord(rx, 0, rz);
        expect(seen.has(k)).toBe(false);
        seen.add(k);
      }
  });

  it("stays a safe integer at the ±16384-section bias domain (the packSectionCoord ceiling — no overflow)", () => {
    // The largest in-domain SECTION coord (±16384) maps to a region coord ÷8 (or ÷4), far smaller — so the
    // region key is well under the packSectionCoord cube ceiling, itself well under Number.MAX_SAFE_INTEGER.
    const sectionCeil = packSectionCoord(16384, 16384, 16384);
    const regionCeil = packRegionCoord(16384, 16384, 16384); // strictly worse-case than any real region coord
    expect(Number.isSafeInteger(sectionCeil)).toBe(true);
    expect(Number.isSafeInteger(regionCeil)).toBe(true);
    // A real region coord (section ÷8) round-trips even at the section-domain extreme.
    const [rx, ry, rz] = regionOf(16383, 16383, 16383);
    expect(unpackRegionCoord(packRegionCoord(rx, ry, rz))).toEqual([rx, ry, rz]);
  });
});
