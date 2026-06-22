import { describe, expect, it } from "vitest";
import { facingRot, mul, sectionLocal, translation } from "./transforms";

// BBE opt #6: per-(baseDeg, FACING) transform matrices are memoized. These tests pin BOTH the cache
// behavior (same args → same reference) AND the immutability contract the cache depends on (every caller
// goes through `mul`, which never mutates its inputs — so handing out a shared matrix is safe).
describe("facingRot per-FACING transform cache (BBE opt #6)", () => {
  it("returns the SAME matrix reference for identical (facing, baseDeg)", () => {
    expect(facingRot("north", 0)).toBe(facingRot("north", 0));
    expect(facingRot("east")).toBe(facingRot("east"));
    expect(facingRot("south", 180)).toBe(facingRot("south", 180));
  });

  it("returns DISTINCT, correct matrices for distinct facings / base angles", () => {
    expect(facingRot("north")).not.toBe(facingRot("east"));
    expect([...facingRot("north")]).not.toEqual([...facingRot("east")]); // different rotation
    expect(facingRot("south", 0)).not.toBe(facingRot("south", 180)); // baseDeg participates in the key
  });

  it("an undefined facing falls back to north (and shares its cache slot)", () => {
    expect(facingRot(undefined)).toBe(facingRot("north"));
  });

  it("downstream mul()/sectionLocal() does NOT mutate the cached matrix (cache stays valid)", () => {
    const before = [...facingRot("west")];
    // Exactly what the BE defs do — feed the cached matrix into mul via sectionLocal.
    sectionLocal({ x: 5, y: 0, z: 7 }, [0, 0, 0], facingRot("west"));
    mul(translation(1, 2, 3), facingRot("west"));
    expect([...facingRot("west")]).toEqual(before); // unchanged → the shared reference is safe to reuse
  });
});
