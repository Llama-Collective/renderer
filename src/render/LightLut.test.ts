import { describe, expect, it } from "vitest";
import { buildLightLut, defaultLightLut, LIGHT_LUT_SIZE } from "./LightLut";

const N = LIGHT_LUT_SIZE;
/** Texel byte (grey channel) at (block, sky) in a 16×16 RGBA8 table. */
function texel(lut: Uint8Array, block: number, sky: number): number {
  return lut[(sky * N + block) * 4];
}

describe("LightLut (P6/LM-1)", () => {
  it("is a 16×16 RGBA8 table (1 KiB)", () => {
    expect(defaultLightLut().length).toBe(N * N * 4);
  });

  it("default LUT maps (block 0, sky 15) to white — the bit-identical full-bright default", () => {
    const lut = defaultLightLut();
    expect(texel(lut, 0, 15)).toBe(255);
    // alpha always opaque
    expect(lut[(15 * N + 0) * 4 + 3]).toBe(255);
  });

  it("default LUT darkens toward (block 0, sky 0)", () => {
    const lut = defaultLightLut();
    expect(texel(lut, 0, 0)).toBe(0); // no ambient floor by default
    expect(texel(lut, 0, 8)).toBeGreaterThan(0);
    expect(texel(lut, 0, 8)).toBeLessThan(255);
  });

  it("block light brightens independently of sky (max-blend)", () => {
    const lut = defaultLightLut();
    expect(texel(lut, 15, 0)).toBe(255); // full block light → white even with no sky
    expect(texel(lut, 15, 15)).toBe(255);
  });

  it("brightness scales the whole table (free global dimming, no remesh)", () => {
    const half = buildLightLut({ brightness: 0.5 });
    expect(texel(half, 0, 15)).toBe(Math.round(0.5 * 255));
    expect(texel(half, 15, 15)).toBe(Math.round(0.5 * 255));
    const dark = buildLightLut({ brightness: 0 });
    expect(texel(dark, 15, 15)).toBe(0);
  });

  it("ambient lifts the floor at light 0 (night-vision / minimum light)", () => {
    const amb = buildLightLut({ ambient: 0.2 });
    expect(texel(amb, 0, 0)).toBe(Math.round(0.2 * 255));
    expect(texel(amb, 0, 15)).toBe(255); // top still white
  });

  it("skyScale dims the sky channel (dusk) without touching block light", () => {
    const night = buildLightLut({ skyScale: 0 });
    expect(texel(night, 0, 15)).toBe(0); // sky contributes nothing
    expect(texel(night, 15, 0)).toBe(255); // block light unaffected
  });

  it("default curve stays linear — byte-identical to the golden table (SL-5)", () => {
    expect(Array.from(buildLightLut({ curve: "linear" }))).toEqual(Array.from(defaultLightLut()));
  });

  it("vanilla curve darkens mid-levels but preserves both endpoints (SL-5, TRAP 6.B)", () => {
    const vanilla = buildLightLut({ curve: "vanilla" });
    const linear = defaultLightLut();
    expect(texel(vanilla, 0, 15)).toBe(255); // full-bright still white — invariant intact
    expect(texel(vanilla, 0, 0)).toBe(0); // dark still black
    expect(texel(vanilla, 0, 8)).toBeLessThan(texel(linear, 0, 8)); // mid-range reads darker than linear
    expect(texel(vanilla, 0, 8)).toBeGreaterThan(0);
  });

  it("gamma defaults to identity and bends mid-tones while keeping endpoints (SL-5)", () => {
    expect(Array.from(buildLightLut({ gamma: 1 }))).toEqual(Array.from(defaultLightLut()));
    const g = buildLightLut({ gamma: 2 });
    expect(texel(g, 0, 15)).toBe(255); // endpoint preserved
    expect(texel(g, 0, 0)).toBe(0);
    expect(texel(g, 0, 8)).toBeLessThan(texel(defaultLightLut(), 0, 8)); // gamma 2 darkens mids
  });
});
