import { describe, expect, it } from "vitest";
import { AO_NONE, FULLBRIGHT_LIGHT, lightCoord, lightLevelOf, packLight, unpackLight } from "./Lighting";

describe("Lighting pack/unpack (LM-4 encoding)", () => {
  it("maps every level to its LUT texel-center coord (level*16+8, range 8..248)", () => {
    expect(lightCoord(0)).toBe(8);
    expect(lightCoord(15)).toBe(248);
    for (let l = 0; l < 16; l++) {
      const c = lightCoord(l);
      expect(c).toBe(l * 16 + 8);
      // coord/256 must land exactly on the (l+0.5)/16 texel center of a 16-wide LUT.
      expect(c / 256).toBeCloseTo((l + 0.5) / 16, 12);
      expect(lightLevelOf(c)).toBe(l);
    }
  });

  it("clamps out-of-range levels onto the LUT (no off-edge read)", () => {
    expect(lightCoord(-3)).toBe(lightCoord(0));
    expect(lightCoord(99)).toBe(lightCoord(15));
  });

  it("round-trips block/sky/ao/emissive through the packed word", () => {
    for (const block of [0, 4, 9, 15]) {
      for (const sky of [0, 7, 15]) {
        for (const ao of [0, 128, 255]) {
          const w = packLight(block, sky, ao, false);
          const u = unpackLight(w);
          expect(u.blockLight).toBe(block);
          expect(u.skyLight).toBe(sky);
          expect(u.ao).toBe(ao);
          expect(u.emissive).toBe(false);
        }
      }
    }
  });

  it("packs little-endian: byte0=block, byte1=sky, byte2=ao, byte3=flags", () => {
    const w = packLight(0, 15, AO_NONE, false);
    expect(w & 0xff).toBe(8); // block coord (level 0)
    expect((w >>> 8) & 0xff).toBe(248); // sky coord (level 15)
    expect((w >>> 16) & 0xff).toBe(255); // ao
    expect((w >>> 24) & 0xff).toBe(0); // flags
  });

  it("emissive lifts block light to full + sets the flag bit", () => {
    const w = packLight(0, 0, AO_NONE, true);
    const u = unpackLight(w);
    expect(u.blockLight).toBe(15);
    expect(u.emissive).toBe(true);
  });

  it("FULLBRIGHT is block 0 / sky 15 / unoccluded (the bit-identical editor default)", () => {
    const u = unpackLight(FULLBRIGHT_LIGHT);
    expect(u).toEqual({ blockLight: 0, skyLight: 15, ao: AO_NONE, emissive: false });
  });
});
