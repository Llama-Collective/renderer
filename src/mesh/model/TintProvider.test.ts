// Biome-capable tint resolution. RENDERER_PLAN §24.8, §22.

import { describe, it, expect } from "vitest";
import { TintProvider, PLAINS, type Colormap } from "./TintProvider";

describe("TintProvider (no colormap → defaults)", () => {
  const t = new TintProvider();
  it("untinted → white", () => {
    expect(t.tintFor("oak_log", -1)).toBe(0xffffff);
    expect(t.tintFor("stone", 0)).toBe(0xffffff);
  });
  it("grass / foliage / water defaults", () => {
    expect(t.tintFor("grass_block", 0)).toBe(0x91bd59);
    expect(t.tintFor("oak_leaves", 0)).toBe(0x77ab2f);
    expect(t.tintFor("water", 0)).toBe(0x3f76e4);
  });
  it("spruce / birch leaves are biome-independent fixed colors", () => {
    expect(t.tintFor("spruce_leaves", 0)).toBe(0x619961);
    expect(t.tintFor("birch_leaves", 0)).toBe(0x80a755);
  });
  it("only vanilla-listed leaves are foliage-tinted; cherry/azalea/pale_oak are untinted (AUDIT M4)", () => {
    for (const n of ["jungle_leaves", "acacia_leaves", "dark_oak_leaves", "mangrove_leaves", "vine"]) {
      expect(t.tintFor(n, 0), n).toBe(0x77ab2f); // biome foliage
    }
    for (const n of ["cherry_leaves", "azalea_leaves", "flowering_azalea_leaves", "pale_oak_leaves"]) {
      expect(t.tintFor(n, 0), n).toBe(0xffffff); // NOT in BlockColors → white, so the texture shows through
    }
  });
});

describe("TintProvider (colormap sampling)", () => {
  it("samples the grass colormap at the climate point", () => {
    const cm: Colormap = { width: 256, height: 256, rgba: new Uint8Array(256 * 256 * 4) };
    // Plains (temp .8, downfall .4): i = (1-.8)*255 = 51, j = (1-.4*.8)*255 = 173.
    const i = Math.floor((1 - 0.8) * 255);
    const j = Math.floor((1 - 0.4 * 0.8) * 255);
    const o = (j * 256 + i) * 4;
    cm.rgba[o] = 0xaa;
    cm.rgba[o + 1] = 0xbb;
    cm.rgba[o + 2] = 0xcc;
    const t = new TintProvider({ grass: cm });
    expect(t.tintFor("grass_block", 0, PLAINS)).toBe(0xaabbcc);
  });

  it("a different climate samples a different texel (biome capability)", () => {
    const cm: Colormap = { width: 256, height: 256, rgba: new Uint8Array(256 * 256 * 4).fill(0) };
    const set = (temp: number, down: number, rgb: number) => {
      const i = Math.floor((1 - temp) * 255);
      const j = Math.floor((1 - down * temp) * 255);
      const o = (j * 256 + i) * 4;
      cm.rgba[o] = (rgb >> 16) & 0xff;
      cm.rgba[o + 1] = (rgb >> 8) & 0xff;
      cm.rgba[o + 2] = rgb & 0xff;
    };
    set(0.8, 0.4, 0x111111);
    set(0.2, 0.9, 0x222222);
    const t = new TintProvider({ grass: cm });
    expect(t.tintFor("grass_block", 0, { temperature: 0.2, downfall: 0.9 })).toBe(0x222222);
  });
});
