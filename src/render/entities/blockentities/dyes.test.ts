// DYE_DIFFUSE must equal vanilla DyeColor.getTextureDiffuseColor (the 3rd enum arg). AUDIT M5.

import { describe, it, expect } from "vitest";
import { DYE_DIFFUSE, dyeRgb, dyeRgba } from "./dyes";

// Vanilla DyeColor enum textureDiffuseColor values (world/item/DyeColor.java).
const VANILLA: Record<string, number> = {
  white: 0xf9fffe, orange: 0xf9801d, magenta: 0xc74ebd, light_blue: 0x3ab3da,
  yellow: 0xfed83d, lime: 0x80c71f, pink: 0xf38baa, gray: 0x474f52,
  light_gray: 0x9d9d97, cyan: 0x169c9c, purple: 0x8932b8, blue: 0x3c44aa,
  brown: 0x835432, green: 0x5e7c16, red: 0xb02e26, black: 0x1d1d21,
};

describe("DYE_DIFFUSE", () => {
  it("matches vanilla DyeColor.getTextureDiffuseColor for all 16 dyes", () => {
    expect(DYE_DIFFUSE).toEqual(VANILLA);
  });

  it("dyeRgb defaults to white and dyeRgba appends an opaque alpha byte", () => {
    expect(dyeRgb(undefined)).toBe(0xf9fffe);
    expect(dyeRgb("nonsense")).toBe(0xf9fffe);
    expect(dyeRgba("red")).toBe(0xb02e26ff >>> 0);
  });
});
