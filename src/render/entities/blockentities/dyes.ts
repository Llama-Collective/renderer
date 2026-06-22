// DyeColor → texture-diffuse RGB (vanilla `DyeColor.getTextureDiffuseColor`). RENDERER_PLAN.md §18.
// Used to tint bed / shulker / banner / pattern layers. 0xRRGGBB (opaque).

import { unpackRgb } from "../../color";

export const DYE_DIFFUSE: Record<string, number> = {
  white: 0xf9fffe, orange: 0xf9801d, magenta: 0xc74ebd, light_blue: 0x3ab3da,
  yellow: 0xfed83d, lime: 0x80c71f, pink: 0xf38baa, gray: 0x474f52,
  light_gray: 0x9d9d97, cyan: 0x169c9c, purple: 0x8932b8, blue: 0x3c44aa,
  brown: 0x835432, green: 0x5e7c16, red: 0xb02e26, black: 0x1d1d21,
};

export const DYE_NAMES = Object.keys(DYE_DIFFUSE);

/** 0xRRGGBB diffuse for a dye name (default white). */
export function dyeRgb(color: string | undefined): number {
  return DYE_DIFFUSE[color ?? "white"] ?? DYE_DIFFUSE.white;
}

/** 0xRRGGBBAA (opaque) for a dye name — the vertex/instance tint format. */
export function dyeRgba(color: string | undefined): number {
  return ((dyeRgb(color) << 8) | 0xff) >>> 0;
}

/** Linear [r,g,b,1] for a dye name — the per-entity uniform tint format. */
export function dyeTintLinear(color: string | undefined): readonly [number, number, number, number] {
  const [r, g, b] = unpackRgb(dyeRgb(color));
  return [r, g, b, 1];
}
