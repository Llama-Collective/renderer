// Shared color helpers for the render layer. RENDERER_PLAN.md §7.

/**
 * Unpack a packed `0xRRGGBB` integer into `[r, g, b]` floats in [0,1] (raw `/255`). These are STRAIGHT
 * colors (NOT sRGB→linear converted) — they feed the colored-vertex / overlay / effect paths that author
 * color directly. Append alpha at the call site. The single home of the `>> 16 & 0xff` unpack so the five
 * call sites can't drift.
 */
export function unpackRgb(packed: number): [number, number, number] {
  return [((packed >> 16) & 0xff) / 255, ((packed >> 8) & 0xff) / 255, (packed & 0xff) / 255];
}
