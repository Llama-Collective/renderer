// Generated-item sprite extrusion (vanilla ItemModelGenerator). AUDIT P2.1 (no itemExtrude tests) +
// P2.4 (silhouette uses strict alpha == 0, not a >0 threshold).

import { describe, it, expect } from "vitest";
import { extrudeItemSprite, type ItemSprite } from "./itemExtrude";
import type { SpriteUv } from "../../types";

// A deliberately non-degenerate atlas rect so cap UVs are distinguishable in u and v.
const RECT: SpriteUv = { u: 0.1, v: 0.2, width: 0.5, height: 0.25 };

/** Build a w×h sprite; `alpha[i]` is the per-pixel alpha (RGB is opaque white). */
function sprite(width: number, height: number, alpha: number[]): ItemSprite {
  const rgba = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    rgba[i * 4] = 255;
    rgba[i * 4 + 3] = alpha[i];
  }
  return { rgba, width, height };
}

describe("itemExtrude (vanilla ItemModelGenerator)", () => {
  it("treats a low-alpha (0<a≤16) pixel as OPAQUE — silhouette uses strict alpha == 0", () => {
    // A 1×1 opaque pixel = 2 slab caps + 4 silhouette side faces (all neighbours out-of-bounds) = 6 quads.
    const opaque = extrudeItemSprite(sprite(1, 1, [255]), RECT);
    expect(opaque.length).toBe(24); // 6 quads × 4 verts

    // alpha = 10 must ALSO be opaque under the ==0 rule → same 6 quads. (The old `<= 16` threshold would
    // have made it transparent → only the 2 caps survive = 8 verts.)
    const lowAlpha = extrudeItemSprite(sprite(1, 1, [10]), RECT);
    expect(lowAlpha.length).toBe(24);

    // alpha = 0 is genuinely transparent: no silhouette sides, just the 2 caps.
    const transparent = extrudeItemSprite(sprite(1, 1, [0]), RECT);
    expect(transparent.length).toBe(8);
  });

  it("the back cap is world-texel-ALIGNED, not mirrored (no 'X'): same UV at each world (x,y)", () => {
    const v = extrudeItemSprite(sprite(1, 1, [255]), RECT);
    const front = v.slice(0, 4); // caps are emitted first; front (+Z, z=8.5px) then back (−Z, z=7.5px)
    const back = v.slice(4, 8);
    expect(front.every((p) => Math.abs(p.z * 16 - 8.5) < 1e-4)).toBe(true);
    expect(back.every((p) => Math.abs(p.z * 16 - 7.5) < 1e-4)).toBe(true);

    const key = (p: { x: number; y: number }) => `${p.x.toFixed(4)},${p.y.toFixed(4)}`;
    const frontUv = new Map(front.map((p) => [key(p), [p.u, p.v] as const]));
    for (const p of back) {
      const fu = frontUv.get(key(p));
      expect(fu, "back-cap vertex shares a world (x,y) with a front-cap vertex").toBeDefined();
      expect(p.u).toBeCloseTo(fu![0], 6); // a mirrored back cap would flip U here → produces the "X"
      expect(p.v).toBeCloseTo(fu![1], 6);
    }
  });
});
