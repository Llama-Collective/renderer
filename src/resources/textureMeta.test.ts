// mcmeta animation parsing — must NOT treat mipmap-only mcmeta as animated. §24.9, §22.

import { describe, it, expect } from "vitest";
import { parseAnimationMeta } from "./textureMeta";

describe("parseAnimationMeta", () => {
  it("parses a real animation block", () => {
    expect(parseAnimationMeta({ animation: { frametime: 2 } })).toMatchObject({ frameTime: 2, interpolate: false });
  });

  it("returns null for mipmap-strategy mcmeta (the cubane false-positive)", () => {
    expect(parseAnimationMeta({ texture: { mipmap_strategy: "mean" } })).toBeNull();
    expect(parseAnimationMeta({ texture: { mipmap_strategy: "dark_cutout" } })).toBeNull();
  });

  it("returns null for empty / non-object input", () => {
    expect(parseAnimationMeta({})).toBeNull();
    expect(parseAnimationMeta(null)).toBeNull();
    expect(parseAnimationMeta("nope")).toBeNull();
  });

  it("reads frame order (ints and {index} objects) + interpolate + default frametime", () => {
    const m = parseAnimationMeta({ animation: { frames: [0, 2, { index: 1, time: 5 }], interpolate: true } });
    expect(m).toMatchObject({ frames: [0, 2, 1], interpolate: true, frameTime: 1 });
  });
});
