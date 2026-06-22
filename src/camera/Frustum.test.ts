// Frustum plane math + the SECTION_MARGIN overhang padding (audit Bug 2). RENDERER_PLAN §14.

import { describe, it, expect } from "vitest";
import { Frustum, SECTION_MARGIN } from "./Frustum";
import type { Mat4 } from "./Camera";

// Identity view-projection ⇒ the frustum IS the WebGPU clip cube: x,y ∈ [-1, 1], z ∈ [0, 1]. With known
// planes we can place a section box a precise distance outside an edge and assert the margin behaviour.
const IDENTITY: Mat4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] as unknown as Mat4;

describe("Frustum — plane test (Gribb–Hartmann, WebGPU z∈[0,1])", () => {
  const f = new Frustum().setFromViewProjection(IDENTITY);

  it("accepts a box straddling the clip cube, rejects one fully outside", () => {
    expect(f.testAab(-0.5, -0.5, 0.2, 0.5, 0.5, 0.8)).toBe(true); // inside
    expect(f.testAab(2, -0.5, 0.2, 3, 0.5, 0.8)).toBe(false); // entirely right of x≤1
    expect(f.testAab(-0.5, -0.5, 1.2, 0.5, 0.5, 1.8)).toBe(false); // entirely beyond far (z≤1)
  });
});

describe("Frustum.testSection — SECTION_MARGIN overhang padding (audit Bug 2)", () => {
  const f = new Frustum().setFromViewProjection(IDENTITY);

  it("uses the documented margin (1.0 + 0.125)", () => {
    expect(SECTION_MARGIN).toBeCloseTo(1.125);
  });

  it("keeps a section whose bare 16³ box sits just outside an edge by LESS than the margin", () => {
    // Origin (1.5,0,0.4): the bare box's nearest face (x=1.5) is 0.5 outside the right plane (x≤1).
    expect(f.testAab(1.5, 0, 0.4, 1.5 + 16, 0 + 16, 0.4 + 16)).toBe(false); // bare box → false-culled
    expect(f.testSection(1.5, 0, 0.4)).toBe(true); // padded by 1.125 > 0.5 → overhanging geometry kept
  });

  it("still culls a section that overhangs by MORE than the margin", () => {
    // Origin x=3 ⇒ padded nearest face = 3 − 1.125 = 1.875, still right of x≤1 ⇒ genuinely outside.
    expect(f.testSection(3, 0, 0.4)).toBe(false);
  });

  it("keeps a section straddling the cube and culls a far-away one", () => {
    expect(f.testSection(-8, -8, 0.2)).toBe(true); // box centred on the origin
    expect(f.testSection(-100, 0, 0.5)).toBe(false); // far to the left
  });
});
