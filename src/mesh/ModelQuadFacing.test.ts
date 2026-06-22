// FS-1 — the single 7-slice facing quantizer. The translucent collector and the opaque/cutout facing
// partition both route through this, so it must classify the 6 axes + UNASSIGNED exactly.

import { describe, it, expect } from "vitest";
import { ModelQuadFacing, facingFromNormal, quadFacing } from "./ModelQuadFacing";
import { makeTQuad, type Vec3 } from "./TranslucentCollector";

describe("ModelQuadFacing (FS-1)", () => {
  it("classifies the six axis normals", () => {
    expect(facingFromNormal(1, 0, 0)).toBe(ModelQuadFacing.PosX);
    expect(facingFromNormal(-1, 0, 0)).toBe(ModelQuadFacing.NegX);
    expect(facingFromNormal(0, 1, 0)).toBe(ModelQuadFacing.PosY);
    expect(facingFromNormal(0, -1, 0)).toBe(ModelQuadFacing.NegY);
    expect(facingFromNormal(0, 0, 1)).toBe(ModelQuadFacing.PosZ);
    expect(facingFromNormal(0, 0, -1)).toBe(ModelQuadFacing.NegZ);
  });

  it("classifies a non-axis normal as UNASSIGNED", () => {
    const s = Math.SQRT1_2;
    expect(facingFromNormal(s, 0, s)).toBe(ModelQuadFacing.Unassigned); // 45° in XZ
    expect(facingFromNormal(0.6, 0.8, 0)).toBe(ModelQuadFacing.Unassigned);
  });

  it("quadFacing derives the GEOMETRIC normal from corners (geometry-only ⇒ NO_SHADE axis faces still bucket)", () => {
    // A unit quad in the plane z=1, wound CCW from +Z outside → outward normal +Z.
    const p: [Vec3, Vec3, Vec3, Vec3] = [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]];
    const f = quadFacing(...p);
    expect(f === ModelQuadFacing.PosZ || f === ModelQuadFacing.NegZ).toBe(true); // axis-aligned, not UNASSIGNED
    expect(f).not.toBe(ModelQuadFacing.Unassigned);
  });

  it("quadFacing agrees with makeTQuad.facing for the same corners (single quantizer, no drift)", () => {
    const cases: [Vec3, Vec3, Vec3, Vec3][] = [
      [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], // a horizontal (±Y) face
      [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], // a vertical (±X) face
      [[0, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1]], // a diagonal → UNASSIGNED on both
    ];
    for (const c of cases) {
      expect(quadFacing(...c)).toBe(makeTQuad(...c).facing);
    }
  });
});
