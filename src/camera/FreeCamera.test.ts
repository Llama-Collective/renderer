// P1.3: FreeCamera was memoized (mirroring Camera) to stop the ~25-35 fresh Float32Array/Vec3 allocations
// per frame the render loop incurred. These tests pin (a) bit-identity with the reference allocating math
// (so the render — and the F1 bundle's viewProjection-bits replay key — is unchanged) and (b) the memoization
// contract: a stable pose returns the SAME buffer (zero-alloc); a pose/projection change re-derives it.

import { describe, it, expect } from "vitest";
import { FreeCamera } from "./FreeCamera";
import { lookAt, multiply, perspectiveZO } from "./math";

describe("FreeCamera (P1.3 memoization)", () => {
  it("viewMatrix / projectionMatrix / viewProjection are BIT-IDENTICAL to the reference allocating math", () => {
    const c = new FreeCamera();
    c.pos = [3, 5, -7]; c.yaw = 0.6; c.pitch = -0.3; c.aspect = 1.7;
    const f = c.forward();
    const refView = lookAt(c.pos, [c.pos[0] + f[0], c.pos[1] + f[1], c.pos[2] + f[2]], [0, 1, 0]);
    const refProj = perspectiveZO(c.fovY, c.aspect, c.near, c.far);
    const refVp = multiply(refProj, refView);
    expect(Array.from(c.viewMatrix())).toEqual(Array.from(refView));
    expect(Array.from(c.projectionMatrix())).toEqual(Array.from(refProj));
    expect(Array.from(c.viewProjection())).toEqual(Array.from(refVp));
  });

  it("returns the SAME buffer on a stable pose (zero-alloc) and re-derives after a change", () => {
    const c = new FreeCamera();
    const vp1 = c.viewProjection();
    expect(c.viewProjection()).toBe(vp1); // identical buffer — no per-frame realloc
    const before = Array.from(vp1);
    c.rotate(0.4, 0.1); // pose changed
    const vp2 = c.viewProjection();
    expect(vp2).toBe(vp1); // SAME reused buffer object…
    expect(Array.from(vp2)).not.toEqual(before); // …with recomputed contents
  });

  it("viewBasis is orthonormal, memoized, and tracks the forward vector", () => {
    const c = new FreeCamera();
    c.pos = [0, 0, 0]; c.yaw = 0.9; c.pitch = 0.2;
    const b1 = c.viewBasis();
    expect(b1).toBe(c.viewBasis()); // memoized on a stable pose
    expect(Array.from(b1.forward)).toEqual(Array.from(c.forward()));
    const len = (v: readonly number[]) => Math.hypot(v[0], v[1], v[2]);
    expect(len(b1.right)).toBeCloseTo(1, 6);
    expect(len(b1.up)).toBeCloseTo(1, 6);
    expect(b1.right[0] * b1.up[0] + b1.right[1] * b1.up[1] + b1.right[2] * b1.up[2]).toBeCloseTo(0, 6);
  });
});
