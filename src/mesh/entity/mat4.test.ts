// A3: the in-place mat4 builders + per-frame arena must be BIT-IDENTICAL to the allocating free functions,
// since the entity build path swaps `translation(…)`→`F.T(…)` etc. and relies on byte-identical geometry.

import { describe, expect, it } from "vitest";
import {
  Mat4Frame,
  identity, identityInto,
  mul, mulChainInto, multiply,
  rotationX, rotationXInto,
  rotationY, rotationYInto,
  rotationZ, rotationZInto,
  scaling, scalingInto,
  translation, translationInto,
  type Mat4,
} from "./mat4";

const arr = (m: Mat4) => Array.from(m);

/** Deterministic pseudo-random mat4 (no Math.random — reproducible). */
function pseudo(seed: number): Mat4 {
  const m = new Float32Array(16);
  let s = seed >>> 0;
  for (let i = 0; i < 16; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    m[i] = (s / 0xffffffff) * 4 - 2;
  }
  return m;
}

describe("mat4 in-place builders are bit-identical to the allocating versions", () => {
  it("identityInto / translationInto / scalingInto", () => {
    const o = new Float32Array(16).fill(9); // pre-dirtied to prove a full overwrite
    identityInto(o); expect(arr(o)).toEqual(arr(identity()));
    translationInto(o, 1.5, -2.25, 3.0); expect(arr(o)).toEqual(arr(translation(1.5, -2.25, 3.0)));
    scalingInto(o, 0.75, -0.5, 2.0); expect(arr(o)).toEqual(arr(scaling(0.75, -0.5, 2.0)));
  });

  it("rotationX/Y/Z Into", () => {
    const o = new Float32Array(16).fill(9);
    for (const rad of [0, 0.3927, Math.PI / 2, Math.PI, -1.234, 2 * Math.PI]) {
      rotationXInto(o, rad); expect(arr(o)).toEqual(arr(rotationX(rad)));
      rotationYInto(o, rad); expect(arr(o)).toEqual(arr(rotationY(rad)));
      rotationZInto(o, rad); expect(arr(o)).toEqual(arr(rotationZ(rad)));
    }
  });
});

describe("mulChainInto is bit-identical to mul(...ms)", () => {
  it("matches across chain lengths 1..7", () => {
    const out = new Float32Array(16);
    const tmp = new Float32Array(16);
    for (let n = 1; n <= 7; n++) {
      const list: Mat4[] = [];
      for (let i = 0; i < n; i++) list.push(pseudo(n * 31 + i + 1));
      mulChainInto(out, tmp, list);
      expect(arr(out)).toEqual(arr(mul(...list))); // exact float equality, not approx
    }
  });

  it("two-matrix chain equals multiply()", () => {
    const a = pseudo(5), b = pseudo(99);
    const out = new Float32Array(16), tmp = new Float32Array(16);
    mulChainInto(out, tmp, [a, b]);
    expect(arr(out)).toEqual(arr(multiply(a, b)));
  });
});

describe("Mat4Frame", () => {
  it("builders match the allocating versions, including nested mul (the entity build pattern)", () => {
    const F = new Mat4Frame();
    expect(arr(F.T(1, 2, 3))).toEqual(arr(translation(1, 2, 3)));
    expect(arr(F.S(2, 3, 4))).toEqual(arr(scaling(2, 3, 4)));
    expect(arr(F.Rx(0.7))).toEqual(arr(rotationX(0.7)));
    expect(arr(F.Ry(0.7))).toEqual(arr(rotationY(0.7)));
    expect(arr(F.Rz(0.7))).toEqual(arr(rotationZ(0.7)));
    // The exact shape a minecart's contained block builds: a 7-deep chain with nested mul args.
    const got = F.mul(F.T(3, 4, 5), F.T(0, 0.375, 0), F.Ry(1.1), F.Rz(-0.2), F.S(0.75, 0.75, 0.75), F.T(-0.5, 0.1, 0.5), F.Ry(Math.PI / 2));
    const want = mul(translation(3, 4, 5), translation(0, 0.375, 0), rotationY(1.1), rotationZ(-0.2), scaling(0.75, 0.75, 0.75), translation(-0.5, 0.1, 0.5), rotationY(Math.PI / 2));
    expect(arr(got)).toEqual(arr(want));
    // scaleAboutCenter pattern (nested mul as an argument).
    const center = F.mul(F.T(0.5, 0.5, 0.5), F.mul(F.S(1.3, 1.3, 1.3), F.T(-0.5, -0.5, -0.5)));
    const centerWant = mul(translation(0.5, 0.5, 0.5), mul(scaling(1.3, 1.3, 1.3), translation(-0.5, -0.5, -0.5)));
    expect(arr(center)).toEqual(arr(centerWant));
  });

  it("reuses buffers across reset() and grows only on demand (X5 gauge)", () => {
    let grows = 0;
    const F = new Mat4Frame(() => grows++);
    const a0 = F.T(1, 0, 0), a1 = F.T(2, 0, 0);
    expect(grows).toBe(2);
    F.reset();
    const b0 = F.T(3, 0, 0), b1 = F.T(4, 0, 0);
    expect(b0).toBe(a0); // same backing buffer reused
    expect(b1).toBe(a1);
    expect(grows).toBe(2); // no new allocation in the second frame
    F.T(5, 0, 0); // exceeds the 2-slot high-water mark → one growth
    expect(grows).toBe(3);
  });

  it("Tw subtracts the anchor in double; anchor [0,0,0] makes Tw === T (M6 off-path byte-identity)", () => {
    const F = new Mat4Frame();
    // Default anchor [0,0,0] ⇒ Tw is byte-identical to T (the camera-relative-OFF path).
    expect(Array.from(F.Tw(3, 5, -7))).toEqual(Array.from(F.T(3, 5, -7)));
    // With an anchor, Tw stores (world − anchor) — and the subtraction is in double, so a far + fractional
    // coordinate keeps full precision that an absolute f32 build then subtracting would lose.
    F.setAnchor(30_000_004.3, 0, 0);
    const rel = F.Tw(30_000_000.5, 2, 9);
    expect(rel[12]).toBe(Math.fround(30_000_000.5 - 30_000_004.3)); // exact relative X
    expect(rel[13]).toBe(2); // unanchored axes unchanged
    expect(rel[14]).toBe(9);
  });
});
