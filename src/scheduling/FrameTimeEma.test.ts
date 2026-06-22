// FrameTimeEma (ABP-1) — deterministic, clock-free averaging for the adaptive budget.

import { describe, it, expect } from "vitest";
import { FrameTimeEma } from "./FrameTimeEma";

describe("FrameTimeEma", () => {
  it("seeds at the initial value", () => {
    const ema = new FrameTimeEma(16);
    expect(ema.average).toBe(16);
  });

  it("blends toward a new steady sample by `ratio` each push", () => {
    const ema = new FrameTimeEma(16, 0.5); // big ratio → fast convergence
    expect(ema.push(20)).toBeCloseTo(18, 6); // 16 + 0.5*(20-16)
    expect(ema.push(20)).toBeCloseTo(19, 6); // 18 + 0.5*(20-18)
  });

  it("converges to a constant input over many frames", () => {
    const ema = new FrameTimeEma(60, 0.05);
    for (let i = 0; i < 500; i++) ema.push(8);
    expect(ema.average).toBeCloseTo(8, 3);
  });

  it("clamps both the folded sample and the stored average to the window", () => {
    const ema = new FrameTimeEma(16, 1, 1, 100); // ratio 1 ⇒ value == clamped sample
    expect(ema.push(99999)).toBe(100); // spike clamped to max
    expect(ema.push(-50)).toBe(1); // floored to min (a 0 ms reading can't zero the budget)
  });

  it("a single spike barely moves a smoothed average (spike resistance)", () => {
    const ema = new FrameTimeEma(16, 0.05);
    const before = ema.average;
    ema.push(100); // one stall (already clamped at the cap)
    expect(ema.average).toBeLessThan(before + 5); // 16 + 0.05*(100-16) = 20.2 — gentle, not a jump to 100
  });
});
