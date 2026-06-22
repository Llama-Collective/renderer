// Frame-time exponential moving average (ABP-1). RENDERER_OPTIMIZATION_PLAN — adaptive budgets.
//
// sizes its per-frame upload/mesh budget from a running frame-time average rather than a fixed
// constant: when frames are cheap (headroom) it does MORE streaming work per frame; when frames are
// expensive (CPU/GPU-bound) it backs off so meshing/upload doesn't deepen a stutter. This is the average.
//
// Deliberately CLOCK-FREE: the caller passes each inter-frame `dt` (already spike-clamped at the call site
// so one GC/tab-switch stall can't poison the average), so the EMA is fully deterministic + unit-testable
// with no `performance.now()` dependency. The folded value is also clamped to a sane [min,max] window.

/** Standard EMA: `avg += ratio * (sample − avg)`. Lower ratio ⇒ smoother / slower to react. */
export class FrameTimeEma {
  private value: number;

  constructor(
    /** Seed average (ms). 16.67 ≈ a 60 Hz frame. */
    initialMs = 1000 / 60,
    /** Smoothing factor in (0,1]; 0.05 follows gentle blend (≈20-frame memory). */
    private readonly ratio = 0.05,
    /** Clamp window (ms) for both the folded sample and the stored average — keeps the budget math sane
     *  even if the caller forgets to clamp, and floors it so a 0 ms reading can't zero the budget. */
    private readonly clampMin = 1,
    private readonly clampMax = 100,
  ) {
    this.value = this.clamp(initialMs);
  }

  private clamp(ms: number): number {
    return Math.min(this.clampMax, Math.max(this.clampMin, ms));
  }

  /** Fold one inter-frame dt (ms) into the average and return the new average. */
  push(dtMs: number): number {
    const dt = this.clamp(dtMs);
    this.value = this.clamp(this.value + this.ratio * (dt - this.value));
    return this.value;
  }

  /** The current smoothed frame time (ms). */
  get average(): number {
    return this.value;
  }
}
