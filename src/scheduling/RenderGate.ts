// Pure decision logic for the render-on-demand ("dynamic FPS") loop.
//
// This is a faithful port of the legacy renderer's `utils/renderGate.ts`, adapted to the WebGPU viewer's
// change signals. It is deliberately free of GPU/DOM types so it is trivially unit-testable without a
// device — the repo's "build behind the seam, test standalone, wire later" discipline (same as
// `SectionVisibility`/`OcclusionCuller`). `SchematicViewer` composes these with its live signals
// (camera change-version, `Scheduler.hasPendingWork`, `AtlasManager.hasAnimated`, …) behind the
// default-OFF `dynamicFps` flag, so the continuous-rAF baseline (which the integration/harness baselines
// rely on) is untouched until a caller opts in.

/**
 * The reasons the loop should keep drawing EVERY frame (vsync cadence) rather than idling. Any one being
 * true pins the loop "continuous": an explicit app reason (the sim running, an active orbit drag, held pan
 * keys), pending mesh work that will commit new geometry within a frame or two, or an active animation.
 */
export interface ContinuousState {
	/** Count of app-pinned reasons (e.g. "simulation", "orbit-drag", "pan"). >0 ⇒ keep drawing. */
	reasonCount: number;
	/** Mesh jobs are queued or off-thread in-flight — keep drawing until they commit (the off-thread result
	 *  lands ≤1 frame later via the worker reply, which has no other wake path). */
	meshPending: boolean;
	/** Animations are ENABLED and something is actually animating this frame (animated textures advancing,
	 *  an explosion playing, a piston sliding). Always false when the animations master switch is off. */
	animating: boolean;
}

/** Whether anything wants the loop pinned to continuous (every-frame) rendering. */
export function isContinuous(s: ContinuousState): boolean {
	return s.reasonCount > 0 || s.meshPending || s.animating;
}

/**
 * Whether THIS frame should be drawn. A frame draws when the scene is explicitly dirty (`needsRender`, set
 * by `invalidate()` after an edit/diff/overlay/brightness change), when the camera moved since the last
 * drawn frame, or while something is continuously animating/pending.
 */
export function shouldRenderFrame(s: {
	needsRender: boolean;
	cameraMoved: boolean;
	continuous: boolean;
}): boolean {
	return s.needsRender || s.cameraMoved || s.continuous;
}

/**
 * How to schedule the NEXT tick once this one finishes.
 * - `raf`: something is still animating/pending → re-arm a vsync-aligned `requestAnimationFrame`.
 * - `idle`: nothing wants drawing → drop to a slow `setTimeout` backstop so the browser frame pipeline
 *    (style/layout/composite) goes quiet. `invalidate()`/`wake()` cancel the idle timer and re-arm a frame
 *    the instant a real change arrives, so the idle poll is only a safety net for direct mutations that
 *    bypass `invalidate()`.
 *
 * Note `needsRender`/`cameraMoved` are intentionally NOT inputs: by the time we pick the next schedule we
 * have already drawn (clearing the dirty flag and snapshotting the camera), so only an ongoing `continuous`
 * reason justifies staying on rAF.
 */
export type FrameSchedule = "raf" | "idle";

export function pickFrameSchedule(s: { continuous: boolean }): FrameSchedule {
	return s.continuous ? "raf" : "idle";
}

/**
 * `setTimeout` delay (ms) for the idle backstop. A non-positive fps falls back to a 1s poll so a
 * misconfigured rate can never busy-loop.
 */
export function idlePollDelayMs(idleFps: number): number {
	return idleFps > 0 ? 1000 / idleFps : 1000;
}
