// scheduling/ — priority/aging/budget orchestration (the frame conductor). RENDERER_PLAN §16.

export { Scheduler } from "./Scheduler";
export { FrameTimeEma } from "./FrameTimeEma"; // P3: part of the scheduling subsystem (ABP-1) — was deep-imported
export * from "./RenderGate"; // render-on-demand ("dynamic FPS") decision helpers
