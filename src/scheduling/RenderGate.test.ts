import { describe, it, expect } from "vitest";
import { isContinuous, shouldRenderFrame, pickFrameSchedule, idlePollDelayMs } from "./RenderGate";

describe("RenderGate.isContinuous", () => {
	it("is false only when nothing is pinned, pending, or animating", () => {
		expect(isContinuous({ reasonCount: 0, meshPending: false, animating: false })).toBe(false);
	});
	it("any single reason pins it continuous", () => {
		expect(isContinuous({ reasonCount: 1, meshPending: false, animating: false })).toBe(true);
		expect(isContinuous({ reasonCount: 0, meshPending: true, animating: false })).toBe(true);
		expect(isContinuous({ reasonCount: 0, meshPending: false, animating: true })).toBe(true);
	});
});

describe("RenderGate.shouldRenderFrame", () => {
	it("draws on dirty, camera-move, or continuous; idles otherwise", () => {
		expect(shouldRenderFrame({ needsRender: false, cameraMoved: false, continuous: false })).toBe(false);
		expect(shouldRenderFrame({ needsRender: true, cameraMoved: false, continuous: false })).toBe(true);
		expect(shouldRenderFrame({ needsRender: false, cameraMoved: true, continuous: false })).toBe(true);
		expect(shouldRenderFrame({ needsRender: false, cameraMoved: false, continuous: true })).toBe(true);
	});
});

describe("RenderGate.pickFrameSchedule", () => {
	it("stays on rAF while continuous, idles otherwise", () => {
		expect(pickFrameSchedule({ continuous: true })).toBe("raf");
		expect(pickFrameSchedule({ continuous: false })).toBe("idle");
	});
});

describe("RenderGate.idlePollDelayMs", () => {
	it("converts fps to a ms delay", () => {
		expect(idlePollDelayMs(4)).toBe(250);
		expect(idlePollDelayMs(20)).toBe(50);
	});
	it("falls back to a 1s backstop on a non-positive fps", () => {
		expect(idlePollDelayMs(0)).toBe(1000);
		expect(idlePollDelayMs(-5)).toBe(1000);
	});
});
