// The pipeline cache key must cover everything that changes the compiled object. AUDIT M3.

import { describe, it, expect } from "vitest";
import { PipelineCache } from "./PipelineCache";
import { FakeGraphicsDevice } from "./testing/FakeGraphicsDevice";
import { TextureFormat, type PassStateDesc, type PipelineDesc, type VertexLayout } from "./GraphicsDevice";

const LAYOUT: VertexLayout = { strideBytes: 20, attributes: [] };
const PASS: PassStateDesc = { depthTest: true, depthWrite: true, blend: false };

describe("PipelineCache key", () => {
  it("hits on an identical desc but misses on a different shader, layout, or constants", () => {
    const device = new FakeGraphicsDevice();
    const cache = new PipelineCache(device);
    const sa = device.createShaderModule("a");
    const sb = device.createShaderModule("b");
    const base: PipelineDesc = { label: "p", shader: sa, vertexLayout: LAYOUT, pass: PASS, colorFormat: TextureFormat.Rgba8Srgb };

    const p1 = cache.get(base);
    expect(cache.get({ ...base })).toBe(p1); // identical → cache HIT
    expect(cache.size).toBe(1);

    // SAME label, DIFFERENT shader → must NOT collide (the bug: a label-only key returned p1's pipeline).
    expect(cache.get({ ...base, shader: sb })).not.toBe(p1);
    // SAME label + shader, DIFFERENT vertex layout → different pipeline.
    expect(cache.get({ ...base, vertexLayout: { strideBytes: 24, attributes: [] } })).not.toBe(p1);
    // DIFFERENT override constants → different pipeline.
    expect(cache.get({ ...base, constants: { ALPHA_TEST: 1 } })).not.toBe(p1);

    expect(cache.size).toBe(4); // base + shader + layout + constants variants
  });
});
