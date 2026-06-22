// Phase-P instrument: bumps are no-ops while disabled (production/tests pay only a branch), accumulate
// while enabled, and the device tracks live buffers + draw calls. RENDERER_OPTIMIZATION_PLAN §Phase P.

import { describe, it, expect, beforeEach } from "vitest";
import { instrument } from "./Instrument";
import { FakeGraphicsDevice } from "./testing/FakeGraphicsDevice";
import { BufferUsage } from "./GraphicsDevice";

describe("Instrument", () => {
  beforeEach(() => {
    instrument.enabled = false;
    instrument.reset();
  });

  it("bumps are no-ops while disabled", () => {
    instrument.bumpAlloc("rawVertex", 100);
    instrument.bumpJob("meshed", 5);
    expect(instrument.alloc.rawVertex).toBe(0);
    expect(instrument.job.meshed).toBe(0);
  });

  it("accumulates per category while enabled; reset() clears", () => {
    instrument.enabled = true;
    instrument.bumpAlloc("rawVertex", 64);
    instrument.bumpAlloc("rawVertex", 36);
    instrument.bumpAlloc("drawList", 12);
    instrument.bumpAlloc("frustum"); // default n = 1
    instrument.bumpJob("dirtied", 3);
    instrument.bumpJob("meshed", 3);
    expect(instrument.alloc.rawVertex).toBe(100);
    expect(instrument.alloc.drawList).toBe(12);
    expect(instrument.alloc.frustum).toBe(1);
    expect(instrument.job.dirtied).toBe(3);
    instrument.reset();
    expect(instrument.alloc.rawVertex).toBe(0);
    expect(instrument.job.dirtied).toBe(0);
  });
});

describe("device buffer/draw counters (Phase-P, via FakeGraphicsDevice)", () => {
  it("tracks live buffers (created − destroyed) and draw calls", () => {
    const dev = new FakeGraphicsDevice();
    const a = dev.createBuffer({ sizeBytes: 16, usage: BufferUsage.Vertex });
    const b = dev.createBuffer({ sizeBytes: 16, usage: BufferUsage.Index });
    expect(dev.liveBufferCount()).toBe(2);
    dev.destroyBuffer(a);
    expect(dev.liveBufferCount()).toBe(1);
    expect(dev.log.buffersCreated).toBe(2);
    expect(dev.log.buffersDestroyed).toBe(1);

    const pass = dev.beginPass({ id: null, width: 4, height: 4 }, { color: [0, 0, 0, 1], depth: 1 });
    pass.drawIndexed(6, 0, 0);
    pass.drawIndexed(6, 0, 0);
    pass.end();
    expect(dev.log.drawCalls).toBe(2);
    dev.destroyBuffer(b);
  });
});
