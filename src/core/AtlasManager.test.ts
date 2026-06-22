// Atlas animation: stable UVs + in-place per-frame pixel re-upload. RENDERER_PLAN.md §7 (TRAP 7.A/13.C).

import { describe, it, expect } from "vitest";
import { AtlasManager } from "./AtlasManager";
import { FakeGraphicsDevice } from "./testing/FakeGraphicsDevice";
import type { GpuTextureHandle, TextureRegion } from "./GraphicsDevice";

interface Write {
  data: Uint8Array;
  region?: TextureRegion;
}

/** Records every texture upload (copying the bytes, since the atlas mutates its buffer in place). */
class RecordingDevice extends FakeGraphicsDevice {
  textureWrites: Write[] = [];
  override writeTexture(_h: GpuTextureHandle, data: ArrayBufferView | ImageBitmap, _mip?: number, region?: TextureRegion): void {
    if (ArrayBuffer.isView(data)) this.textureWrites.push({ data: new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)), region });
  }
}

const fill = (n: number, rgba: [number, number, number, number]): Uint8Array => {
  const out = new Uint8Array(n * n * 4);
  for (let i = 0; i < n * n; i++) out.set(rgba, i * 4);
  return out;
};

/** Most recent texture upload (ES2020 target has no Array.prototype.at). */
const lastWrite = (dev: RecordingDevice): Write => dev.textureWrites[dev.textureWrites.length - 1];

/** Top-left pixel [r,g,b] of a sprite's cell in an upload. A per-cell sub-rect upload (AUDIT #4) puts the
 *  cell's top-left at pixel 0; a full-atlas upload (build) indexes by the sprite's UV-derived cell pos. */
const cellPixel = (atlas: AtlasManager, w: Write, name: string): number[] => {
  if (w.region) return [w.data[0], w.data[1], w.data[2]];
  const [W, H] = atlas.size;
  const r = atlas.uvFor(name);
  const x = Math.round(r.u * W - 0.5);
  const y = Math.round(r.v * H - 0.5);
  const o = (y * W + x) * 4;
  return [w.data[o], w.data[o + 1], w.data[o + 2]];
};

describe("AtlasManager mixed tile sizes", () => {
  it("packs mixed-resolution sprites at NATIVE size (vanilla Stitcher; no throw, no upscaling)", () => {
    const dev = new RecordingDevice();
    const atlas = new AtlasManager(dev);
    expect(() =>
      atlas.build([
        { name: "small", size: 2, rgba: fill(2, [255, 0, 0, 255]), frameCount: 1 },
        { name: "big", size: 4, rgba: fill(4, [0, 255, 0, 255]), frameCount: 1 },
      ]),
    ).not.toThrow();
    expect(atlas.uvFor("small")).toBeDefined();
    expect(atlas.uvFor("big")).toBeDefined();
    // Native sizes preserved → the 4px sprite covers a larger normalized rect than the 2px one.
    expect(atlas.uvFor("big").width).toBeGreaterThan(atlas.uvFor("small").width);
    const [w, h] = atlas.size;
    expect(w).toBeGreaterThanOrEqual(4);
    expect(h).toBeGreaterThanOrEqual(4);
  });
});

describe("AtlasManager animation", () => {
  it("animated sprites keep STABLE UVs while re-uploading the frame's pixels", () => {
    const dev = new RecordingDevice();
    const atlas = new AtlasManager(dev);
    // One animated sprite (2 frames: red → green) + one static (blue). Tile 2px.
    const animFrames = new Uint8Array([...fill(2, [200, 0, 0, 255]), ...fill(2, [0, 200, 0, 255])]);
    atlas.build([
      { name: "anim", size: 2, rgba: animFrames, frameCount: 2, frameTimeTicks: 1 },
      { name: "static", size: 2, rgba: fill(2, [0, 0, 200, 255]), frameCount: 1 },
    ]);

    expect(atlas.isAnimated("anim")).toBe(true);
    expect(atlas.isAnimated("static")).toBe(false);

    const rect = atlas.uvFor("anim");
    const writesAfterBuild = dev.textureWrites.length; // 1 (the initial upload)
        expect(cellPixel(atlas, lastWrite(dev), "anim")).toEqual([200, 0, 0]); // frame 0 = red

    atlas.tick(1); // advance to frame 1
    expect(dev.textureWrites.length).toBe(writesAfterBuild + 1); // re-uploaded
    expect(atlas.uvFor("anim")).toEqual(rect); // UVs are STABLE — vertices never re-baked (TRAP 13.C)
    // AUDIT #4: the tick upload is the sprite's 2×2 CELL sub-rect, not the whole atlas.
    expect(lastWrite(dev).region).toEqual({ x: expect.any(Number), y: expect.any(Number), width: 2, height: 2 });
    expect(lastWrite(dev).data.length).toBe(2 * 2 * 4);
    expect(cellPixel(atlas, lastWrite(dev), "anim")).toEqual([0, 200, 0]); // frame 1 = green

    atlas.tick(1); // wrap back to frame 0
    expect(cellPixel(atlas, lastWrite(dev), "anim")).toEqual([200, 0, 0]);
  });

  it("respects frametime: a frame advances only after `frametime` DELTA ticks accumulate", () => {
    const dev = new RecordingDevice();
    const atlas = new AtlasManager(dev);
    // frametime 2 ⇒ 2 ticks (100ms at 20 TPS) per frame. tick() receives DELTA ticks, not cumulative.
    const frames = new Uint8Array([...fill(2, [200, 0, 0, 255]), ...fill(2, [0, 200, 0, 255])]);
    atlas.build([{ name: "anim", size: 2, rgba: frames, frameCount: 2, frameTimeTicks: 2 }]);
    const writes = dev.textureWrites.length;
    
    atlas.tick(1); // 1 of 2 ticks — not enough to advance
    expect(dev.textureWrites.length).toBe(writes);
    expect(cellPixel(atlas, lastWrite(dev), "anim")).toEqual([200, 0, 0]);

    atlas.tick(1); // accumulates to 2 → advance to frame 1
    expect(dev.textureWrites.length).toBe(writes + 1);
    expect(cellPixel(atlas, lastWrite(dev), "anim")).toEqual([0, 200, 0]);
  });

  it("cross-fades between frames when `interpolate` is set (AUDIT M2)", () => {
    const dev = new RecordingDevice();
    const atlas = new AtlasManager(dev);
    // frame 0 = red, frame 1 = green; frametime 2, interpolate → blend across the 2 sub-frame ticks.
    const frames = new Uint8Array([...fill(2, [200, 0, 0, 255]), ...fill(2, [0, 200, 0, 255])]);
    atlas.build([{ name: "magma", size: 2, rgba: frames, frameCount: 2, frameTimeTicks: 2, interpolate: true }]);
    const writes = dev.textureWrites.length;
    expect(cellPixel(atlas, lastWrite(dev), "magma")).toEqual([200, 0, 0]); // frame 0

    atlas.tick(1); // sub-frame 1/2 → halfway blend frame0→frame1 (a hard swap would still show frame 0)
    expect(dev.textureWrites.length).toBe(writes + 1); // interpolation re-uploads every tick
    expect(cellPixel(atlas, lastWrite(dev), "magma")).toEqual([100, 100, 0]);

    atlas.tick(1); // accumulates to frame 1 at sub-frame 0 → exactly frame 1
    expect(cellPixel(atlas, lastWrite(dev), "magma")).toEqual([0, 200, 0]);
  });

  it("follows an explicit frame ORDER (lava-style ping-pong) instead of plain 0..N", () => {
    const dev = new RecordingDevice();
    const atlas = new AtlasManager(dev);
    // 3 distinct frames (red/green/blue) played as a ping-pong: 0 → 1 → 2 → 1 → (wrap) 0 …
    const frames = new Uint8Array([...fill(2, [200, 0, 0, 255]), ...fill(2, [0, 200, 0, 255]), ...fill(2, [0, 0, 200, 255])]);
    atlas.build([{ name: "lava", size: 2, rgba: frames, frameCount: 3, frameTimeTicks: 1, frameOrder: [0, 1, 2, 1] }]);
        expect(cellPixel(atlas, lastWrite(dev), "lava")).toEqual([200, 0, 0]); // order[0] = frame 0 (red)

    const seen: number[][] = [];
    for (let i = 0; i < 4; i++) { atlas.tick(1); seen.push(cellPixel(atlas, lastWrite(dev), "lava")); }
    // Steps visit order[1..3] then wrap to order[0]: green, blue, green, red — no 2→0 hard jump.
    expect(seen).toEqual([[0, 200, 0], [0, 0, 200], [0, 200, 0], [200, 0, 0]]);
  });

  it("a still sprite never triggers a re-upload on tick", () => {
    const dev = new RecordingDevice();
    const atlas = new AtlasManager(dev);
    atlas.build([{ name: "s", size: 2, rgba: fill(2, [10, 20, 30, 255]), frameCount: 1 }]);
    const writes = dev.textureWrites.length;
    atlas.tick(100);
    expect(dev.textureWrites.length).toBe(writes); // no animated sprite → no upload
  });
});

describe("AtlasManager ATL-1 incremental append", () => {
  it("appends a new static sprite WITHOUT moving existing UVs (TRAP 1.C) or recreating the texture", () => {
    const dev = new RecordingDevice();
    const atlas = new AtlasManager(dev);
    atlas.build([
      { name: "a", size: 4, rgba: fill(4, [10, 20, 30, 255]), frameCount: 1 },
      { name: "b", size: 4, rgba: fill(4, [40, 50, 60, 255]), frameCount: 1 },
    ]);
    const uvA = atlas.uvFor("a");
    const uvB = atlas.uvFor("b");
    const tex = atlas.texture;
    const [W, H] = atlas.size;

    expect(atlas.append([{ name: "c", size: 4, rgba: fill(4, [7, 8, 9, 255]), frameCount: 1 }])).toBe(true);

    expect(atlas.uvFor("a")).toEqual(uvA); // existing UVs UNCHANGED — already-baked geometry stays valid
    expect(atlas.uvFor("b")).toEqual(uvB);
    expect(atlas.texture).toBe(tex); // SAME GPU texture (sub-rect upload, no recreate → bind group stable)
    expect(atlas.size).toEqual([W, H]); // dimensions fixed at build (headroom reserved)
    expect(atlas.has("c")).toBe(true);
    const w = lastWrite(dev);
    // Upload covers the 4×4 cell PLUS its 1px right/bottom gutter (edge-to-edge UVs mirror the border into
    // the gutter so a face UV reaching exactly 1.0 stays clean) ⇒ a 5×5 sub-rect from the CPU mirror.
    expect(w.region).toEqual({ x: expect.any(Number), y: expect.any(Number), width: 5, height: 5 });
    expect([w.data[0], w.data[1], w.data[2]]).toEqual([7, 8, 9]); // c's pixels uploaded (cell top-left)
  });

  it("rejects an animated sprite so the caller falls back to a full rebuild", () => {
    const dev = new RecordingDevice();
    const atlas = new AtlasManager(dev);
    atlas.build([{ name: "a", size: 2, rgba: fill(2, [1, 2, 3, 255]), frameCount: 1 }]);
    const frames = new Uint8Array([...fill(2, [200, 0, 0, 255]), ...fill(2, [0, 200, 0, 255])]);
    expect(atlas.append([{ name: "anim", size: 2, rgba: frames, frameCount: 2 }])).toBe(false);
    expect(atlas.has("anim")).toBe(false);
  });

  it("returns false when the headroom strip is exhausted (no overflow / corruption)", () => {
    const dev = new RecordingDevice();
    const atlas = new AtlasManager(dev);
    atlas.build([{ name: "a", size: 4, rgba: fill(4, [1, 2, 3, 255]), frameCount: 1 }]);
    const [W] = atlas.size;
    let appended = 0;
    let i = 0;
    for (; i < 10000; i++) {
      if (!atlas.append([{ name: `s${i}`, size: 4, rgba: fill(4, [i & 255, 0, 0, 255]), frameCount: 1 }])) break;
      appended++;
    }
    expect(appended).toBeGreaterThan(0); // some fit
    expect(i).toBeLessThan(10000); // it DID stop (returned false) — no infinite growth
    const cap = Math.floor(256 / 4) * Math.floor(W / 4); // shelves × per-shelf
    expect(appended).toBeLessThanOrEqual(cap);
  });
});
