// renderInterleaved — the fix for "entities paint over glass". Terrain is split into TWO passes (opaque,
// then translucent) with an external callback (entities) BETWEEN them, so entities depth-order against glass
// (vanilla order: opaque terrain → entities → translucent terrain). F1 is preserved with a SEPARATE bundle
// per pass. This pins the interleave ORDER and the split-bundle record/replay state machine.

import { describe, it, expect } from "vitest";
import { TerrainRenderer, type SectionDraw } from "./TerrainRenderer";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { AtlasManager } from "../core/AtlasManager";
import { BufferUsage, TextureFormat } from "../core/GraphicsDevice";
import { SortType } from "../mesh/SortTypes";
import { Camera } from "../camera/Camera";
import { TerrainPass, type Vec3 } from "../types";
import type { CameraView } from "../camera/Camera";
import type { Mat4 } from "../camera/math";

function tinyAtlas(device: FakeGraphicsDevice): AtlasManager {
  const atlas = new AtlasManager(device);
  atlas.build([{ name: "x", size: 2, rgba: new Uint8Array(2 * 2 * 4).fill(255), frameCount: 1 }]);
  return atlas;
}

/** A camera whose permissive viewProjection keeps a section at the origin in frustum (constant ⇒ "settled"). */
function fakeCamera(pos: Vec3): CameraView {
  const vp = new Float32Array(16);
  vp[0] = 0.05; vp[5] = 0.05; vp[10] = 0.05; vp[15] = 1;
  return { position: pos, aspect: 1, viewProjection: () => vp as unknown as Mat4 };
}

function quad(device: FakeGraphicsDevice, pass: TerrainPass, over: Partial<SectionDraw> = {}): SectionDraw {
  return {
    originBlocks: [0, 0, 0],
    vertex: device.createBuffer({ sizeBytes: 4 * 20, usage: BufferUsage.Vertex }),
    quadCount: 1,
    pass,
    ...over,
  };
}

function makeRenderer(device: FakeGraphicsDevice): TerrainRenderer {
  return new TerrainRenderer(device, tinyAtlas(device), TextureFormat.Bgra8Srgb, TextureFormat.Depth24Plus);
}

describe("TerrainRenderer.renderInterleaved (entities between opaque and translucent terrain)", () => {
  it("draws opaque terrain, then runs the callback, then translucent — and never drops a translucent draw", () => {
    const device = new FakeGraphicsDevice();
    const r = makeRenderer(device);
    r.bundleReplay = false; // force direct encode so the live pass records drawPipelines
    const solid = quad(device, TerrainPass.Solid);
    const terrainGlass = quad(device, TerrainPass.Translucent);
    const movingGlass = quad(device, TerrainPass.Translucent, {
      index: device.createBuffer({ sizeBytes: 6 * 4, usage: BufferUsage.Index }),
      sortType: SortType.StaticTopo,
      depthWrite: true,
    });

    let callbackCount = 0;
    let pipelinesAtCallback: string[] = [];
    r.renderInterleaved([terrainGlass, movingGlass, solid], fakeCamera([8, 8, 30]), 64, 64, 0, () => {
      callbackCount++;
      pipelinesAtCallback = device.log.drawPipelines.slice(); // everything drawn BEFORE the callback (= opaque only)
    });

    expect(callbackCount).toBe(1); // the entity callback runs exactly once, between the two terrain passes
    expect(pipelinesAtCallback).toEqual(["terrain-solid"]); // only opaque terrain is drawn before entities
    // After the callback: the moving-glass (depth-write) then terrain translucent draw — all 3 terrain draws survive.
    expect(device.log.drawPipelines).toEqual([
      "terrain-solid",
      "terrain-translucent-depthwrite",
      "terrain-translucent",
    ]);
  });

  it("runs the callback even with no terrain at all (entities still render over the cleared frame)", () => {
    const device = new FakeGraphicsDevice();
    const r = makeRenderer(device);
    let called = 0;
    r.renderInterleaved([], fakeCamera([8, 8, 30]), 64, 64, 0, () => { called++; });
    expect(called).toBe(1);
    expect(device.log.drawCalls).toBe(0); // no terrain drawn; the clear pass + (empty) entity callback
  });

  it("F1 split bundles: records two bundles once the camera settles, then replays them (callback runs every frame)", () => {
    const device = new FakeGraphicsDevice();
    const r = makeRenderer(device);
    const camera = new Camera();
    camera.target = [8, 8, 8];
    const draws: SectionDraw[] = [
      quad(device, TerrainPass.Solid),
      quad(device, TerrainPass.Translucent, { index: device.createBuffer({ sizeBytes: 6 * 4, usage: BufferUsage.Index }), sortType: SortType.StaticTopo, depthWrite: true }),
    ];
    let calls = 0;
    const frame = () => r.renderInterleaved(draws, camera, 256, 256, 0, () => { calls++; });

    frame(); // frame 1: first ever → not settled → direct encode, no bundles
    expect(device.log.bundlesCreated).toBe(0);
    expect(r.bundleRecords).toBe(0);

    frame(); // frame 2: settled (same draws + camera) → RECORD two bundles (opaque + translucent) + execute
    expect(device.log.bundlesCreated).toBe(2);
    expect(r.bundleRecords).toBe(1);
    expect(device.log.bundlesExecuted).toBe(2);

    frame(); // frame 3: REPLAY both bundles (camera still bit-static)
    frame(); // frame 4: REPLAY
    expect(device.log.bundlesCreated).toBe(2); // never re-recorded while static
    expect(r.bundleReplays).toBe(2);
    expect(calls).toBe(4); // the entity callback runs EVERY frame (entities re-render even when terrain replays)

    // Rotating every frame drops the bundles and never re-records (the moving-camera regression case).
    const recordsBefore = device.log.bundlesCreated;
    for (let i = 0; i < 3; i++) { camera.yaw += 0.1; frame(); }
    expect(device.log.bundlesCreated).toBe(recordsBefore); // a moving camera pays no record cost
    expect(calls).toBe(7);
  });

  it("a fresh draws array invalidates the split bundles (TRAP 4.A — no stale replay)", () => {
    const device = new FakeGraphicsDevice();
    const r = makeRenderer(device);
    const camera = new Camera();
    camera.target = [8, 8, 8];
    const draws: SectionDraw[] = [quad(device, TerrainPass.Solid)];
    const noop = () => {};
    r.renderInterleaved(draws, camera, 256, 256, 0, noop); // direct
    r.renderInterleaved(draws, camera, 256, 256, 0, noop); // record
    r.renderInterleaved(draws, camera, 256, 256, 0, noop); // replay
    expect(r.bundleReplays).toBe(1);

    const fresh = draws.map((d) => ({ ...d })); // a commit/relocation/occlusion change always mints a fresh array
    const replaysBefore = r.bundleReplays;
    r.renderInterleaved(fresh, camera, 256, 256, 0, noop);
    expect(r.bundleReplays).toBe(replaysBefore); // did NOT replay the stale bundles → direct-encoded the new array
  });
});
