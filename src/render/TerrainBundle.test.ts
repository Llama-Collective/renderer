// F1 — adaptive render-bundle replay. The bundle is RECORDED only after the camera settles (a frame whose
// (draws, viewProj) matched the previous frame) and REPLAYED while both stay identical; ANY change (camera
// motion OR a fresh draw-list array — which a commit/relocation/occlusion change always produces) drops it.
// This guarantees a moving/rotating camera (the regression case) never pays the record cost, and a stale
// bundle is never replayed against a relocated buffer (TRAP 4.A).

import { describe, it, expect } from "vitest";
import { TerrainRenderer, type SectionDraw } from "./TerrainRenderer";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { AtlasManager } from "../core/AtlasManager";
import { Camera } from "../camera/Camera";
import { BufferUsage, TextureFormat } from "../core/GraphicsDevice";
import { TerrainPass } from "../types";

function tinyAtlas(device: FakeGraphicsDevice): AtlasManager {
  const atlas = new AtlasManager(device);
  atlas.build([{ name: "x", size: 2, rgba: new Uint8Array(2 * 2 * 4).fill(255), frameCount: 1 }]);
  return atlas;
}

function setup() {
  const device = new FakeGraphicsDevice();
  const renderer = new TerrainRenderer(device, tinyAtlas(device), TextureFormat.Bgra8Srgb, TextureFormat.Depth24Plus);
  const camera = new Camera();
  camera.target = [8, 8, 8]; // a section at origin [0,16]³ sits dead-centre in view → passes the frustum
  // One opaque draw for the section at world origin.
  const draws: SectionDraw[] = [
    { originBlocks: [0, 0, 0], vertex: device.createBuffer({ sizeBytes: 80, usage: BufferUsage.Vertex }), quadCount: 1, pass: TerrainPass.Solid },
  ];
  return { device, renderer, camera, draws };
}

describe("TerrainRenderer F1 adaptive render-bundle replay", () => {
  it("records only after the camera settles, then replays — and a moving camera never records", () => {
    const { device, renderer, camera, draws } = setup();

    // Frame 1: first ever frame → no prior frame to match → direct encode, no bundle.
    renderer.render(draws, camera, 256, 256);
    expect(device.log.drawCalls).toBe(1); // the section was in view (frustum didn't cull it)
    expect(device.log.bundlesCreated).toBe(0);
    expect(renderer.bundleRecords).toBe(0);

    // Frame 2: identical (draws, camera) as frame 1 → camera settled → RECORD a bundle + execute it.
    renderer.render(draws, camera, 256, 256);
    expect(device.log.bundlesCreated).toBe(1);
    expect(renderer.bundleRecords).toBe(1);
    expect(device.log.bundlesExecuted).toBe(1);
    expect(device.log.drawCalls).toBe(2); // replay still counts the logical draw (metric consistency)

    // Frames 3-5: still static → REPLAY (no new record).
    renderer.render(draws, camera, 256, 256);
    renderer.render(draws, camera, 256, 256);
    renderer.render(draws, camera, 256, 256);
    expect(device.log.bundlesCreated).toBe(1); // never re-recorded while static
    expect(renderer.bundleReplays).toBe(3);
    expect(device.log.drawCalls).toBe(5); // +1 each replayed frame

    // Now ROTATE every frame (the regression case): viewProj changes each frame → the bundle is dropped and
    // NEVER re-recorded (each frame differs from the previous one).
    const recordsBefore = device.log.bundlesCreated;
    for (let i = 0; i < 5; i++) {
      camera.yaw += 0.1; // rotate in place
      renderer.render(draws, camera, 256, 256);
    }
    expect(device.log.bundlesCreated).toBe(recordsBefore); // a moving camera pays ZERO record cost
    expect(device.log.drawCalls).toBe(10); // still exactly +1 logical draw per frame (direct encode)
  });

  it("invalidates the bundle when the draw-list ARRAY changes (commit/relocation/occlusion → fresh array)", () => {
    const { device, renderer, camera, draws } = setup();
    renderer.render(draws, camera, 256, 256); // frame 1: direct
    renderer.render(draws, camera, 256, 256); // frame 2: record + execute
    renderer.render(draws, camera, 256, 256); // frame 3: replay
    expect(device.log.bundlesCreated).toBe(1);
    expect(renderer.bundleReplays).toBe(1);

    // A fresh draws array (same content) — exactly what DrawList/occlusion produce on any change. Even with
    // a bit-static camera, the bundle must NOT replay (it could reference relocated buffers): direct encode.
    const draws2 = draws.map((d) => ({ ...d }));
    const replaysBefore = renderer.bundleReplays;
    renderer.render(draws2, camera, 256, 256);
    expect(renderer.bundleReplays).toBe(replaysBefore); // did NOT replay the stale bundle
    expect(device.log.drawCalls).toBe(4); // direct-encoded the new array (+1)

    // Settles again on the new array → records a fresh bundle.
    renderer.render(draws2, camera, 256, 256);
    expect(device.log.bundlesCreated).toBe(2);
  });

  it("a layout-revision bump invalidates the bundle even on the SAME draws array (INFRA-6)", () => {
    const { device, renderer, camera, draws } = setup();
    renderer.render(draws, camera, 256, 256, 0); // frame 1: direct (layoutRev 0)
    renderer.render(draws, camera, 256, 256, 0); // frame 2: record
    renderer.render(draws, camera, 256, 256, 0); // frame 3: replay
    expect(renderer.bundleReplays).toBe(1);
    expect(device.log.bundlesCreated).toBe(1);

    // Same draws array + bit-static camera, but the arena RELOCATED (layoutRev bumped to 1). The bundle's
    // baked (buffer, offset) snapshots are now stale → it must NOT replay, even though the array identity
    // is unchanged (the TRAP-4.A corner the array-identity key alone would miss).
    renderer.render(draws, camera, 256, 256, 1);
    expect(renderer.bundleReplays).toBe(1); // did NOT replay the stale-layout bundle
    expect(device.log.bundlesCreated).toBe(2); // re-recorded against the new layoutRev (camera still settled)

    // Stable on the new layoutRev → replays again (the F1 win is preserved once layout settles).
    renderer.render(draws, camera, 256, 256, 1);
    expect(renderer.bundleReplays).toBe(2);
  });

  it("a constant layoutRev still hits replay (no regression to the F1 win)", () => {
    const { device, renderer, camera, draws } = setup();
    renderer.render(draws, camera, 256, 256, 7);
    renderer.render(draws, camera, 256, 256, 7); // record
    renderer.render(draws, camera, 256, 256, 7); // replay
    renderer.render(draws, camera, 256, 256, 7); // replay
    expect(renderer.bundleReplays).toBe(2);
    expect(device.log.bundlesCreated).toBe(1); // recorded once; the constant layout never invalidated it
  });

  it("bundleReplay=false keeps the pure direct-encode path (no bundles ever)", () => {
    const { device, renderer, camera, draws } = setup();
    renderer.bundleReplay = false;
    for (let i = 0; i < 5; i++) renderer.render(draws, camera, 256, 256);
    expect(device.log.bundlesCreated).toBe(0);
    expect(device.log.bundlesExecuted).toBe(0);
    expect(device.log.drawCalls).toBe(5); // one direct draw per frame
  });

  it("PREC-1: cameraRelative=true still records + replays (the relative-vp upload does not break the F1 key)", () => {
    // Settled camera at large world coordinates, with the section in view, and the camera-relative path ON.
    // The F1 key compares the UPLOADED matrix (viewProjRel) to itself across frames, so a bit-static camera
    // (⇒ static camOrigin ⇒ static viewProjRel) records once and replays — the F1 win survives PREC-1.
    const device = new FakeGraphicsDevice();
    const renderer = new TerrainRenderer(device, tinyAtlas(device), TextureFormat.Bgra8Srgb, TextureFormat.Depth24Plus);
    const camera = new Camera();
    const FAR = 30_000_000;
    camera.target = [FAR + 8, 8, 8];
    const draws: SectionDraw[] = [
      { originBlocks: [FAR, 0, 0], vertex: device.createBuffer({ sizeBytes: 80, usage: BufferUsage.Vertex }), quadCount: 1, pass: TerrainPass.Solid },
    ];
    renderer.cameraRelative = true;

    renderer.render(draws, camera, 256, 256); // frame 1: direct
    expect(renderer.stats.drawn).toBe(1);     // section in view
    renderer.render(draws, camera, 256, 256); // frame 2: settle → RECORD
    expect(device.log.bundlesCreated).toBe(1);
    expect(renderer.bundleRecords).toBe(1);
    renderer.render(draws, camera, 256, 256); // frame 3: REPLAY
    renderer.render(draws, camera, 256, 256); // frame 4: REPLAY
    expect(device.log.bundlesCreated).toBe(1); // never re-recorded while static
    expect(renderer.bundleReplays).toBe(2);

    // Moving the camera (the relative vp changes too) drops the bundle — same behaviour as the world-vp path.
    const before = device.log.bundlesCreated;
    camera.yaw += 0.1;
    renderer.render(draws, camera, 256, 256);
    expect(device.log.bundlesCreated).toBe(before); // a moving camera pays no record cost
  });
});
