// BUG_REPORT.md fix — the moving-piston translucent draw is its own depth-WRITING pass (vanilla
// `translucentMovingBlock`), drawn AFTER opaque/cutout but BEFORE terrain translucent, and it never
// removes a terrain section. This pins the encode ORDER (via the recording FakeGraphicsDevice's
// per-draw pipeline labels) and the depth-write pipeline state.

import { describe, it, expect } from "vitest";
import { TerrainRenderer, type SectionDraw } from "./TerrainRenderer";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { AtlasManager } from "../core/AtlasManager";
import { BufferUsage, CompareFn, TextureFormat } from "../core/GraphicsDevice";
import { SortType } from "../mesh/SortTypes";
import { TerrainPass, type Vec3 } from "../types";
import type { CameraView } from "../camera/Camera";
import type { Mat4 } from "../camera/math";

function tinyAtlas(device: FakeGraphicsDevice): AtlasManager {
  const atlas = new AtlasManager(device);
  atlas.build([{ name: "x", size: 2, rgba: new Uint8Array(2 * 2 * 4).fill(255), frameCount: 1 }]);
  return atlas;
}

/** A camera whose permissive viewProjection keeps a section at the origin in frustum. */
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

function renderer(device: FakeGraphicsDevice): TerrainRenderer {
  const r = new TerrainRenderer(device, tinyAtlas(device), TextureFormat.Bgra8Srgb, TextureFormat.Depth24Plus);
  r.bundleReplay = false; // force direct encode so the live pass records drawPipelines
  return r;
}

describe("moving-translucent transient pass (BUG_REPORT.md)", () => {
  it("draws SOLID → moving-translucent (depth-write) → terrain translucent, keeping every terrain draw", () => {
    const device = new FakeGraphicsDevice();
    const r = renderer(device);
    const solid = quad(device, TerrainPass.Solid);
    const terrainGlass = quad(device, TerrainPass.Translucent); // a static terrain glass section (depth-write OFF)
    const movingGlass = quad(device, TerrainPass.Translucent, {
      index: device.createBuffer({ sizeBytes: 6 * 4, usage: BufferUsage.Index }),
      sortType: SortType.StaticTopo,
      depthWrite: true, // the moving piston block — vanilla `translucentMovingBlock`
    });

    // Deliberately shuffled input order — the renderer must reorder by pass + depth-write.
    r.render([terrainGlass, movingGlass, solid], fakeCamera([8, 8, 30]), 64, 64);

    // All three draws survive — the moving block NEVER suppresses the terrain glass section.
    expect(device.log.drawCmds).toHaveLength(3);
    expect(device.log.drawPipelines).toEqual([
      "terrain-solid",
      "terrain-translucent-depthwrite", // moving block writes depth, BEFORE terrain translucent
      "terrain-translucent",
    ]);
  });

  it("the moving-translucent pipeline writes depth with LESS_EQUAL; terrain translucent does not", () => {
    const device = new FakeGraphicsDevice();
    const r = renderer(device);
    r.render([quad(device, TerrainPass.Translucent, {
      index: device.createBuffer({ sizeBytes: 6 * 4, usage: BufferUsage.Index }),
      sortType: SortType.StaticTopo,
      depthWrite: true,
    })], fakeCamera([8, 8, 30]), 64, 64);

    const dw = device.pipelineDescByLabel("terrain-translucent-depthwrite")!;
    expect(dw.pass.depthWrite).toBe(true); // vanilla DepthStencilState.DEFAULT writeDepth=true
    expect(dw.pass.blend).toBe(true);      // still alpha-blended
    expect(dw.depthCompare).toBe(CompareFn.LessEqual);

    const terrain = device.pipelineDescByLabel("terrain-translucent")!;
    expect(terrain.pass.depthWrite).toBe(false); // terrain translucent stays depth-write OFF (TRAP 12.C)
  });

  it("a moving-translucent draw with no terrain present still renders (empty-frame guard counts it)", () => {
    const device = new FakeGraphicsDevice();
    const r = renderer(device);
    r.render([quad(device, TerrainPass.Translucent, {
      index: device.createBuffer({ sizeBytes: 6 * 4, usage: BufferUsage.Index }),
      sortType: SortType.StaticTopo,
      depthWrite: true,
    })], fakeCamera([8, 8, 30]), 64, 64);

    expect(device.log.drawCmds).toHaveLength(1);
    expect(device.log.drawPipelines).toEqual(["terrain-translucent-depthwrite"]);
  });
});
