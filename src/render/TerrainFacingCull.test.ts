// FS-4 — encodeDraws emits one drawIndexed per VISIBLE facing run (camera-away slices dropped). Uses the
// recording FakeGraphicsDevice (log.drawCmds) to pin the baseVertex (vertices) / indexCount (indices) math.

import { describe, it, expect } from "vitest";
import { TerrainRenderer, type SectionDraw } from "./TerrainRenderer";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { AtlasManager } from "../core/AtlasManager";
import { BufferUsage, TextureFormat } from "../core/GraphicsDevice";
import { TerrainPass, type Vec3 } from "../types";
import type { CameraView } from "../camera/Camera";
import type { Mat4 } from "../camera/math";

function tinyAtlas(device: FakeGraphicsDevice): AtlasManager {
  const atlas = new AtlasManager(device);
  atlas.build([{ name: "x", size: 2, rgba: new Uint8Array(2 * 2 * 4).fill(255), frameCount: 1 }]);
  return atlas;
}

/** A camera at `pos` with a permissive scale viewProjection that keeps a section at the origin in frustum. */
function fakeCamera(pos: Vec3): CameraView {
  const vp = new Float32Array(16);
  vp[0] = 0.05; vp[5] = 0.05; vp[10] = 0.05; vp[15] = 1; // column-major diagonal scale → [0,16]³ well inside clip
  return { position: pos, aspect: 1, viewProjection: () => vp as unknown as Mat4 };
}

/** One partitioned opaque section: 6 quads, one per axis facing (PosX..NegZ), UNASSIGNED empty. */
function cubeDraw(device: FakeGraphicsDevice): SectionDraw {
  return {
    originBlocks: [0, 0, 0],
    vertex: device.createBuffer({ sizeBytes: 6 * 4 * 20, usage: BufferUsage.Vertex }),
    quadCount: 6,
    pass: TerrainPass.Solid,
    facingVertexCounts: Uint32Array.of(4, 4, 4, 4, 4, 4, 0),
    sliceMask: 0b0111111,
  };
}

describe("TerrainRenderer facing-slice cull (FS-4)", () => {
  it("off (default): one draw for the whole opaque section (legacy path)", () => {
    const device = new FakeGraphicsDevice();
    const renderer = new TerrainRenderer(device, tinyAtlas(device), TextureFormat.Bgra8Srgb, TextureFormat.Depth24Plus);
    renderer.bundleReplay = false; // force direct encode so the live pass records drawCmds
    renderer.render([cubeDraw(device)], fakeCamera([100, 100, 100]), 64, 64);
    expect(device.log.drawCmds).toHaveLength(1);
    expect(device.log.drawCmds[0]).toMatchObject({ indexCount: 6 * 6, baseVertex: 0, firstInstance: 0 }); // all 6 quads
    expect(renderer.sliceDrawsCulled).toBe(0);
  });

  it("on: a camera at +++ draws only the 3 positive facings (one coalesced run) and culls the 3 negatives", () => {
    const device = new FakeGraphicsDevice();
    const renderer = new TerrainRenderer(device, tinyAtlas(device), TextureFormat.Bgra8Srgb, TextureFormat.Depth24Plus);
    renderer.bundleReplay = false;
    renderer.faceSliceCull = true;
    renderer.render([cubeDraw(device)], fakeCamera([100, 100, 100]), 64, 64);
    // Visible facings PosX(0),PosY(1),PosZ(2) are contiguous → ONE coalesced draw: verts [0,12), 3 quads.
    expect(device.log.drawCmds).toHaveLength(1);
    expect(device.log.drawCmds[0]).toMatchObject({ indexCount: 3 * 6, firstIndex: 0, baseVertex: 0, firstInstance: 0 });
    expect(renderer.sliceDrawsCulled).toBe(3); // NegX, NegY, NegZ dropped
  });

  it("on: a non-contiguous visible set splits into multiple draws at the right baseVertex", () => {
    const device = new FakeGraphicsDevice();
    const renderer = new TerrainRenderer(device, tinyAtlas(device), TextureFormat.Bgra8Srgb, TextureFormat.Depth24Plus);
    renderer.bundleReplay = false;
    renderer.faceSliceCull = true;
    // Camera far +X, far -Y, centred Z → visible: PosX(0), NegY(4), PosZ(2), NegZ(5) [Z centred].
    // Buffer runs: 0[0,4) 1[4,8) 2[8,12) 3[12,16) 4[16,20) 5[20,24). Visible {0,2,4,5}.
    renderer.render([cubeDraw(device)], fakeCamera([100, -100, 8]), 64, 64);
    const cmds = device.log.drawCmds;
    // Expected coalesced runs: {0} → baseVertex 0; {2} → baseVertex 8; {4,5} → baseVertex 16, 2 quads.
    const runs = cmds.map((c) => ({ baseVertex: c.baseVertex, quads: c.indexCount / 6 }));
    expect(runs).toEqual([
      { baseVertex: 0, quads: 1 }, // PosX
      { baseVertex: 8, quads: 1 }, // PosZ
      { baseVertex: 16, quads: 2 }, // NegY + NegZ coalesced (adjacent, both visible)
    ]);
    expect(cmds.every((c) => c.firstInstance === 0)).toBe(true); // every slice uses the section's origin slot
  });

  it("bind-once: an arena-resident section adds its baseVertex to every draw (slice runs included)", () => {
    const device = new FakeGraphicsDevice();
    const renderer = new TerrainRenderer(device, tinyAtlas(device), TextureFormat.Bgra8Srgb, TextureFormat.Depth24Plus);
    renderer.bundleReplay = false;
    renderer.faceSliceCull = true;
    // A section whose vertices live at byte offset 200 in a shared arena ⇒ baseVertex = 200/20 = 10.
    const d = cubeDraw(device);
    d.vertexOffset = 200;
    d.baseVertex = 10;
    renderer.render([d], fakeCamera([100, -100, 8]), 64, 64); // visible {PosX, PosZ, NegY, NegZ}
    const runs = device.log.drawCmds.map((c) => ({ baseVertex: c.baseVertex, quads: c.indexCount / 6 }));
    // Same runs as the offset-0 case but every baseVertex is shifted by the section base (10): 0→10, 8→18, 16→26.
    expect(runs).toEqual([
      { baseVertex: 10, quads: 1 }, // PosX  (base 10 + run 0)
      { baseVertex: 18, quads: 1 }, // PosZ  (base 10 + run 8)
      { baseVertex: 26, quads: 2 }, // NegY+NegZ (base 10 + run 16)
    ]);
  });

  it("bind-once: many sections sharing one arena buffer bind the vertex buffer ONCE, not per draw", () => {
    const device = new FakeGraphicsDevice();
    const renderer = new TerrainRenderer(device, tinyAtlas(device), TextureFormat.Bgra8Srgb, TextureFormat.Depth24Plus);
    renderer.bundleReplay = false; // direct encode so setVertexBuffer hits the recording live encoder
    // 3 opaque sections all in the SAME arena buffer at different baseVertex offsets (one shared handle). All
    // share origin [0,0,0] so the permissive fakeCamera keeps them in frustum — the point is the bind count.
    const arena = device.createBuffer({ sizeBytes: 3 * 6 * 4 * 20, usage: BufferUsage.Vertex });
    const section = (base: number): SectionDraw => ({
      originBlocks: [0, 0, 0], vertex: arena, vertexOffset: base * 20, baseVertex: base, quadCount: 6, pass: TerrainPass.Solid,
    });
    renderer.render([section(0), section(24), section(48)], fakeCamera([8, 8, 8]), 64, 64);
    expect(device.log.drawCmds).toHaveLength(3); // three sections drawn
    expect(device.log.vertexBinds).toBe(1); // …but the shared arena buffer was bound ONCE (region bind)
    expect(device.log.drawCmds.map((c) => c.baseVertex)).toEqual([0, 24, 48]); // each section located by its baseVertex
  });
});
