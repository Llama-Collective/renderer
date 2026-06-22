// FS-3 — buildSectionDraws threads the opaque/cutout per-facing runs (+ sliceMask) onto the SectionDraw,
// and omits them for unpartitioned sections. No pixel change yet (FS-4 consumes them).

import { describe, it, expect } from "vitest";
import { buildSectionDraws, commitWorld } from "./scene";
import { GpuSectionUploader, type GpuCommittedSection } from "../render/GpuSectionUploader";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { makeBuildOutput } from "../workers/BuildOutput";
import { TerrainPass } from "../types";
import { sectionKey } from "../world/SectionKey";
import type { RenderSection } from "../world/RenderSection";
import type { WebGPUDevice } from "../core/webgpu/WebGPUDevice";
import type { BlockSource } from "../world/SnapshotSource";
import { ModelBaker, type BakedBlockModel, type BlockEntry } from "../mesh/model/BakedBlockModel";
import { TintProvider } from "../mesh/model/TintProvider";
import { fixtureBlockstates, fixtureProvider } from "../mesh/model/__fixtures__/load";
import type { BakedModelProvider } from "../mesh/SectionMesher";

const KEY = sectionKey(0, 0, 0);
const STRIDE = 20;

const STONE = 1;
function makeProvider(): BakedModelProvider {
  const baker = new ModelBaker({
    blockstate: (name) => fixtureBlockstates[name],
    models: fixtureProvider,
    uvFor: () => ({ u: 0, v: 0, width: 1, height: 1 }),
    opacityOf: () => ({ hasTransparent: false, hasTranslucent: false }),
    tint: new TintProvider(),
  });
  const palette: Record<number, BlockEntry> = { [STONE]: { name: "stone", props: {} } };
  const cache = new Map<number, BakedBlockModel | null>();
  return { get(id) { if (id === 0) return null; if (!cache.has(id)) cache.set(id, palette[id] ? baker.bake(palette[id]) : null); return cache.get(id) ?? null; } };
}

function sectionWith(committed: GpuCommittedSection): RenderSection {
  return { key: KEY, presented: committed } as unknown as RenderSection;
}

describe("buildSectionDraws facing partition (FS-3)", () => {
  it("carries facingVertexCounts + the sliceMask for an opaque pass", () => {
    const up = new GpuSectionUploader(new FakeGraphicsDevice());
    const quads = 4;
    const counts = Uint32Array.of(4, 4, 0, 4, 4, 0, 0); // PosX,PosY,NegX,NegY present; rest empty
    const out = makeBuildOutput({ sectionKey: KEY, generation: 1, info: { flags: 0 }, parts: { [TerrainPass.Solid]: { vertexData: new ArrayBuffer(quads * 4 * STRIDE), quadCount: quads, facingVertexCounts: counts } } });
    const committed = up.upload(out) as GpuCommittedSection;

    const solid = buildSectionDraws(sectionWith(committed)).find((d) => d.pass === TerrainPass.Solid)!;
    expect(Array.from(solid.facingVertexCounts!)).toEqual([4, 4, 0, 4, 4, 0, 0]);
    expect(solid.sliceMask).toBe((1 << 0) | (1 << 1) | (1 << 3) | (1 << 4)); // OR of the non-empty facings
  });

  it("omits facing data for an unpartitioned build (legacy single draw)", () => {
    const up = new GpuSectionUploader(new FakeGraphicsDevice());
    const out = makeBuildOutput({ sectionKey: KEY, generation: 1, info: { flags: 0 }, parts: { [TerrainPass.Solid]: { vertexData: new ArrayBuffer(80), quadCount: 1 } } });
    const committed = up.upload(out) as GpuCommittedSection;

    const solid = buildSectionDraws(sectionWith(committed)).find((d) => d.pass === TerrainPass.Solid)!;
    expect(solid.facingVertexCounts).toBeUndefined();
    expect(solid.sliceMask).toBeUndefined();
  });
});

describe("commitWorld facing partition (FS-5 initial-commit wiring)", () => {
  // A FakeGraphicsDevice stands in for the WebGPUDevice (commitWorld only uses GraphicsDevice methods).
  const device = new FakeGraphicsDevice() as unknown as WebGPUDevice;
  const world: BlockSource = { getBlock: (x, y, z) => (x === 0 && y === 0 && z === 0 ? STONE : 0) };

  it("partitionFacing flows into the initial commit → the section's draw carries facing runs", () => {
    const store = commitWorld(device, world, makeProvider(), [1, 1, 1], undefined, undefined, { partitionFacing: true });
    const solid = buildSectionDraws(store.get(KEY)!).find((d) => d.pass === TerrainPass.Solid)!;
    expect(solid.facingVertexCounts).toBeDefined();
    expect(Array.from(solid.facingVertexCounts!)).toEqual([4, 4, 4, 4, 4, 4, 0]); // a stone cube
    expect(solid.sliceMask).toBe(0b0111111); // six axis facings
  });

  it("default (no options) commits byte-identical single draws (no facing runs)", () => {
    const store = commitWorld(device, world, makeProvider(), [1, 1, 1]);
    const solid = buildSectionDraws(store.get(KEY)!).find((d) => d.pass === TerrainPass.Solid)!;
    expect(solid.facingVertexCounts).toBeUndefined();
    expect(solid.sliceMask).toBeUndefined();
  });
});

describe("commitWorld smooth lighting (SL-4 wiring)", () => {
  // A concave corner (origin block + diagonal neighbour) so AO has something to darken.
  const concave: BlockSource = { getBlock: (x, y, z) => ((x === 0 && y === 0 && z === 0) || (x === 1 && y === 1 && z === 1) ? STONE : 0) };

  function solidBytes(dev: FakeGraphicsDevice, store: ReturnType<typeof commitWorld>): Uint8Array {
    const c = store.get(KEY)!.presented as GpuCommittedSection;
    const arena = c.vertexArena[TerrainPass.Solid]!;
    const r = arena.rangeOf(c.vertexAlloc[TerrainPass.Solid]!);
    return dev.read(arena.gpuBuffer).subarray(r.offsetBytes, r.offsetBytes + r.sizeBytes);
  }

  it("smoothLighting flows commitWorld → apron-2 snapshots → meshSection: committed bytes differ from flat", () => {
    const devFlat = new FakeGraphicsDevice();
    const devSmooth = new FakeGraphicsDevice();
    const flat = commitWorld(devFlat as unknown as WebGPUDevice, concave, makeProvider(), [1, 1, 1]);
    const smooth = commitWorld(devSmooth as unknown as WebGPUDevice, concave, makeProvider(), [1, 1, 1], undefined, undefined, { smoothLighting: true });
    const flatBytes = solidBytes(devFlat, flat);
    const smoothBytes = solidBytes(devSmooth, smooth);
    expect(smoothBytes.length).toBe(flatBytes.length); // same geometry shape
    expect(smoothBytes).not.toEqual(flatBytes); // AO/light bytes changed (smooth lighting applied)
  });
});
