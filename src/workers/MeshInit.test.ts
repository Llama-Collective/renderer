// P5a: worker meshing via TRANSFERRED baked models must be byte-identical to inline meshing.
// RENDERER_OPTIMIZATION_PLAN Phase 5.

import { describe, it, expect } from "vitest";
import { meshSection, type BakedModelProvider } from "../mesh/SectionMesher";
import { ModelBaker, type BakedBlockModel, type BlockEntry } from "../mesh/model/BakedBlockModel";
import { TintProvider } from "../mesh/model/TintProvider";
import { fixtureBlockstates, fixtureProvider } from "../mesh/model/__fixtures__/load";
import { SnapshotSource, type BlockSource } from "../world/SnapshotSource";
import { sectionKey } from "../world/SectionKey";
import { DirtyReason, TERRAIN_PASSES } from "../types";
import { FluidType, type FluidAppearance, type FluidContext } from "../mesh/FluidMesher";
import type { SectionBuildOutput } from "./BuildOutput";
import { contextFromInit, extractMeshInit, extractPaletteModels, providerFromModels, unknownAtlasedPaletteIds } from "./MeshInit";

const STONE = 1;
const GLASS = 2;
const PALETTE: Record<number, BlockEntry> = {
  [STONE]: { name: "stone", props: {} },
  [GLASS]: { name: "glass", props: {}, cullGroup: "glass" },
};
const opacityOf = (sprite: string) =>
  sprite.includes("glass") ? { hasTransparent: false, hasTranslucent: true } : { hasTransparent: false, hasTranslucent: false };

function makeProvider(): BakedModelProvider {
  const baker = new ModelBaker({
    blockstate: (name) => fixtureBlockstates[name],
    models: fixtureProvider,
    uvFor: () => ({ u: 0, v: 0, width: 1, height: 1 }),
    opacityOf,
    tint: new TintProvider(),
  });
  const cache = new Map<number, BakedBlockModel | null>();
  return {
    get(id) {
      if (id === 0) return null;
      if (!cache.has(id)) cache.set(id, PALETTE[id] ? baker.bake(PALETTE[id]) : null);
      return cache.get(id) ?? null;
    },
  };
}

function meshWith(provider: BakedModelProvider, cells: Map<string, number>): SectionBuildOutput {
  const src: BlockSource = { getBlock: (x, y, z) => cells.get(`${x},${y},${z}`) ?? 0 };
  const snap = new SnapshotSource(src).buildSnapshot(sectionKey(0, 0, 0), 1, DirtyReason.InitialLoad);
  return meshSection(snap, provider);
}

/** Assert two build outputs are byte-for-byte identical (vertices, counts, sort, visibility). */
function expectIdentical(a: SectionBuildOutput, b: SectionBuildOutput): void {
  for (const pass of TERRAIN_PASSES) {
    const pa = a.parts[pass];
    const pb = b.parts[pass];
    expect(!!pa).toBe(!!pb);
    if (pa && pb) {
      expect(pa.quadCount).toBe(pb.quadCount);
      expect(new Uint8Array(pa.vertexData)).toEqual(new Uint8Array(pb.vertexData));
    }
  }
  expect(!!a.translucent).toBe(!!b.translucent);
  if (a.translucent && b.translucent) {
    expect(a.translucent.quadCount).toBe(b.translucent.quadCount);
    expect(a.translucent.sortType).toBe(b.translucent.sortType);
    expect(a.translucent.quadHash).toBe(b.translucent.quadHash);
    expect(new Uint8Array(a.translucent.vertexData)).toEqual(new Uint8Array(b.translucent.vertexData));
  }
  expect(a.info.visibilityData?.[0]).toBe(b.info.visibilityData?.[0]);
}

describe("MeshInit — worker provider via transferred baked models", () => {
  const world = new Map<string, number>([
    ["0,0,0", STONE], ["1,0,0", STONE], ["2,0,0", GLASS],
    ["0,1,0", GLASS], ["5,5,5", STONE], ["5,6,5", GLASS],
  ]);

  it("a provider rebuilt from extracted models meshes identically (same objects)", () => {
    const direct = makeProvider();
    const models = extractPaletteModels(direct, [0, STONE, GLASS]);
    const rebuilt = providerFromModels(models);
    expectIdentical(meshWith(rebuilt, world), meshWith(direct, world));
  });

  it("a provider rebuilt from STRUCTURED-CLONED models (simulating postMessage) meshes identically", () => {
    const direct = makeProvider();
    const models = extractPaletteModels(direct, [0, STONE, GLASS]);
    // structuredClone is exactly what `postMessage` does to the payload — the worker gets a deep copy.
    const transferred = structuredClone(models);
    const worker = providerFromModels(transferred);
    expectIdentical(meshWith(worker, world), meshWith(direct, world));
  });

  it("unknown ids (not in the transferred palette) resolve to null (air)", () => {
    const worker = providerFromModels(extractPaletteModels(makeProvider(), [STONE]));
    expect(worker.get(GLASS)).toBeNull(); // GLASS wasn't extracted → null in the worker
    expect(worker.get(STONE)).not.toBeNull();
    expect(worker.get(999)).toBeNull();
  });
});

const WATER = 20;
const LAVA = 21;
const app = (translucent: boolean, shade: boolean): FluidAppearance => ({ sprite: { u: 0, v: 0, width: 1, height: 1 }, colorRGBA: 0xffffffff, translucent, shade });
const FLUIDS: FluidContext = {
  fluidOf: (id) => (id === WATER ? { type: FluidType.Water, level: 0 } : id === LAVA ? { type: FluidType.Lava, level: 0 } : null),
  appearance: (type) => (type === FluidType.Water ? app(true, true) : app(false, false)),
};

function meshFluidWith(provider: BakedModelProvider, fluids: FluidContext, cells: Map<string, number>): SectionBuildOutput {
  const src: BlockSource = { getBlock: (x, y, z) => cells.get(`${x},${y},${z}`) ?? 0 };
  const snap = new SnapshotSource(src).buildSnapshot(sectionKey(0, 0, 0), 1, DirtyReason.InitialLoad);
  return meshSection(snap, provider, fluids);
}

describe("MeshInit — full init payload (models + fluids) round-trips identically", () => {
  const world = new Map<string, number>([
    ["0,0,0", WATER], ["1,0,0", WATER], ["3,0,0", LAVA], ["2,1,0", STONE], ["4,4,4", GLASS],
  ]);

  it("a worker context rebuilt from a structured-cloned init meshes fluids+terrain identically", () => {
    const direct = makeProvider();
    const init = extractMeshInit(direct, [0, STONE, GLASS, WATER, LAVA], FLUIDS);
    const transferred = structuredClone(init); // what postMessage does to the init payload
    const ctx = contextFromInit(transferred);
    expectIdentical(meshFluidWith(ctx.provider, ctx.fluids!, world), meshFluidWith(direct, FLUIDS, world));
  });
});

describe("MeshInit — mesh options (FS-5 / SL-4)", () => {
  it("carries meshOptions through extract → structuredClone (so the worker partitions like inline)", () => {
    const init = extractMeshInit(makeProvider(), [0, STONE], undefined, { partitionFacing: true });
    expect(init.meshOptions).toEqual({ partitionFacing: true });
    expect(structuredClone(init).meshOptions).toEqual({ partitionFacing: true }); // survives postMessage
  });

  it("absent meshOptions ⇒ undefined (byte-identical default)", () => {
    expect(extractMeshInit(makeProvider(), [0, STONE], undefined).meshOptions).toBeUndefined();
  });
});

describe("unknownAtlasedPaletteIds — id-level worker-model staleness (audit Bug 1)", () => {
  const palette = (entries: ReadonlyArray<readonly [number, string]>): Record<number, { name: string }> =>
    Object.fromEntries(entries.map(([id, name]) => [id, { name }]));

  it("returns a NEW property-variant id whose name is already atlased (the AIR-bug case)", () => {
    // ids 2 and 3 are both redstone_lamp (lit=false / lit=true) — same NAME, different props ⇒ different id.
    const p = palette([[1, "minecraft:stone"], [2, "minecraft:redstone_lamp"], [3, "minecraft:redstone_lamp"]]);
    const known = new Set([0, 1, 2]); // worker knows AIR + stone + lamp(unlit); lamp(lit)=id 3 is the new one
    const atlased = new Set(["minecraft:stone", "minecraft:redstone_lamp"]);
    expect(unknownAtlasedPaletteIds(p, known, atlased)).toEqual([3]);
  });

  it("SKIPS a new id whose name is NOT atlased — a full rebuild re-inits the pool for it", () => {
    const p = palette([[1, "minecraft:stone"], [2, "minecraft:observer"]]);
    const known = new Set([0, 1]);
    const atlased = new Set(["minecraft:stone"]); // observer not atlased yet → name-keyed rebuild handles it
    expect(unknownAtlasedPaletteIds(p, known, atlased)).toEqual([]);
  });

  it("returns nothing when the worker already knows every id", () => {
    const p = palette([[1, "a"], [2, "b"]]);
    expect(unknownAtlasedPaletteIds(p, new Set([0, 1, 2]), new Set(["a", "b"]))).toEqual([]);
  });
});
