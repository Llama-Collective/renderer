// Model-path meshing: per-quad cull + layer buckets, using the real baker over pack fixtures.
// RENDERER_PLAN §6, §24, §22.

import { describe, it, expect } from "vitest";
import { meshSection, MESH_CANCELLED, type BakedModelProvider } from "./SectionMesher";
import { ModelBaker, type BakedBlockModel, type BlockEntry } from "./model/BakedBlockModel";
import { TintProvider } from "./model/TintProvider";
import { fixtureBlockstates, fixtureProvider } from "./model/__fixtures__/load";
import { SnapshotSource, isAllAirCore, type BlockSource } from "../world/SnapshotSource";
import { sectionKey } from "../world/SectionKey";
import { DirtyReason, Direction, TerrainPass } from "../types";
import { isConnected, VIS_ALL, VIS_NONE } from "../world/SectionVisibility";
import { SortType } from "./SortTypes";
import { FluidType, type FluidAppearance, type FluidContext } from "./FluidMesher";

// Block palette for the test world. 0 = air.
const STONE = 1;
const GLASS = 2;
const PALETTE: Record<number, BlockEntry> = {
  [STONE]: { name: "stone", props: {} },
  [GLASS]: { name: "glass", props: {}, cullGroup: "glass" },
};

// Sprite opacity: everything opaque except the glass texture (so glass → non-occluding, translucent).
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

function meshWorld(solid: Map<string, number>) {
  const src: BlockSource = { getBlock: (x, y, z) => solid.get(`${x},${y},${z}`) ?? 0 };
  const snap = new SnapshotSource(src).buildSnapshot(sectionKey(0, 0, 0), 1, DirtyReason.InitialLoad);
  return meshSection(snap, makeProvider());
}

const solidQuads = (out: ReturnType<typeof meshWorld>) => out.parts[TerrainPass.Solid]?.quadCount ?? 0;
// Translucent is emitted separately into out.translucent (NOT parts) so it can carry its own
// sorted index order + per-quad metadata (§12).
const translucentQuads = (out: ReturnType<typeof meshWorld>) => out.translucent?.quadCount ?? 0;

describe("SectionMesher (model path)", () => {
  it("a lone stone cube emits 6 solid quads", () => {
    expect(solidQuads(meshWorld(new Map([["0,0,0", STONE]])))).toBe(6);
  });

  it("culls the shared face between two adjacent stone cubes", () => {
    expect(solidQuads(meshWorld(new Map([["0,0,0", STONE], ["1,0,0", STONE]])))).toBe(10);
  });

  it("a solid 16³ section against air emits only its outer shell", () => {
    const solid = new Map<string, number>();
    for (let y = 0; y < 16; y++) for (let z = 0; z < 16; z++) for (let x = 0; x < 16; x++) solid.set(`${x},${y},${z}`, STONE);
    expect(solidQuads(meshWorld(solid))).toBe(6 * 16 * 16);
  });

  it("glass goes to the translucent part and self-culls (skip group)", () => {
    const out = meshWorld(new Map([["0,0,0", GLASS], ["1,0,0", GLASS]]));
    expect(solidQuads(out)).toBe(0);
    // Two adjacent glass: shared faces culled by the skip group → 12 - 2 = 10 translucent quads.
    expect(translucentQuads(out)).toBe(10);
    // A convex glass box (2×1×1) is order-independent (case C) → AnyOrder: shared EBO, no
    // per-section index, no carried quads, back-face culled.
    expect(out.translucent?.sortType).toBe(SortType.AnyOrder);
    expect(out.translucent?.indexData).toBeUndefined();
    expect(out.translucent?.quads).toBeUndefined();
  });

  it("a single glass cube → AnyOrder (convex; its ≤3 visible faces never overlap)", () => {
    const out = meshWorld(new Map([["0,0,0", GLASS]]));
    expect(solidQuads(out)).toBe(0);
    expect(translucentQuads(out)).toBe(6);
    expect(out.translucent?.sortType).toBe(SortType.AnyOrder);
  });

  it("empty section produces no geometry", () => {
    const out = meshWorld(new Map());
    expect(out.approxBytes).toBe(0);
    expect(out.translucent).toBeUndefined();
  });
});

describe("isAllAirCore (#15 all-air skip)", () => {
  const snap = (cells: Map<string, number>) =>
    new SnapshotSource({ getBlock: (x, y, z) => cells.get(`${x},${y},${z}`) ?? 0 }).buildSnapshot(sectionKey(0, 0, 0), 1, DirtyReason.InitialLoad);
  it("true for an empty section", () => expect(isAllAirCore(snap(new Map()))).toBe(true));
  it("false when a core cell is non-air", () => expect(isAllAirCore(snap(new Map([["8,8,8", STONE]])))).toBe(false));
  it("true when only an APRON cell (neighbor) is non-air — it belongs to another section", () =>
    expect(isAllAirCore(snap(new Map([["-1,0,0", STONE]])))).toBe(true));
});

// Fluid routing: water → translucent part, lava → solid part (TRAP 13.A).
const WATER = 20, LAVA = 21;
const dummyApp = (translucent: boolean, shade: boolean): FluidAppearance => ({ sprite: { u: 0, v: 0, width: 1, height: 1 }, colorRGBA: 0xffffffff, translucent, shade });
const FLUIDS: FluidContext = {
  fluidOf: (id) => (id === WATER ? { type: FluidType.Water, level: 0 } : id === LAVA ? { type: FluidType.Lava, level: 0 } : null),
  appearance: (type) => (type === FluidType.Water ? dummyApp(true, true) : dummyApp(false, false)),
};

function meshFluidWorld(cells: Map<string, number>) {
  const src: BlockSource = { getBlock: (x, y, z) => cells.get(`${x},${y},${z}`) ?? 0 };
  const snap = new SnapshotSource(src).buildSnapshot(sectionKey(0, 0, 0), 1, DirtyReason.InitialLoad);
  return meshSection(snap, makeProvider(), FLUIDS);
}

describe("SectionMesher fluids", () => {
  it("a water source meshes into the TRANSLUCENT part, not solid", () => {
    const out = meshFluidWorld(new Map([["0,0,0", WATER]]));
    expect(out.translucent?.quadCount ?? 0).toBeGreaterThan(0);
    expect(out.parts[TerrainPass.Solid]?.quadCount ?? 0).toBe(0);
  });

  it("a lava source meshes into the SOLID part, not translucent", () => {
    const out = meshFluidWorld(new Map([["0,0,0", LAVA]]));
    expect(out.parts[TerrainPass.Solid]?.quadCount ?? 0).toBeGreaterThan(0);
    expect(out.translucent?.quadCount ?? 0).toBe(0);
  });

  it("without a fluid context, fluid blocks emit nothing (model path only)", () => {
    expect(meshWorld(new Map([["0,0,0", WATER]])).approxBytes).toBe(0);
  });
});

describe("SectionMesher visibility graph (OCC-1)", () => {
  const visOf = (out: ReturnType<typeof meshWorld>) => out.info.visibilityData?.[0];

  it("an all-air section is fully connected (VIS_ALL)", () => {
    expect(visOf(meshWorld(new Map()))).toBe(VIS_ALL);
  });

  it("a fully solid 16³ stone section blocks all sight (VIS_NONE)", () => {
    const solid = new Map<string, number>();
    for (let y = 0; y < 16; y++) for (let z = 0; z < 16; z++) for (let x = 0; x < 16; x++) solid.set(`${x},${y},${z}`, STONE);
    expect(visOf(meshWorld(solid))).toBe(VIS_NONE);
  });

  it("a stone wall at z=8 disconnects North↔South but leaves West↔East open", () => {
    const w = new Map<string, number>();
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) w.set(`${x},${y},8`, STONE);
    const vis = visOf(meshWorld(w))!;
    expect(isConnected(vis, Direction.North, Direction.South)).toBe(false);
    expect(isConnected(vis, Direction.West, Direction.East)).toBe(true);
    expect(isConnected(vis, Direction.Down, Direction.Up)).toBe(true);
  });

  it("glass does NOT count as a sight blocker (sight passes through — no over-cull)", () => {
    // A full glass plane at z=8: glass is not a full opaque cube, so the section stays fully connected.
    const w = new Map<string, number>();
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) w.set(`${x},${y},8`, GLASS);
    expect(isConnected(visOf(meshWorld(w))!, Direction.North, Direction.South)).toBe(true);
  });
});

// WRK-3: the optional per-Y-layer cooperative-cancel poll. Proves (1) a poll that fires returns the distinct
// MESH_CANCELLED sentinel (no SectionBuildOutput escapes), and (2) `isCancelled=()=>false` is byte-identical
// to the no-arg call (the poll is transparent when it never fires — the default-off / surviving-build path).
function meshWorldCancellable(solid: Map<string, number>, isCancelled: () => boolean) {
  const src: BlockSource = { getBlock: (x, y, z) => solid.get(`${x},${y},${z}`) ?? 0 };
  const snap = new SnapshotSource(src).buildSnapshot(sectionKey(0, 0, 0), 1, DirtyReason.InitialLoad);
  return meshSection(snap, makeProvider(), undefined, undefined, isCancelled);
}

/** Serialize EVERY part's vertex bytes into one comparable string (geometry byte-identity over all passes). */
function allBytes(out: ReturnType<typeof meshWorld>): string {
  const chunks: string[] = [];
  for (const pass of [TerrainPass.Solid, TerrainPass.Cutout]) {
    const p = out.parts[pass];
    if (p) chunks.push(`${pass}:${[...new Uint8Array(p.vertexData)].join(",")}`);
  }
  if (out.translucent) chunks.push(`t:${out.translucent.quadHash}`);
  return chunks.join("|");
}

describe("SectionMesher (WRK-3 cooperative-cancel poll)", () => {
  // A multi-Y-layer world so the poll has several layers to bail across.
  const world = new Map<string, number>();
  for (let y = 0; y < 16; y++) world.set(`0,${y},0`, STONE); // one stone column spanning all 16 Y-layers

  it("returns the MESH_CANCELLED sentinel when the poll fires (no SectionBuildOutput, no quads)", () => {
    const out = meshWorldCancellable(world, () => true); // cancel on the very first Y-layer poll
    expect(out).toBe(MESH_CANCELLED); // distinct sentinel — the worker posts `cancelled`, never buildDone
  });

  it("cancels after the first Y-layer too (the poll runs per-layer, not once)", () => {
    let polls = 0;
    const out = meshWorldCancellable(world, () => ++polls > 1); // false on ly=0, true on ly=1
    expect(out).toBe(MESH_CANCELLED);
    expect(polls).toBe(2); // bailed at the SECOND layer's poll — proves per-Y-layer granularity
  });

  it("isCancelled=()=>false is byte-identical to the no-arg call (transparent when it never fires)", () => {
    const baseline = meshWorld(world); // no-arg (the inline / default-off path)
    const withFalsePoll = meshWorldCancellable(world, () => false);
    expect(withFalsePoll).not.toBe(MESH_CANCELLED);
    expect(allBytes(withFalsePoll as ReturnType<typeof meshWorld>)).toBe(allBytes(baseline));
  });
});

// OCC-2: perPerspectiveVisibility widens `visibilityData` to the 4-element DIRECTION_SETS. It lives OUTSIDE
// the FNV vertex hash + the per-pass vertex bytes, so geometry must be byte-identical flag on/off, AND
// `visibilityData[0]` must stay the symmetric union (the off-path word). Proves the flag is geometry-neutral.
function meshWorldOpts(solid: Map<string, number>, opts: { perPerspectiveVisibility?: boolean }) {
  const src: BlockSource = { getBlock: (x, y, z) => solid.get(`${x},${y},${z}`) ?? 0 };
  const snap = new SnapshotSource(src).buildSnapshot(sectionKey(0, 0, 0), 1, DirtyReason.InitialLoad);
  return meshSection(snap, makeProvider(), undefined, opts);
}

describe("SectionMesher (OCC-2 per-perspective visibility)", () => {
  // A mixed world (solid + cutout + translucent) so allBytes covers every pass + the translucent quadHash.
  const world = new Map<string, number>();
  world.set("0,0,0", STONE);
  world.set("2,0,0", STONE);
  for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) world.set(`${x},${y},8`, STONE); // a wall (non-trivial vis)
  world.set("5,5,5", GLASS); // a translucent quad so the translucent quadHash is part of the gate

  it("quadHash + all vertex bytes are byte-identical with the flag ON vs OFF (geometry-neutral)", () => {
    const off = meshWorld(world); // default-off (1-element visibilityData)
    const on = meshWorldOpts(world, { perPerspectiveVisibility: true });
    expect(allBytes(on as ReturnType<typeof meshWorld>)).toBe(allBytes(off)); // geometry golden gate
    // The translucent quadHash specifically is identical (visibilityData is outside the FNV vertex hash).
    expect(on.translucent?.quadHash).toBe(off.translucent?.quadHash);
  });

  it("emits a 4-element visibilityData whose [0] is the symmetric union (off-readers unaffected)", () => {
    const off = meshWorld(world);
    const on = meshWorldOpts(world, { perPerspectiveVisibility: true });
    expect(off.info.visibilityData?.length).toBe(1); // off-path stays a 1-element array
    expect(on.info.visibilityData?.length).toBe(4); // on-path widens to the 4 DIRECTION_SETS
    expect(on.info.visibilityData![0]).toBe(off.info.visibilityData![0]); // [0] === the symmetric word
  });
});
