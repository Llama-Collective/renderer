// Off-thread meshing init — the baked-model provider, TRANSFERRED (not re-baked) to the worker.
// RENDERER_OPTIMIZATION_PLAN.md Phase 5; "Directional Visibility Data" / ClonedChunkSection.
//
// The meshing risk in moving off-thread is GEOMETRY DIVERGENCE: if the worker re-bakes models from a
// serialized resource pack, any baking difference yields different geometry than the main thread →
// holes/corruption. We sidestep that entirely. `BakedBlockModel` is pure data ("GPU-free … can run in a
// worker" — BakedBlockModel.ts), so we snapshot the main thread's ALREADY-baked palette models and
// structured-clone them to the worker. The worker's provider is then a plain Map lookup over the exact
// same baked data — meshing is provider-source-agnostic, so worker output is byte-identical to inline.
//
// New block ids minted AFTER init aren't in the map (worker returns null → the block would mesh as AIR).
// The live wiring closes this: a new block NAME triggers a full scene rebuild (re-init with the new
// palette), and a new property-variant id of an ALREADY-atlased name is pushed via the `addModels` message
// (`SchematicViewer.syncWorkerModels`, selecting ids with `unknownAtlasedPaletteIds`). Without that push a
// runtime-minted id (a lamp lighting, a door opening) renders as AIR — audit Bug 1. Fluids are a separate
// context (functions, not data) — fluid sections need their FluidContext serialized (handled in `extractFluids`).

import type { BakedBlockModel } from "../mesh/model/BakedBlockModel";
import type { BakedModelProvider, MeshOptions } from "../mesh/SectionMesher";
import { FluidType, type FluidAppearance, type FluidContext, type FluidState } from "../mesh/FluidMesher";

/** Serialized provider payload: [block id, baked model | null] pairs. Structured-cloneable / transferable. */
export type MeshModels = ReadonlyArray<readonly [number, BakedBlockModel | null]>;

/** Serialized FluidContext — pure data (FluidState/FluidAppearance are numbers/booleans/UV rects). */
export interface MeshFluids {
  /** Fluid block id → state (water/lava + level), for the palette ids that ARE fluids. */
  states: ReadonlyArray<readonly [number, FluidState]>;
  water: FluidAppearance;
  lava: FluidAppearance;
}

/** Everything a worker needs to mesh a section identically to the main thread (no re-baking). */
export interface MeshInitPayload {
  models: MeshModels;
  fluids?: MeshFluids;
  /** FS-5 / SL-4: mesh-time toggles (partitionFacing, smoothLighting) so off-thread output matches the
   *  inline path bit-for-bit. Set at init; flipping one forces a full scene rebuild (re-init), so the
   *  whole world re-meshes under the new option together. Absent ⇒ the byte-identical defaults. */
  meshOptions?: MeshOptions;
}

/** Snapshot each palette id's baked model from a live provider (builds the worker init payload). */
export function extractPaletteModels(provider: BakedModelProvider, ids: Iterable<number>): MeshModels {
  const out: Array<readonly [number, BakedBlockModel | null]> = [];
  for (const id of ids) out.push([id, provider.get(id)]);
  return out;
}

/**
 * Palette ids the worker pool does NOT yet know AND whose block name the current atlas already covers —
 * the set that must be shipped via `addModels` so off-thread meshing renders them instead of AIR (audit
 * Bug 1). Staleness is tracked at the ID level, not the NAME level: a property-only edit (lamp `lit=true`,
 * door `open=true`, redstone `power=N`) mints a NEW id whose name is already atlased, so the name-keyed
 * full-rebuild guard never fires for it. Ids with an UNATLASED name are deliberately EXCLUDED — those do
 * trigger a full scene rebuild, which re-inits the pool with the new atlas + palette.
 */
export function unknownAtlasedPaletteIds(
  palette: Readonly<Record<number, { readonly name: string }>>,
  knownIds: ReadonlySet<number>,
  atlasedNames: ReadonlySet<string>,
): number[] {
  const out: number[] = [];
  for (const idStr in palette) {
    const id = Number(idStr);
    if (knownIds.has(id)) continue;
    if (!atlasedNames.has(palette[id].name)) continue; // unatlased name ⇒ a full rebuild will re-init the pool
    out.push(id);
  }
  return out;
}

/** Rebuild a provider from transferred baked models — a pure Map lookup (no re-baking ⇒ no divergence). */
export function providerFromModels(models: MeshModels): BakedModelProvider {
  const map = new Map<number, BakedBlockModel | null>(models);
  return { get: (id) => map.get(id) ?? null };
}

/** Snapshot a FluidContext for the palette ids (fluid classification + the two appearances). */
export function extractFluids(fluids: FluidContext, ids: Iterable<number>): MeshFluids {
  const states: Array<readonly [number, FluidState]> = [];
  for (const id of ids) {
    const s = fluids.fluidOf(id);
    if (s) states.push([id, s]);
  }
  return { states, water: fluids.appearance(FluidType.Water), lava: fluids.appearance(FluidType.Lava) };
}

/** Rebuild a FluidContext from transferred data — a Map lookup + the two fixed appearances. */
export function fluidsFromSerialized(f: MeshFluids): FluidContext {
  const map = new Map<number, FluidState>(f.states);
  return {
    fluidOf: (id) => map.get(id) ?? null,
    appearance: (t) => (t === FluidType.Water ? f.water : f.lava),
  };
}

/** Build the full worker init payload from the live scene (models + optional fluids + mesh toggles). */
export function extractMeshInit(provider: BakedModelProvider, ids: Iterable<number>, fluids?: FluidContext, meshOptions?: MeshOptions): MeshInitPayload {
  const idArr = [...ids];
  return { models: extractPaletteModels(provider, idArr), fluids: fluids ? extractFluids(fluids, idArr) : undefined, meshOptions };
}

/** Reconstruct the worker-side mesh context (provider + fluids) from a transferred init payload. */
export function contextFromInit(p: MeshInitPayload): { provider: BakedModelProvider; fluids?: FluidContext } {
  return { provider: providerFromModels(p.models), fluids: p.fluids ? fluidsFromSerialized(p.fluids) : undefined };
}
