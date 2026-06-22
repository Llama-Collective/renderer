// Shared renderer pipeline: pack → baker → mesh → GPU draws. Owns the translucent draw-assembly
// (index buffer, sort type, sort metadata) in ONE place so the app façade AND every harness page
// exercise the real §12 path. Moved here from harness/scene.ts (which now re-exports this) so the
// app-facing `src/app` layer doesn't depend on `harness/`. RENDERER_INTEGRATION_PLAN Step 1/2.

import type { CanvasDevice } from "../core/createDevice";
import type { GpuTextureHandle } from "../core/GraphicsDevice";
import { AtlasManager } from "../core/AtlasManager";
import { instrument } from "../core/Instrument";
import { ResourcePack } from "../resources/ResourcePack";
import { ModelBaker, type BakedBlockModel, type BlockEntry } from "../mesh/model/BakedBlockModel";
import { TintProvider } from "../mesh/model/TintProvider";
import { resolveBlockState, type BlockProps } from "../mesh/model/BlockStateResolver";
import { bakeParts } from "../mesh/model/FaceBakery";
import { stripNamespace, type RawBlockState, type RawVariant, type VariantPart } from "../mesh/model/ModelTypes";
import type { RawModelProvider } from "../mesh/model/ModelResolver";
import { meshSection, type BakedModelProvider, type MeshOptions } from "../mesh/SectionMesher";
import { type BlockSource } from "../world/SnapshotSource";
import { SnapshotSource, isAllAirCore } from "../world/SnapshotSource";
import { SectionStore } from "../world/SectionStore";
import type { RegionStore } from "../world/RegionStore";
import type { RenderSection } from "../world/RenderSection";
import { computeVisibleSections, packSectionCoord, type OcclusionQuery, type SectionInfo } from "../world/OcclusionCuller";
import { GpuSectionUploader, type GpuCommittedSection } from "../render/GpuSectionUploader";
import { TerrainRenderer, type SectionDraw } from "../render/TerrainRenderer";
import { VERTEX_STRIDE_BYTES } from "../mesh/VertexFormat";
import { EntityAtlasManager } from "../core/EntityAtlasManager";
import { EntityWorld } from "../render/entities/EntityWorld";
import { allBETextures, hasBlockEntityBoxModel } from "../render/entities/blockentities";
import { FONT_TEX, scanFontWidths, setFontWidths } from "../render/entities/blockentities/font";
import { entityModelTextures } from "../render/entities/boxModels";
import { CART_CONTENTS } from "../render/entities/EntityModelFactory";
import { parseBlockState } from "../render/entities/blockDisplay";
import { ItemGeometry, resolveItem, type ResolvedItem } from "../render/items/ItemGeometry";
import type { ItemSprite } from "../render/items/itemExtrude";
import { sectionKey, type SectionKey } from "../world/SectionKey";
import { clamp, DirtyReason, TerrainPass, TERRAIN_PASSES, type SpriteUv, type Vec3, type Vec3i } from "../types";
import { FluidType, type FluidAppearance, type FluidContext, type FluidState } from "../mesh/FluidMesher";

export const AIR = 0;

// Fluid sprites the model baker never discovers (water/lava have no block model). Pass as
// `extraSprites` so the atlas stitches them; `makeFluidContext` then resolves their UVs.
export const WATER_STILL = "block/water_still";
export const LAVA_STILL = "block/lava_still";
export const FLUID_SPRITES = [WATER_STILL, LAVA_STILL];

/** Blocks that ride inside special minecarts — always pre-baked + atlased so a spawned cart's contents
 *  render even when no such block is placed in the world. Derived from the canonical CART_CONTENTS map. */
export const ENTITY_CONTAINED_BLOCKS: { name: string; props: BlockProps }[] = [
  ...Object.values(CART_CONTENTS)
    .filter((s): s is string => !!s)
    .map((s) => parseBlockState(s)),
  // Item-frame entities render a BLOCK model (frame border + back panel) from block-atlas textures
  // (birch_planks, item_frame). They're ENTITIES, not placed blocks, so their sprites are never in the
  // world palette — pre-atlas every (map) variant so a spawned frame renders + never trips a re-atlas.
  { name: "minecraft:item_frame", props: { map: "false" } },
  { name: "minecraft:item_frame", props: { map: "true" } },
  { name: "minecraft:glow_item_frame", props: { map: "false" } },
  { name: "minecraft:glow_item_frame", props: { map: "true" } },
];

/** Blocks the piston animation introduces at RUNTIME but that aren't in a resting scene's palette: the
 *  sliding `piston_head` (its models AREN'T pulled in by the base piston's model chain — that's
 *  `piston_extended`, not the head) and the invisible `moving_piston`. Always preload + atlas them, like
 *  the cart contents, so the FIRST piston fire neither bakes an empty head (the head stays invisible for
 *  the whole first slide) nor forces a full re-atlas/remesh (the freeze) when these names first appear.
 *  See the "first piston activation" investigation in PISTON_PLAN. */
export const PISTON_TRANSIENT_BLOCKS: { name: string; props: BlockProps }[] = [
  { name: "minecraft:piston_head", props: {} },
  { name: "minecraft:moving_piston", props: {} },
];

/** Every block NAME `buildScene` atlases/preloads BEYOND the world palette (cart contents + piston
 *  transients). Seed the façade's `atlasedNames` with these so placing/animating one never marks the
 *  palette stale → no spurious full rebuild; they're already covered. */
export const ALWAYS_ATLASED_BLOCK_NAMES: readonly string[] = [
  ...ENTITY_CONTAINED_BLOCKS.map((b) => b.name),
  ...PISTON_TRANSIENT_BLOCKS.map((b) => b.name),
];

export interface Scene {
  atlas: AtlasManager;
  provider: BakedModelProvider;
  renderer: TerrainRenderer;
  /** The shared block baker — reused by entities/BEs to bake block-display geometry (Phase 4.5). */
  baker: ModelBaker;
  pack: ResourcePack;
  /**
   * ATL-1: incrementally grow the atlas + palette for new block states placed at runtime, WITHOUT a full
   * `buildScene`/`commitWorld` rebuild. Appends the states' new sprites into the atlas headroom (existing
   * UVs invariant — TRAP 1.C), merges their opacity, and drops the new ids' cached bakes so they re-bake
   * against the grown atlas. Returns false (caller then does a full rebuild) if the block is unknown, a new
   * sprite is animated, or the headroom is exhausted. The CALLER re-meshes only the edit's dirtied sections
   * (already queued) — existing sections keep their geometry. Resolves once the grow is uploaded.
   */
  growPalette(states: ReadonlyArray<{ id: number; name: string; props: BlockProps }>): Promise<boolean>;
}

/**
 * Every sprite a blockstate could ever reference: the CURRENT props plus ALL variant/multipart cases.
 * Walking all cases (not just the loaded state) means a later state change — a note block's `note`, a
 * redstone wire's `power`, a block's `facing` — never needs a sprite the atlas lacks. A missing sprite
 * falls back to the whole-atlas UV rect and renders as corrupted garbage, so completeness here is what
 * keeps state changes from "breaking textures" (the note-block bug). One bad case must not blank the page.
 */
function collectStateSprites(bs: RawBlockState, props: BlockProps, provider: RawModelProvider, out: Set<string>): void {
  const add = (parts: VariantPart[]): void => {
    try {
      for (const q of bakeParts(parts, provider)) out.add(q.sprite);
    } catch {
      /* unbakeable case — skip it */
    }
  };
  add(resolveBlockState(bs, props)); // current state (also covers plain blocks with neither variants nor multipart)
  const cases: (RawVariant | RawVariant[])[] = [];
  if (bs.variants) for (const k of Object.keys(bs.variants)) cases.push(bs.variants[k]);
  if (bs.multipart) for (const c of bs.multipart) cases.push(c.apply);
  for (const v of cases) {
    const variant = Array.isArray(v) ? v[0] : v;
    add([{ model: stripNamespace(variant.model), x: variant.x ?? 0, y: variant.y ?? 0, z: variant.z ?? 0, uvlock: variant.uvlock === true }]);
  }
}

/**
 * Load the pack, build the atlas + baker for a palette, and create a TerrainRenderer.
 * Pass `pack` to reuse an already-loaded ResourcePack (the app loads it from an ArrayBuffer);
 * otherwise it fetches "pack.zip" (the harness default).
 */
export async function buildScene(
  device: CanvasDevice,
  palette: Record<number, BlockEntry>,
  onStatus?: (text: string) => void,
  extraSprites?: Iterable<string>,
  renderDistanceBlocks?: number,
  pack?: ResourcePack,
): Promise<Scene> {
  onStatus?.("loading pack…");
  const resolvedPack = pack ?? (await ResourcePack.load("pack.zip"));
  // Blocks contained inside minecarts (chest/hopper/furnace/tnt/command_block) are baked by the entity
  // renderer on demand, but they may never appear in the world palette — so always preload their models +
  // atlas their sprites here, else a spawned special cart bakes geometry against an atlas that lacks them
  // and the contained block renders blank/garbage (the "hopper cart has no block inside" bug). The same
  // applies to the piston animation's runtime-only blocks (piston_head / moving_piston) — preload + atlas
  // them up front so the first piston fire renders the head immediately and never forces a rebuild.
  const entityBlocks = [...ENTITY_CONTAINED_BLOCKS, ...PISTON_TRANSIENT_BLOCKS];
  await resolvedPack.preloadBlocks([...Object.values(palette).map((e) => e.name), ...entityBlocks.map((b) => b.name)]);

  // Discover every sprite the palette references (bake geometry once just to enumerate sprites),
  // plus any caller-supplied sprites the model path can't surface (fluids) and the cart-contained blocks.
  const spriteIds = new Set<string>(extraSprites ?? []);
  const mp = resolvedPack.modelProvider();
  for (const entry of Object.values(palette)) {
    const bs = resolvedPack.getBlockState(entry.name);
    if (!bs) continue;
    collectStateSprites(bs, entry.props, mp, spriteIds);
  }
  for (const b of entityBlocks) {
    const bs = resolvedPack.getBlockState(b.name);
    if (bs) collectStateSprites(bs, b.props, mp, spriteIds);
  }

  onStatus?.("decoding textures…");
  const { sprites, opacity } = await resolvedPack.loadSprites(spriteIds);
  const colormaps = await resolvedPack.loadColormaps();
  const atlas = new AtlasManager(device);
  atlas.build(sprites);
  const loaded = new Set(sprites.map((s) => s.name));

  const baker = new ModelBaker({
    blockstate: (n) => resolvedPack.getBlockState(n),
    models: resolvedPack.modelProvider(),
    uvFor: (s) => (loaded.has(s) ? atlas.uvFor(s) : undefined),
    opacityOf: (s) => opacity.get(s),
    tint: new TintProvider(colormaps),
  });
  const bakedCache = new Map<number, BakedBlockModel | null>();
  const provider: BakedModelProvider = {
    get(id) {
      if (id === AIR) return null;
      if (!bakedCache.has(id)) {
        let baked: BakedBlockModel | null = null;
        try {
          // Box-model BEs (chest/shulker/bell/sign/…) ARE the block — they're drawn by the BE renderer, so
          // skip baking a (wrong) terrain cube for them. But special-only BEs (spawner/vault/campfire/shelf/
          // beacon/brushable) keep a real TERRAIN model (cage/frame/logs/planks) that the BE only augments
          // with embedded items — those MUST still bake here, or the block vanishes leaving a floating item.
          // minecraft:moving_piston is INVISIBLE in vanilla (RenderShape.INVISIBLE — meshes nothing); the
          // transient piston pass owns the cell's visual, so skip it explicitly (PISTON_PLAN §2.1 / §5 D4)
          // rather than relying on its model JSON happening to bake empty.
          const name = palette[id]?.name;
          baked = palette[id] && name !== "minecraft:moving_piston" && !hasBlockEntityBoxModel(name) ? baker.bake(palette[id]) : null;
        } catch (e) {
          console.warn(`[scene] bake failed for id ${id}`, e); // unbakeable block → empty (BE still renders)
        }
        bakedCache.set(id, baked);
      }
      return bakedCache.get(id) ?? null;
    },
  };

  // ATL-1 incremental grow: append a new state's sprites into the atlas headroom (existing UVs invariant),
  // merge opacity, drop the new ids' cached bakes. The provider already sees new ids (`palette` is the live
  // `model.palette` reference), so once the atlas has the sprites, the next mesh of the edit's dirtied
  // sections bakes them correctly — no full rebuild, existing sections untouched. Returns false ⇒ full rebuild.
  const growPalette = async (states: ReadonlyArray<{ id: number; name: string; props: BlockProps }>): Promise<boolean> => {
    if (states.length === 0) return true;
    try {
      await resolvedPack.preloadBlocks(states.map((s) => s.name));
      const gp = resolvedPack.modelProvider();
      const want = new Set<string>();
      for (const st of states) {
        const bs = resolvedPack.getBlockState(st.name);
        if (!bs) return false; // still unknown after preload → fall back to a full rebuild
        collectStateSprites(bs, st.props, gp, want);
      }
      const need = [...want].filter((s) => !loaded.has(s));
      if (need.length > 0) {
        const grown = await resolvedPack.loadSprites(new Set(need));
        if (!atlas.append(grown.sprites)) return false; // animated sprite / headroom exhausted → full rebuild
        for (const s of grown.sprites) loaded.add(s.name);
        for (const [k, v] of grown.opacity) opacity.set(k, v);
      }
      for (const st of states) bakedCache.delete(st.id); // re-bake the new ids against the grown atlas
      return true;
    } catch (e) {
      console.warn("[scene] incremental atlas grow failed — falling back to full rebuild", e);
      return false;
    }
  };

  const renderer = new TerrainRenderer(device, atlas, device.colorFormat, device.depthFormat, undefined, renderDistanceBlocks);
  // (W6 landed the WebGL2 render-bundle replay — createRenderBundle/executeBundles now record + replay a
  // thunk list — so the F1 static-camera fast path runs on both backends; no WebGL2 bundleReplay guard.)
  return { atlas, provider, renderer, baker, pack: resolvedPack, growPalette };
}

/** Mesh + commit every 16³ section covering [0,dims) — PLUS any `extraSections` — through the
 *  invariant-enforcing store. `extraSections` carries sections OUTSIDE the dense box (negative coords or
 *  beyond dims): without them a full rebuild drops blocks placed in a negative-coordinate chunk, since the
 *  dims scan only walks [0,dims) (WorldModel.overflowSections supplies them). */
export function commitWorld(
  device: CanvasDevice,
  world: BlockSource,
  provider: BakedModelProvider,
  dims: Vec3,
  fluids?: FluidContext,
  extraSections?: readonly SectionKey[],
  meshOptions?: MeshOptions,
): SectionStore {
  const store = new SectionStore(new GpuSectionUploader(device));
  // SL-3/SL-4: smooth lighting needs apron 2 so a boundary face's diagonal corner reads stay in-bounds.
  const snapshots = new SnapshotSource(world, meshOptions?.smoothLighting ? 2 : undefined);
  const done = new Set<SectionKey>();
  const commitSection = (key: SectionKey): void => {
    if (done.has(key)) return;
    done.add(key);
    const gen = store.markDirty(key, DirtyReason.InitialLoad);
    const snap = snapshots.buildSnapshot(key, gen, DirtyReason.InitialLoad);
    if (isAllAirCore(snap)) return; // #15: empty section — don't dispatch a mesh job (fresh store ⇒ nothing presented)
    const out = meshSection(snap, provider, fluids, meshOptions); // FS-5: initial commit honors partitionFacing
    store.acceptBuild(out);
    store.commit(store.get(key)!);
  };
  for (let X = 0; X < Math.ceil(dims[0] / 16); X++)
    for (let Y = 0; Y < Math.ceil(dims[1] / 16); Y++)
      for (let Z = 0; Z < Math.ceil(dims[2] / 16); Z++) commitSection(sectionKey(X, Y, Z));
  for (const key of extraSections ?? []) commitSection(key);
  return store;
}

/**
 * Build the FluidContext from a palette + the stitched atlas. A block is water if its name is
 * `water` (or it's `waterlogged`), lava if `lava`; `level` 0 = source, 1..7 flowing. Water gets the
 * vanilla blue tint at ~0.8 opacity (translucent); lava is opaque, full-bright, untinted.
 */
export function makeFluidContext(palette: Record<number, BlockEntry>, atlas: AtlasManager): FluidContext {
  const cache = new Map<number, FluidState | null>();
  const classify = (id: number): FluidState | null => {
    const e = palette[id];
    if (!e) return null;
    const name = e.name.replace("minecraft:", "");
    const level = clampLevel(e.props.level);
    if (name === "water") return { type: FluidType.Water, level };
    if (name === "lava") return { type: FluidType.Lava, level };
    if (e.props.waterlogged === "true") return { type: FluidType.Water, level: 0 };
    return null;
  };
  // A10: the two appearances are constant per context — build them ONCE (the mesher calls appearance()
  // per fluid cell, so a per-call object + SpriteUv allocation would churn the mesh hot loop).
  const WATER_APP: FluidAppearance = { sprite: atlas.uvFor(WATER_STILL), colorRGBA: 0x3f76e4cc, translucent: true, shade: true };
  const LAVA_APP: FluidAppearance = { sprite: atlas.uvFor(LAVA_STILL), colorRGBA: 0xffffffff, translucent: false, shade: false };
  return {
    fluidOf(id) {
      if (id === AIR) return null;
      if (!cache.has(id)) cache.set(id, classify(id));
      return cache.get(id) ?? null;
    },
    appearance(type) {
      return type === FluidType.Water ? WATER_APP : LAVA_APP;
    },
  };
}

function clampLevel(raw: string | undefined): number {
  const n = Number.parseInt(raw ?? "0", 10);
  return Number.isFinite(n) ? n : 0;
}

export interface DrawTotals {
  draws: SectionDraw[];
  total: number;
  cutout: number;
  translucent: number;
}

/**
 * Build the per-pass draw commands for ONE committed section (empty when it carries no drawable geometry).
 * Resolves each pass's shared-arena CURRENT buffer + this section's byte offset HERE — the stable
 * `AllocationId`, not a buffer handle, is what `presented` carries (§11), so an arena grow/compaction is
 * invisible to `presented` as long as the caller re-resolves (collectDraws every commit, or the F3 DrawList
 * whenever the arena layout revision changes).
 */
export function buildSectionDraws(section: RenderSection): SectionDraw[] {
  const c = section.presented as GpuCommittedSection | null;
  if (!c) return [];
  const [kx, ky, kz] = key3(section.key);
  const origin: Vec3 = [kx * 16, ky * 16, kz * 16];
  const out: SectionDraw[] = [];
  for (const pass of TERRAIN_PASSES) {
    const alloc = c.vertexAlloc[pass];
    const arena = c.vertexArena[pass];
    const quadCount = c.quadCount[pass] ?? 0;
    if (alloc === undefined || !arena || quadCount === 0) continue;
    const range = arena.rangeOf(alloc);
    // baseVertex (bind-once draws): a real vertex allocation is always `4·20·quadCount` bytes, so its
    // arena byte offset is always a multiple of the 20B stride ⇒ `offset/20` is an exact vertex index, and the
    // renderer binds the arena ONCE and locates each section by baseVertex. A non-stride-aligned offset can only
    // arise from a synthetic fixture (never real geometry); for those we leave baseVertex undefined and the
    // renderer falls back to the per-draw `setVertexBuffer(offset)` bind — correct either way.
    const aligned = range.offsetBytes % VERTEX_STRIDE_BYTES === 0;
    const draw: SectionDraw = { originBlocks: origin, vertex: arena.gpuBuffer, vertexOffset: range.offsetBytes, baseVertex: aligned ? range.offsetBytes / VERTEX_STRIDE_BYTES : undefined, quadCount, pass };
    // FS-3: carry the opaque/cutout per-facing runs so FS-4 can cull camera-away slices. Opaque passes only
    // (the translucent stream stays a single un-sliced draw — TRAP 6.D global sort).
    if (pass !== TerrainPass.Translucent) {
      const counts = c.facingVertexCounts?.[pass];
      if (counts) {
        draw.facingVertexCounts = counts;
        let mask = 0;
        for (let f = 0; f < counts.length; f++) if (counts[f] > 0) mask |= 1 << f;
        draw.sliceMask = mask;
      }
    }
    if (pass === TerrainPass.Translucent) {
      // Resolve the shared INDEX arena's current buffer + byte offset HERE too. Absent ⇒ the shared quad EBO.
      const ialloc = c.indexAlloc[pass];
      const iarena = c.indexArena[pass];
      if (ialloc !== undefined && iarena) {
        draw.index = iarena.gpuBuffer;
        draw.indexOffset = iarena.rangeOf(ialloc).offsetBytes;
      }
      draw.sortType = c.sortType;
      draw.sortQuads = c.sortQuads;
      draw.cpuVertex = c.translucentVertexData;
    }
    out.push(draw);
  }
  return out;
}

/** Gather per-section, per-pass draws, carrying the translucent index buffer + sort metadata. O(world) —
 *  the full rebuild. For per-commit incremental maintenance use `DrawList` (F3). */
export function collectDraws(store: SectionStore): DrawTotals {
  const draws: SectionDraw[] = [];
  let total = 0;
  let cutout = 0;
  let translucent = 0;
  for (const section of store.all()) {
    for (const draw of buildSectionDraws(section)) {
      draws.push(draw);
      total += draw.quadCount;
      if (draw.pass === TerrainPass.Cutout) cutout += draw.quadCount;
      else if (draw.pass === TerrainPass.Translucent) translucent += draw.quadCount;
    }
  }
  return { draws, total, cutout, translucent };
}

/**
 * F3 — incremental draw-list membership. `collectDraws` rebuilds the WHOLE list (re-resolving every
 * section's arena buffer+offset, re-allocating every `SectionDraw`) on each committing tick — O(world).
 * `DrawList` keeps a per-section cache and re-resolves only the sections whose `presented` changed since
 * the last call (`store.presentedChanges`), reusing the unchanged draw objects → O(changed). When an arena
 * GREW or COMPACTED (`store.uploadLayoutRevision()` changed) every cached (buffer, offset) is stale, so it
 * falls back to a full re-resolve that call (TRAP 4.A — exactly the relocation the old once-per-commit full
 * rebuild hid). The flat array is the SAME reference when nothing changed (so the occlusion-filter cache
 * holds) and a fresh one when it does — the same contract `collectDraws` gave by returning a new array.
 */
export class DrawList {
  private readonly bySection = new Map<SectionKey, SectionDraw[]>();
  private flat: SectionDraw[] = [];
  private lastLayoutRev = -1;
  private built = false;

  /** The draw list, brought up to date with `store`'s committed/disposed sections since the last call.
   *  Drains `store.presentedChanges`, so call it once per frame that may have committed (the viewer does). */
  flatten(store: SectionStore): SectionDraw[] {
    const layoutRev = store.uploadLayoutRevision();
    let changed = false;
    if (!this.built || layoutRev !== this.lastLayoutRev) {
      // First build, or an arena relocated → every cached offset is stale: re-resolve ALL sections.
      this.bySection.clear();
      for (const section of store.all()) {
        const d = buildSectionDraws(section);
        if (d.length > 0) { this.bySection.set(section.key, d); instrument.bumpAlloc("drawList", d.length); } // X5
      }
      this.built = true;
      this.lastLayoutRev = layoutRev;
      changed = true;
    } else if (store.presentedChanges.size > 0) {
      for (const key of store.presentedChanges) {
        const section = store.get(key);
        const d = section ? buildSectionDraws(section) : []; // gone (disposed) ⇒ drop its entry
        if (d.length > 0) { this.bySection.set(key, d); instrument.bumpAlloc("drawList", d.length); } // X5: re-resolved entries
        else this.bySection.delete(key);
      }
      changed = true;
    }
    store.clearPresentedChanges();
    if (changed) {
      const flat: SectionDraw[] = [];
      for (const arr of this.bySection.values()) for (const d of arr) flat.push(d);
      this.flat = flat;
    }
    return this.flat;
  }
}

function key3(k: string): [number, number, number] {
  const [x, y, z] = k.split(",").map(Number);
  return [x, y, z];
}

// ── OCC-1 occlusion culling (camera BFS over the section visibility graph) ──────────────────────────
// `inFrustum` is supplied by the caller (it owns the camera/frustum + render distance). The BFS uses the
// store for per-section connectivity + has-geometry, then we filter the terrain draw list to the visible
// set. Empty/absent sections are traversed (OCC-6) but never drawn. Entities/transients are NOT filtered.
//
// The traversal is CLAMPED to the populated-section AABB (+1 air-shell margin) — the finite domain
// gets for free from its loaded-section graph. Without it, OCC-6 lets the BFS flood every air section in
// the frustum (far=1000 ⇒ ~150k sections/frame on an empty world — the rotation regression).

/** 1-section air shell around the populated region: keeps boundary faces + section-overflowing block
 *  models reachable (addNearbySections concern), at trivial cost (a thin surface shell). */
const OCC_BOUNDS_MARGIN = 1;

/** The clamped home section the occlusion BFS seeds from (= `floor(camPos)>>4` per axis, CLAMPED into the
 *  margin-padded `sectionBounds()` box — initOutsideWorld). This is the OCC-1 cache key: the reachable
 *  set changes only when this section crosses a boundary (or the graph revision bumps). Returns null on an
 *  empty world (no drawable geometry ⇒ nothing to seed). The clamp is what keeps a camera flying far OUTSIDE
 *  the populated region from re-keying every frame on a new (unclamped) section — it pins to the box edge. */
export function occlusionHomeSection(store: SectionStore, cameraPos: Vec3): Vec3i | null {
  const world = store.sectionBounds();
  if (!world) return null;
  const lo0 = world.lo[0] - OCC_BOUNDS_MARGIN, lo1 = world.lo[1] - OCC_BOUNDS_MARGIN, lo2 = world.lo[2] - OCC_BOUNDS_MARGIN;
  const hi0 = world.hi[0] + OCC_BOUNDS_MARGIN, hi1 = world.hi[1] + OCC_BOUNDS_MARGIN, hi2 = world.hi[2] + OCC_BOUNDS_MARGIN;
  const cx = Math.floor(cameraPos[0]) >> 4, cy = Math.floor(cameraPos[1]) >> 4, cz = Math.floor(cameraPos[2]) >> 4;
  return [clamp(cx, lo0, hi0), clamp(cy, lo1, hi1), clamp(cz, lo2, hi2)];
}

/**
 * MEM-1 region cull pre-pass options (default-off — the whole pre-pass is gated on this being supplied AND
 * `store.regionStore` being present). `regionInFrustum` tests a REGION's world-block AABB against the SAME
 * frustum the per-section `inFrustum` uses; absent (or always-true) ⇒ no region is frustum-rejected and the
 * bounds simply tighten to the populated-region union (still a valid traversal-domain shrink). Conservative
 * by construction (a region box contains its section boxes), so it never drops an in-frustum section.
 */
export interface RegionCullOptions {
  regionInFrustum?: (loX: number, loY: number, loZ: number, hiX: number, hiY: number, hiZ: number) => boolean;
}

/** Reused scratch for the tightened bounds (zero per-frame alloc; render-thread single-flight). Plain
 *  mutable arrays (filled in place); handed to the BFS as `Vec3i` (a readonly view of the same data). */
const regionBoundsScratchLo: number[] = [0, 0, 0];
const regionBoundsScratchHi: number[] = [0, 0, 0];

/**
 * MEM-1: build the BFS traversal bounds from the UNION of surviving region AABBs (+ the seed origin, always
 * visited), intersected with the global box. A region SURVIVES iff it holds a drawable section (it is in the
 * RegionStore) AND its world-block AABB passes `regionInFrustum` (always-survive when no predicate). Falls
 * back to the global box when no region survives (e.g. all out-of-frustum) so the BFS still seeds correctly
 * — the per-section `inFrustum` then drops everything anyway, preserving the empty visible set.
 */
function tightenedRegionBounds(
  regions: RegionStore,
  origin: Vec3i,
  lo: Vec3i,
  hi: Vec3i,
  opts: RegionCullOptions,
): { lo: Vec3i; hi: Vec3i } {
  const test = opts.regionInFrustum;
  let uLoX = Infinity, uLoY = Infinity, uLoZ = Infinity;
  let uHiX = -Infinity, uHiY = -Infinity, uHiZ = -Infinity;
  let any = false;
  for (const [, r] of regions.entries()) {
    if (r.drawableCount === 0) continue; // no drawable section here — nothing the BFS could return
    if (test) {
      // The region's world-block AABB, padded by the same OCC margin shell so a boundary section that
      // overhangs its region isn't false-rejected (parity with the per-section bound's +margin).
      const wLoX = (r.loX - OCC_BOUNDS_MARGIN) * 16, wLoY = (r.loY - OCC_BOUNDS_MARGIN) * 16, wLoZ = (r.loZ - OCC_BOUNDS_MARGIN) * 16;
      const wHiX = (r.hiX + 1 + OCC_BOUNDS_MARGIN) * 16, wHiY = (r.hiY + 1 + OCC_BOUNDS_MARGIN) * 16, wHiZ = (r.hiZ + 1 + OCC_BOUNDS_MARGIN) * 16;
      if (!test(wLoX, wLoY, wLoZ, wHiX, wHiY, wHiZ)) continue; // provably out-of-frustum region — reject whole
    }
    any = true;
    if (r.loX < uLoX) uLoX = r.loX;
    if (r.loY < uLoY) uLoY = r.loY;
    if (r.loZ < uLoZ) uLoZ = r.loZ;
    if (r.hiX > uHiX) uHiX = r.hiX;
    if (r.hiY > uHiY) uHiY = r.hiY;
    if (r.hiZ > uHiZ) uHiZ = r.hiZ;
  }
  if (!any) {
    // No surviving region (all out-of-frustum). Keep the global box so the seed origin still expands; the
    // per-section inFrustum prunes everything ⇒ empty visible set (parity with the flags-off run).
    regionBoundsScratchLo[0] = lo[0]; regionBoundsScratchLo[1] = lo[1]; regionBoundsScratchLo[2] = lo[2];
    regionBoundsScratchHi[0] = hi[0]; regionBoundsScratchHi[1] = hi[1]; regionBoundsScratchHi[2] = hi[2];
    return { lo: regionBoundsScratchLo as unknown as Vec3i, hi: regionBoundsScratchHi as unknown as Vec3i };
  }
  // Pad the union by the OCC margin shell (addNearbySections) and UNION in the seed origin — the
  // origin is always visited regardless of bounds, so its expansion must not be artificially blocked.
  regionBoundsScratchLo[0] = Math.min(uLoX - OCC_BOUNDS_MARGIN, origin[0]);
  regionBoundsScratchLo[1] = Math.min(uLoY - OCC_BOUNDS_MARGIN, origin[1]);
  regionBoundsScratchLo[2] = Math.min(uLoZ - OCC_BOUNDS_MARGIN, origin[2]);
  regionBoundsScratchHi[0] = Math.max(uHiX + OCC_BOUNDS_MARGIN, origin[0]);
  regionBoundsScratchHi[1] = Math.max(uHiY + OCC_BOUNDS_MARGIN, origin[1]);
  regionBoundsScratchHi[2] = Math.max(uHiZ + OCC_BOUNDS_MARGIN, origin[2]);
  // Never widen past the global box (the populated-region clamp the BFS already trusts).
  if (regionBoundsScratchLo[0] < lo[0]) regionBoundsScratchLo[0] = lo[0];
  if (regionBoundsScratchLo[1] < lo[1]) regionBoundsScratchLo[1] = lo[1];
  if (regionBoundsScratchLo[2] < lo[2]) regionBoundsScratchLo[2] = lo[2];
  if (regionBoundsScratchHi[0] > hi[0]) regionBoundsScratchHi[0] = hi[0];
  if (regionBoundsScratchHi[1] > hi[1]) regionBoundsScratchHi[1] = hi[1];
  if (regionBoundsScratchHi[2] > hi[2]) regionBoundsScratchHi[2] = hi[2];
  return { lo: regionBoundsScratchLo as unknown as Vec3i, hi: regionBoundsScratchHi as unknown as Vec3i };
}

/**
 * The set of visible terrain section keys (packed ints) from `cameraPos` via the occlusion BFS.
 *
 * `inFrustum` is supplied by the caller (it owns the camera/frustum + render distance). When `reachableOut`
 * is provided the result is written into that caller-owned set (cleared first) instead of allocating a fresh
 * one — the OCC-1 zero-alloc path; the same set object is returned. Pass an `inFrustum` that always returns
 * true (a reachability query) to get the orientation-INDEPENDENT reachable set that OCC-1 caches across a
 * pure rotation; pass the live frustum predicate (the legacy off-path) to fuse reachability + on-screen in
 * one pass, exactly as before.
 */
export function occlusionVisibleSet(
  store: SectionStore,
  cameraPos: Vec3,
  inFrustum: (sx: number, sy: number, sz: number) => boolean,
  reachableOut?: Set<number>,
  occ2?: { perPerspective?: boolean; angleMask?: boolean },
  regionCull?: RegionCullOptions,
): Set<number> {
  const set = reachableOut ?? new Set<number>();
  set.clear();
  const world = store.sectionBounds();
  if (!world) return set; // no drawable geometry anywhere — nothing to traverse or draw (empty world)

  // P1.6: fill the REUSED bounds box + clamped origin scratch (single-flight render-thread, like
  // `occVisibleScratch`) instead of allocating fresh arrays per run. `occLo`/`occHi` are the populated box.
  occLo[0] = world.lo[0] - OCC_BOUNDS_MARGIN; occLo[1] = world.lo[1] - OCC_BOUNDS_MARGIN; occLo[2] = world.lo[2] - OCC_BOUNDS_MARGIN;
  occHi[0] = world.hi[0] + OCC_BOUNDS_MARGIN; occHi[1] = world.hi[1] + OCC_BOUNDS_MARGIN; occHi[2] = world.hi[2] + OCC_BOUNDS_MARGIN;

  // Seed at the box section nearest the camera (initOutsideWorld): when the camera is OUTSIDE the
  // populated region (zoomed out), clamping the origin into the box starts the search at the boundary
  // rather than flooding the camera→world gap. For every in-box section the camera's per-axis side is
  // preserved by the clamp, so outward-direction culling is unchanged ⇒ no over-cull; the seed exits all
  // faces (conservative). Result: the search never explores beyond the populated region + margin.
  occOrigin[0] = clamp(Math.floor(cameraPos[0]) >> 4, occLo[0], occHi[0]);
  occOrigin[1] = clamp(Math.floor(cameraPos[1]) >> 4, occLo[1], occHi[1]);
  occOrigin[2] = clamp(Math.floor(cameraPos[2]) >> 4, occLo[2], occHi[2]);

  // MEM-1 region cull PRE-PASS (default-off — `regionCull` undefined). When the RegionStore tier is present
  // AND enabled, tighten the BFS `bounds` to the UNION of in-frustum, populated region AABBs (+ the seed
  // origin, always visited). PARITY (TRAP MEM-1.C): a region is rejected only if its WHOLE box fails the
  // SAME conservative frustum test; because a region box CONTAINS every section box inside it, the frustum
  // test never rejects a region holding an in-frustum section. The per-section BFS already dead-ends any
  // neighbor failing `inFrustum`, so every section it would RETURN lives in a surviving region — feeding the
  // survivors' union as bounds drops nothing from the visible SET; only the traversal domain shrinks. The
  // per-section BFS, its `inFrustum`, and `sectionAt` are UNCHANGED ⇒ byte-identical draw output.
  const regions = store.regionStore;
  // Reuse the single query object: only the fields that vary per run are reassigned (single-flight safe).
  occQuery.bounds = regionCull && regions
    ? tightenedRegionBounds(regions, occOrigin as unknown as Vec3i, occLo as unknown as Vec3i, occHi as unknown as Vec3i, regionCull)
    : occBoundsBox;
  occQuery.inFrustum = inFrustum;
  // OCC-2 (default-off): pass the WORLD camera position (sub-section accuracy, TRAP OCC-2 quadrant) only when
  // perPerspective is on, so the BFS joins each section's DIRECTION_SETS by camera quadrant into a SUBSET of
  // the symmetric mask. Absent ⇒ the exact symmetric BFS (off-path, byte-identical visible set).
  occQuery.cameraPos = occ2?.perPerspective ? cameraPos : undefined;
  occQuery.angleMask = occ2?.perPerspective ? occ2.angleMask === true : false;
  occQueryStore = store; // read by the shared `occSectionAt` closure during the (synchronous) BFS only
  const visible = computeVisibleSections(occQuery, occVisibleScratch);
  occQueryStore = null; // don't retain the store across calls (a scene swap could otherwise pin the old one)
  for (const v of visible) set.add(packSectionCoord(v[0], v[1], v[2]));
  return set;
}

// ── P1.6 reused BFS query scaffolding (single-flight render-thread, like `occVisibleScratch`) ────────────────
/** Reused [sx,sy,sz]-triple drain buffer for the BFS result (filled by `computeVisibleSections`, immediately
 *  copied into the packed-int set). Render-thread single-flight, same as the BFS scratch it backs. */
const occVisibleScratch: Vec3i[] = [];
// Mutable scratch we write through (Vec3i is readonly, so the query holds cast views of the same arrays).
const occLo: number[] = [0, 0, 0];
const occHi: number[] = [0, 0, 0];
const occOrigin: number[] = [0, 0, 0];
const occBoundsBox = { lo: occLo as unknown as Vec3i, hi: occHi as unknown as Vec3i }; // populated-box bounds (region cull substitutes its own)
/** The store the BFS is currently reading — set immediately before `computeVisibleSections` and cleared right
 *  after (the BFS is synchronous, so the shared `sectionAt` closure always sees the right store). */
let occQueryStore: SectionStore | null = null;
const occSectionAt = (sx: number, sy: number, sz: number): SectionInfo | null => {
  const s = occQueryStore!.getByCoords(sx, sy, sz); // S1: numeric lookup, no `"sx,sy,sz"` string per BFS node
  return s && s.presented ? s : null; // null ⇒ absent/uncommitted ⇒ VIS_ALL + no geometry (traversable — OCC-6)
};
/** The single reused query object — per-run varying fields are reassigned in `occlusionVisibleSet`. */
const occQuery: OcclusionQuery = { origin: occOrigin as unknown as Vec3i, bounds: occBoundsBox, inFrustum: () => true, sectionAt: occSectionAt };


/** Filter terrain draws to the occlusion-visible section set (in place into `out`). */
export function filterDrawsByOcclusion(
  draws: readonly SectionDraw[],
  visible: Set<number>,
  out: SectionDraw[],
): SectionDraw[] {
  out.length = 0;
  for (const d of draws) {
    const o = d.originBlocks;
    if (visible.has(packSectionCoord(o[0] >> 4, o[1] >> 4, o[2] >> 4))) out.push(d); // S1: numeric key, no string
  }
  return out;
}

/** Every entity + block-entity skin the entity atlas stitches (Phase 4.5b/4.5c). Box-model skins
 * (sheep/pig/cow/minecart) come from the box-model registry; BE skins from `allBETextures()`. */
export const ENTITY_TEXTURES = [...new Set([...entityModelTextures(), ...allBETextures()])];

/**
 * Build an EntityWorld over a terrain Scene: stitches the entity atlas from the pack's entity textures
 * and wires the shared block baker (block-display entities + BE contents). Box-model entities + BEs only
 * render if their textures are present in the pack; otherwise it degrades to block-display entities.
 */
export async function buildEntityWorld(device: CanvasDevice, scene: Scene, renderDistanceBlocks?: number, itemIds?: readonly string[]): Promise<EntityWorld> {
  const raw = await scene.pack.loadRawTextures(ENTITY_TEXTURES);
  // Scan per-glyph advance widths from the bitmap font (for sign text layout — vanilla does the same).
  const font = raw.find((t) => t.name === FONT_TEX);
  if (font) setFontWidths(scanFontWidths(font.rgba, font.width, font.height));
  const ownedAtlases: { dispose(): void }[] = []; // O1: atlas managers released on EntityWorld.dispose
  let mgr: EntityAtlasManager | undefined;
  if (raw.length > 0) {
    mgr = new EntityAtlasManager(device);
    mgr.build(raw);
    ownedAtlases.push(mgr);
  }
  // End-portal shader textures (sampled directly, repeat-wrapped — NOT atlased, which would clamp).
  const portal = await scene.pack.loadRawTextures(["environment/end_sky", "entity/end_portal/end_portal"]);
  const byName = (id: string) => portal.find((t) => t.name === id);
  return new EntityWorld({
    device,
    colorFormat: device.colorFormat,
    depthFormat: device.depthFormat,
    bakeBlock: (name, props) => scene.baker.bake({ name, props }),
    blockAtlas: scene.atlas.texture,
    entityAtlas: mgr?.texture,
    entityUvFor: mgr ? (name) => (mgr!.has(name) ? mgr!.uvFor(name) : undefined) : undefined,
    renderDistanceBlocks,
    portalTextures: { endSky: byName("environment/end_sky"), endPortal: byName("entity/end_portal/end_portal") },
    // Hand the EntityWorld's own entity atlas to the item geometry so `builtin/entity` items (shulker box,
    // chest, …) bake their BE box model against it instead of stitching a duplicate.
    itemGeometry: await buildItemGeometry(device, scene, itemIds, ownedAtlases, mgr ? { atlas: mgr.texture, uvFor: (s) => (mgr!.has(s) ? mgr!.uvFor(s) : undefined) } : undefined),
    ownedAtlases,
  });
}

/**
 * Build item geometry for a set of item ids: resolve each (generated sprite vs block model), load the
 * generated item sprites into a dedicated item atlas (keeping RGBA for the silhouette extrusion), and wire
 * the block baker for block items. Exposed so the simulator can preload its inventory's items.
 */
export async function buildItemGeometry(device: CanvasDevice, scene: Scene, itemIds?: readonly string[], ownedAtlases?: { dispose(): void }[], entity?: { atlas: GpuTextureHandle; uvFor: (s: string) => SpriteUv | undefined }): Promise<ItemGeometry | undefined> {
  if (!itemIds || itemIds.length === 0) return undefined;
  const resolved = new Map<string, ResolvedItem>();
  const spriteIds = new Set<string>();
  // Resolve item models in parallel (each is an independent JSON read) — sequential awaits over a large
  // inventory (~1.5k items) would dominate load time.
  const entries = await Promise.all(
    itemIds.map(async (id) => {
      const name = id.replace(/^minecraft:/, "");
      return [name, resolveItem(name, await scene.pack.readItemModel(name))] as const;
    }),
  );
  for (const [name, r] of entries) {
    resolved.set(name, r);
    if (r.kind === "generated" && r.sprite) spriteIds.add(r.sprite);
  }
  const rawSprites = await scene.pack.loadRawTextures([...spriteIds]);
  const itemSprites = new Map<string, ItemSprite>(rawSprites.map((t) => [t.name, { rgba: t.rgba, width: t.width, height: t.height }]));
  let itemAtlas = scene.atlas.texture;
  let itemUvFor: (s: string) => SpriteUv | undefined = () => undefined;
  if (rawSprites.length > 0) {
    const mgr = new EntityAtlasManager(device);
    mgr.build(rawSprites);
    itemAtlas = mgr.texture;
    itemUvFor = (s) => (mgr.has(s) ? mgr.uvFor(s) : undefined);
    ownedAtlases?.push(mgr); // O1: released on EntityWorld.dispose
  }
  // `builtin/entity` items (shulker box, …) bake their BE box model against the ENTITY atlas. Reuse the
  // EntityWorld's (passed via `entity`); the standalone GUI path (no EntityWorld) stitches its own.
  let entityAtlas = entity?.atlas;
  let entityUvFor = entity?.uvFor;
  if (!entityAtlas && [...resolved.values()].some((r) => r.kind === "entity")) {
    const rawBE = await scene.pack.loadRawTextures(ENTITY_TEXTURES);
    if (rawBE.length > 0) {
      const mgr = new EntityAtlasManager(device);
      mgr.build(rawBE);
      entityAtlas = mgr.texture;
      entityUvFor = (s) => (mgr.has(s) ? mgr.uvFor(s) : undefined);
      ownedAtlases?.push(mgr);
    }
  }
  return new ItemGeometry({
    resolved,
    itemSprites,
    itemUvFor,
    itemAtlas,
    bakeBlock: (name, props) => scene.baker.bake({ name, props }),
    blockAtlas: scene.atlas.texture,
    entityAtlas,
    entityUvFor,
  });
}
