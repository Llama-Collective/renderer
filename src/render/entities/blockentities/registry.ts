// Block-entity renderer registry. RENDERER_PLAN.md §18, Phase 4.5c (the BBE-adopted hybrid model).
//
// Every block entity is a `BlockEntityDef`. Two render kinds, both optional per def:
//   - BOX geometry (`bake`): vanilla `ModelPart` box models baked in SECTION-LOCAL coords (so idle ones
//     live in the section's static buffer + cull with it; animating ones re-pose per frame). Build these
//     with `boxBE({...})` — a declarative config (parts + texture + transform + animation).
//   - SPECIAL draws (`special`): non-box renderers (beacon beam, end portal, wireframe box, embedded
//     spinning block/item) returned as descriptors the scene dispatches to dedicated renderers.
//
// Defs register themselves (see blockentities/index.ts) keyed by BE type id. The hybrid classifier
// (`splitSectionBEs`) and the scene baker consume the registry — no per-type branching in the engine.

import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";
import type { RawVertex } from "../../../mesh/VertexFormat";
import { bakeModel, type ModelPartDef, type PartPose } from "../../../mesh/entity/ModelPart";
import { TerrainPass, type SpriteUv, type Vec3 } from "../../../types";
import { sectionLocal, type Mat4 } from "./transforms";

/** Context for baking a BE's geometry into a section. */
export interface BEBakeContext {
  uvFor: (texture: string) => SpriteUv | undefined;
  /** Section world origin in blocks (geometry is emitted relative to this). */
  sectionOrigin: Vec3;
  /** Camera world position (present only on the per-frame animating path) — for camera-facing billboards
   *  like the conduit eye. Absent on the cached static bake (those parts fall back to a fixed orientation). */
  cameraPos?: Vec3;
  /** Container lid openness ∈ [0,1] this frame for an `openable` BE (chest/shulker), integrated by the
   *  baker toward the BE's `open` target at the vanilla rate. Drives the lid pose (chest cubic-eased lift,
   *  shulker lid drop + 270° spin). Absent ⇒ closed (0): the cached static bake passes none, so idle/closed
   *  containers bake with a shut lid. See BlockEntitySceneBaker. */
  openness?: number;
}

/** A special (non-box) per-frame draw a BE can request; the scene routes each to its renderer. */
export type BESpecialDraw =
  | { kind: "beam"; pos: Vec3; color: number; height: number; gateway?: boolean }
  | { kind: "portal"; pos: Vec3; gateway?: boolean }
  | { kind: "lineBox"; min: Vec3; max: Vec3; color: number; xray?: boolean }
  | { kind: "blockModel"; block: string; model: Mat4 }
  | { kind: "item"; item: string; model: Mat4 };

export interface BlockEntityDef {
  /** BE type ids this def renders (namespaced or bare). */
  types: readonly string[];
  /** Entity-atlas texture keys this def needs stitched. */
  textures: readonly string[];
  /** True if the BE re-poses every frame when its `animating` flag is set (chest lid, bell, …). */
  animated: boolean;
  /** IDLE-LOOP (BBE §5, the "inverted animation-trigger"): the BE's motion is a CONTINUOUS render-clock
   *  loop vanilla always shows (banner wave, conduit spin + camera-facing eye), NOT an event a user toggles.
   *  An idle-loop BE is ALWAYS classified into the per-frame path (`splitSectionBEs`) regardless of its
   *  record `animating` flag — so it animates by default instead of baking frozen. Implies `animated`.
   *  Event-driven BEs (chest lid / bell swing / shulker open / pot wobble) leave this false: they bake idle
   *  and animate only when the editor toggles `animating` (the deliberate-preview path). */
  idleLoop?: boolean;
  /** OPENABLE container (chest / trapped / ender / shulker): its lid animation is driven by a per-position
   *  openness ∈ [0,1] that the baker integrates toward the BE's `open` target (set from the sim's
   *  open-viewer count). The BE is drawn per-frame WHILE openness > 0 (opening / open / closing) and baked
   *  into the static section mesh (lid shut) when fully closed — so idle closed containers stay BBB-static
   *  (AO-lit, cullable) and only an actively-open one costs a per-frame draw. `animate(rec, clock, ctx)`
   *  reads `ctx.openness` for the lid pose. Implies `animated`. */
  openable?: boolean;
  /** HYBRID: the box geometry only ADDS to a real terrain frame (bell posts/bars, lectern stand,
   *  enchanting-table cube) instead of REPLACING the block — so terrain must NOT be suppressed. Default
   *  false: a box BE replaces the block (chest/shulker/sign/skull/banner/pot/conduit have empty terrain). */
  hybridTerrain?: boolean;
  /** Bake section-local box geometry; `animating` selects rest vs animated pose. null = nothing. */
  bake?(rec: BlockEntityRecord, ctx: BEBakeContext, clock: number, animating: boolean): RawVertex[] | null;
  /** Per-frame special draws (drawn every frame regardless of the static/animating split). */
  special?(rec: BlockEntityRecord, clock: number): BESpecialDraw[];
}

/** Declarative box-model BE config (the common case). */
export interface BoxBEConfig {
  types: readonly string[];
  texWidth: number;
  texHeight: number;
  /** Model parts — fixed, or selected per record (chest single/left/right, bed head/foot, sign variants). */
  parts: ModelPartDef[] | ((rec: BlockEntityRecord) => ModelPartDef[]);
  /** Atlas key to sample for a given record (resolves dye/wood/copper/variant). */
  texture: (rec: BlockEntityRecord) => string;
  /** Model→block-local transform (per-FACING rotation, flat-lay, scale). */
  transform: (rec: BlockEntityRecord) => Mat4;
  /** All atlas keys this BE may use (so the atlas stitches them). */
  textures: readonly string[];
  /** Per-instance vertex tint 0xRRGGBBAA (bed/shulker dye etc.); default opaque white. */
  tint?: (rec: BlockEntityRecord) => number;
  /** Animated part poses (lid open, swing, …) — only applied when `animating`. `ctx.openness` carries the
   *  container lid openness for `openable` BEs (chest/shulker); time-driven motions (bell swing) ignore it. */
  animate?: (rec: BlockEntityRecord, clock: number, ctx: BEBakeContext) => Record<string, PartPose> | undefined;
  /** Override the auto `animated` flag (default = has an `animate` fn). */
  animated?: boolean;
  /** OPENABLE container — lid driven by `ctx.openness` (chest/shulker). See BlockEntityDef.openable. */
  openable?: boolean;
  /** HYBRID: box geometry ADDS to a real terrain frame instead of replacing the block (bell/lectern/
   *  enchanting_table) — terrain must keep baking the block. See BlockEntityDef.hybridTerrain. */
  hybridTerrain?: boolean;
  /** Optional extra special draws (e.g. shelf/campfire items) alongside the box. */
  special?: (rec: BlockEntityRecord, clock: number) => BESpecialDraw[];
}

/** Build a box-model BlockEntityDef from declarative config. */
export function boxBE(cfg: BoxBEConfig): BlockEntityDef {
  return {
    types: cfg.types,
    textures: cfg.textures,
    animated: cfg.animated ?? !!cfg.animate,
    openable: cfg.openable,
    hybridTerrain: cfg.hybridTerrain,
    special: cfg.special,
    bake(rec, ctx, clock, animating) {
      const rect = ctx.uvFor(cfg.texture(rec));
      if (!rect) return null;
      const o = ctx.sectionOrigin;
      const base = sectionLocal(rec, o, cfg.transform(rec));
      const poses = animating ? cfg.animate?.(rec, clock, ctx) : undefined;
      const parts = typeof cfg.parts === "function" ? cfg.parts(rec) : cfg.parts;
      return bakeModel(parts, {
        texWidth: cfg.texWidth,
        texHeight: cfg.texHeight,
        uvRect: rect,
        base,
        poses,
        color: cfg.tint?.(rec),
        pass: TerrainPass.Cutout,
        shade: true,
      });
    },
  };
}

// ── Registry ────────────────────────────────────────────────────────────────

const registry = new Map<string, BlockEntityDef>();

export function registerBE(def: BlockEntityDef): void {
  for (const t of def.types) registry.set(strip(t), def);
}

export function getBEDef(type: string): BlockEntityDef | undefined {
  return registry.get(strip(type));
}

export function hasBEDef(type: string): boolean {
  return registry.has(strip(type));
}

export function allBEDefs(): BlockEntityDef[] {
  return [...new Set(registry.values())];
}

/** Every entity-atlas texture key any registered BE needs. */
export function allBETextures(): string[] {
  return [...new Set(allBEDefs().flatMap((d) => d.textures))];
}

/** Registered BE type ids (bare, deduped) — for demos / enumeration. */
export function registeredBETypes(): string[] {
  return [...registry.keys()];
}

/** Strip the `minecraft:` prefix from a block-entity type id. The single home of this strip — the def
 *  files import it instead of re-typing the regex. */
export function strip(type: string): string {
  return type.replace(/^minecraft:/, "");
}

/** The 12 wood types that gain sign + shelf block-entity variants. ORDER IS LOAD-BEARING for signs: it
 *  drives `SIGN.textures`, which feeds the entity-atlas stitch order ⇒ the baked UVs — so this is signs'
 *  exact historical order. Shelf uses it order-independently (its only consumers are a type list + a
 *  prefix `find`, both order-neutral). The single home of the list `signs.ts`/`special.ts` each hand-kept. */
export const WOOD_TYPES = ["oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "crimson", "warped", "mangrove", "cherry", "bamboo", "pale_oak"];
