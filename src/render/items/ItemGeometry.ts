// Item geometry — the shared core both item forms (dropped entity + inventory slot) reuse.
// RENDERER_PLAN.md §18. Resolves an item id to either a `generated` flat sprite (→ extruded 3D model,
// vanilla `ItemModelGenerator`) or a `block` item (→ the block's baked model). Returns the model's quads
// in [0,1] block space + which atlas to sample; the caller applies the per-context display transform
// (`itemDisplay.ts`) and a placement matrix. GPU-free.

import type { GpuTextureHandle } from "../../core/GraphicsDevice";
import type { RawBlockModel } from "../../mesh/model/ModelTypes";
import type { BakedBlockModel } from "../../mesh/model/BakedBlockModel";
import type { BlockProps } from "../../mesh/model/BlockStateResolver";
import { TERRAIN_PASSES, type SpriteUv } from "../../types";
import type { RawVertex } from "../../mesh/VertexFormat";
import { bakedModelToEntityVerts, parseBlockState } from "../entities/blockDisplay";
import { bakeBlockEntity, hasBlockEntityBoxModel } from "../entities/blockentities";
import type { BlockEntityRecord } from "../../world/BlockEntityIndex";
import { extrudeItemSprite, type ItemSprite } from "./itemExtrude";
import type { ItemKind } from "./itemDisplay";

const strip = (s: string): string => s.replace(/^minecraft:/, "");

export interface ResolvedItem {
  kind: ItemKind;
  sprite?: string; // generated: the `item/<name>` sprite key
  block?: string; // block / entity: the block id
}

/**
 * Resolve an item id from its `models/item/<name>.json`:
 *  - a `layer0` texture → `generated` (extruded flat sprite);
 *  - else a `builtin/entity` item whose id is a REPLACEMENT box block-entity (shulker box, chest, banner,
 *    bed, decorated pot, …) → `entity` (renders as the BE box model — vanilla draws these via the BE
 *    renderer, they have NO flat sprite and an EMPTY terrain block model);
 *  - else `block` (the baked block model).
 * `hasBlockEntityBoxModel` excludes hybrid/special-only BEs (lectern/enchanting_table/bell), which keep a
 * real terrain block model and must stay on the `block` path.
 */
export function resolveItem(name: string, model: RawBlockModel | undefined): ResolvedItem {
  const layer0 = model?.textures?.layer0;
  if (typeof layer0 === "string") return { kind: "generated", sprite: strip(layer0) };
  const bare = strip(name);
  if (hasBlockEntityBoxModel(bare)) return { kind: "entity", block: bare };
  return { kind: "block", block: bare };
}

export interface ItemGeom {
  kind: ItemKind;
  verts: RawVertex[]; // model quads in [0,1] block space
  texture: GpuTextureHandle; // atlas to sample (item atlas for generated, block atlas for block)
}

export interface ItemGeometryDeps {
  resolved: Map<string, ResolvedItem>;
  itemSprites: Map<string, ItemSprite>; // sprite id → decoded RGBA (for the silhouette extrusion)
  itemUvFor: (sprite: string) => SpriteUv | undefined;
  itemAtlas: GpuTextureHandle;
  bakeBlock: (name: string, props: BlockProps) => BakedBlockModel | null;
  blockAtlas: GpuTextureHandle;
  /** `builtin/entity` items (shulker box, …) bake their BE box model against the ENTITY atlas. Absent when
   *  no entity-kind item was resolved (then `kind: "entity"` items return null — nothing to render). */
  entityAtlas?: GpuTextureHandle;
  entityUvFor?: (texture: string) => SpriteUv | undefined;
}

/** Cache of resolved + built item geometry (extruded sprite or baked block model). */
export class ItemGeometry {
  private readonly cache = new Map<string, ItemGeom | null>();

  constructor(private readonly deps: ItemGeometryDeps) {}

  /** Build (once) the item's model quads + atlas. Returns null if unrenderable (missing sprite/model). */
  get(item: string): ItemGeom | null {
    const key = strip(item);
    if (!this.cache.has(key)) this.cache.set(key, this.build(key));
    return this.cache.get(key) ?? null;
  }

  private build(key: string): ItemGeom | null {
    const r = this.deps.resolved.get(key) ?? { kind: "block" as const, block: key };
    if (r.kind === "generated" && r.sprite) {
      const sprite = this.deps.itemSprites.get(r.sprite);
      const rect = this.deps.itemUvFor(r.sprite);
      if (!sprite || !rect) return null;
      return { kind: "generated", verts: extrudeItemSprite(sprite, rect), texture: this.deps.itemAtlas };
    }
    if (r.kind === "entity" && r.block) {
      // builtin/entity item → bake its BE box model in MODEL-LOCAL [0,1] space (sectionOrigin [0,0,0] +
      // facing "up" so the box sits upright for GUI/ground/frame, like vanilla ItemRenderer draws it).
      const atlas = this.deps.entityAtlas;
      const uvFor = this.deps.entityUvFor;
      if (!atlas || !uvFor) return null;
      const rec: BlockEntityRecord = { x: 0, y: 0, z: 0, type: r.block, props: { facing: "up" }, animating: false };
      const verts = bakeBlockEntity(rec, { uvFor, sectionOrigin: [0, 0, 0] }, 0, false);
      return verts && verts.length ? { kind: "entity", verts, texture: atlas } : null;
    }
    // Block item — the ordinary block model (all passes flattened; items render alpha-tested).
    const { name, props } = parseBlockState(r.block ?? key);
    const baked = this.deps.bakeBlock(name, props);
    if (!baked) return null;
    const byPass = bakedModelToEntityVerts(baked);
    const verts: RawVertex[] = [];
    for (const pass of TERRAIN_PASSES) if (byPass[pass]) verts.push(...byPass[pass]!);
    return verts.length ? { kind: "block", verts, texture: this.deps.blockAtlas } : null;
  }
}
