// The app/editor world model: a SPARSE, chunk-keyed block grid + palette interner that presents a
// `BlockSource` to the mesher, plus the block-entity and entity lists extracted from an EditableSchematic.
// RENDERER_INTEGRATION_PLAN.md Step 1. This is the authoritative block state the SchematicViewer façade owns.
//
// Storage is one dense 16³ Uint16Array per occupied 16³ section, held in a Map keyed by SectionKey. This
// works for ANY coordinate — including NEGATIVE chunks — with no [0,dims) box and no overflow special case
// (the earlier dense-[0,dims)+overflow design dropped negative-coordinate blocks on rebuild). Local indices
// use `coord & 15`, which is the correct in-section offset for negative coordinates too. Block ids are
// palette indices: 0 = AIR (never stored).

import type { BlockSource } from "../world/SnapshotSource";
import type { BlockEntry } from "../mesh/model/BakedBlockModel";
import { sectionOfBlock, type SectionKey } from "../world/SectionKey";
import { hasBlockEntityModel } from "../render/entities/blockentities";
import type { EditableBlock, EditableEntity, EditableSchematic } from "../../../schematic-io/types";

export const AIR = 0;

const SECTION_VOLUME = 16 * 16 * 16;

/** A block carrying block-entity data (sign text / NBT) — routed to the BE renderer, not terrain. */
export interface BlockEntityCandidate {
  x: number;
  y: number;
  z: number;
  /** The block name (also the BE type id, e.g. "minecraft:chest"). */
  name: string;
  props: Readonly<Record<string, string>>;
  block: EditableBlock;
}

/** Stable, order-independent key for a (name, props) blockstate. */
export function blockStateKey(name: string, props: Readonly<Record<string, string>>): string {
  const keys = Object.keys(props).sort();
  if (keys.length === 0) return name;
  let s = name + "|";
  for (let i = 0; i < keys.length; i++) s += `${keys[i]}=${props[keys[i]]};`;
  return s;
}

/** Result of a single-block mutation, for the dirty/remesh path. */
export interface SetBlockResult {
  changed: boolean;
  oldId: number;
  newId: number;
}

/** Axis-aligned block bounds: `min` inclusive, `max` EXCLUSIVE (one past the last occupied cell). */
export interface WorldBounds {
  min: [number, number, number];
  max: [number, number, number];
}

export class WorldModel {
  /** Palette id → blockstate. Id 0 (AIR) is implicit and never stored. */
  readonly palette: Record<number, BlockEntry> = {};
  private readonly interner = new Map<string, number>();
  private nextId = 1;

  /** One dense 16³ id array per occupied section (any coordinate, incl. negative). */
  private readonly sections = new Map<SectionKey, Uint16Array>();
  /** Inclusive bounding box of placed (non-air) cells, for camera framing + the grid (expands only). */
  private bMin: [number, number, number] | null = null;
  private bMax: [number, number, number] | null = null;

  /** The immutable read view handed to SnapshotSource/mesher. World coords; empty cell → AIR. */
  get source(): BlockSource {
    return { getBlock: (x, y, z) => this.getBlock(x, y, z) };
  }

  /** In-section offset for (x,y,z); `& 15` is the correct local index for negative coords too. */
  private static local(x: number, y: number, z: number): number {
    return (((y & 15) * 16) + (z & 15)) * 16 + (x & 15);
  }

  getBlock(x: number, y: number, z: number): number {
    const arr = this.sections.get(sectionOfBlock(x, y, z));
    return arr ? arr[WorldModel.local(x, y, z)] : AIR;
  }

  /** Intern a blockstate into the palette, returning its id. AIR maps to 0. */
  internBlock(name: string, props: Readonly<Record<string, string>>): number {
    if (name === "minecraft:air" || name === "air") return AIR;
    const key = blockStateKey(name, props);
    let id = this.interner.get(key);
    if (id === undefined) {
      id = this.nextId++;
      this.interner.set(key, id);
      this.palette[id] = { name, props: { ...props } };
    }
    return id;
  }

  /** Set a block by state; returns whether the cell's id changed (drives remesh). */
  setBlock(x: number, y: number, z: number, name: string, props: Readonly<Record<string, string>>): SetBlockResult {
    const oldId = this.getBlock(x, y, z);
    const newId = this.internBlock(name, props);
    if (newId !== oldId) {
      const key = sectionOfBlock(x, y, z);
      let arr = this.sections.get(key);
      if (!arr) {
        arr = new Uint16Array(SECTION_VOLUME);
        this.sections.set(key, arr);
      }
      arr[WorldModel.local(x, y, z)] = newId;
      if (newId !== AIR) this.expandBounds(x, y, z);
    }
    return { changed: newId !== oldId, oldId, newId };
  }

  /** Remove a block (set to AIR); returns whether it changed. */
  removeBlock(x: number, y: number, z: number): SetBlockResult {
    const oldId = this.getBlock(x, y, z);
    if (oldId !== AIR) {
      const arr = this.sections.get(sectionOfBlock(x, y, z));
      if (arr) arr[WorldModel.local(x, y, z)] = AIR;
    }
    return { changed: oldId !== AIR, oldId, newId: AIR };
  }

  /** Section keys that have storage — exactly the chunks `commitWorld` must mesh (any coordinate). */
  occupiedSections(): SectionKey[] {
    return [...this.sections.keys()];
  }

  /** Bounding box of placed blocks (`max` exclusive), or null if the world is empty. For framing/grid. */
  bounds(): WorldBounds | null {
    if (!this.bMin || !this.bMax) return null;
    return { min: [...this.bMin], max: [this.bMax[0] + 1, this.bMax[1] + 1, this.bMax[2] + 1] };
  }

  private expandBounds(x: number, y: number, z: number): void {
    if (!this.bMin || !this.bMax) {
      this.bMin = [x, y, z];
      this.bMax = [x, y, z];
      return;
    }
    if (x < this.bMin[0]) this.bMin[0] = x;
    if (y < this.bMin[1]) this.bMin[1] = y;
    if (z < this.bMin[2]) this.bMin[2] = z;
    if (x > this.bMax[0]) this.bMax[0] = x;
    if (y > this.bMax[1]) this.bMax[1] = y;
    if (z > this.bMax[2]) this.bMax[2] = z;
  }
}

/** Whether an EditableBlock should be drawn by the block-entity renderer instead of as terrain. True if
 *  it carries BE data (sign text / NBT) OR its block TYPE is one vanilla renders via a BlockEntityRenderer
 *  (chest, shulker, bell, sign, banner, …) — those have no/empty terrain models, so they must route to the
 *  BE path regardless of contents (else they fall back to a wrong terrain cube or vanish). */
export function hasBlockEntityData(block: EditableBlock): boolean {
  return block.signText !== undefined || block.blockEntity !== undefined || hasBlockEntityModel(block.name);
}

export interface LoadedWorld {
  model: WorldModel;
  /** Blocks carrying block-entity data (signs, containers, …) — candidates for the BE renderer. */
  blockEntities: BlockEntityCandidate[];
  entities: EditableEntity[];
}

/**
 * Build a WorldModel + extracted block-entity / entity lists from a parsed EditableSchematic.
 * The full palette is interned up front so the scene builder can atlas + bake every referenced block
 * before the first commit (the tick-0 synchronous-load invariant).
 */
export function loadWorld(schematic: EditableSchematic): LoadedWorld {
  const model = new WorldModel();
  const blockEntities: BlockEntityCandidate[] = [];
  for (const block of schematic.blocks) {
    model.setBlock(block.x, block.y, block.z, block.name, block.properties);
    if (hasBlockEntityData(block)) {
      blockEntities.push({ x: block.x, y: block.y, z: block.z, name: block.name, props: block.properties, block });
    }
  }
  return { model, blockEntities, entities: schematic.entities ?? [] };
}
