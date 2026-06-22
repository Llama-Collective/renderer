// Minecraft block-model + blockstate JSON types (as in a resource pack's
// assets/minecraft/{models,blockstates}/...). RENDERER_PLAN §24.3–§24.4.
//
// These mirror the vanilla JSON exactly (verified against public/pack.zip, pack_format 84). Pure
// data, GPU-free. Namespaces ("minecraft:") and `#refs` are resolved by ModelResolver; the bakery
// consumes the Resolved* shapes below.

import { Direction } from "../../types";

/** The six model face names, JSON spelling, in `Direction` enum order [Down,Up,North,South,West,East]. */
export const FACE_NAMES = ["down", "up", "north", "south", "west", "east"] as const;
export type FaceName = (typeof FACE_NAMES)[number];

/** FaceName → Direction enum. */
export const FACE_TO_DIRECTION: Record<FaceName, Direction> = {
  down: Direction.Down,
  up: Direction.Up,
  north: Direction.North,
  south: Direction.South,
  west: Direction.West,
  east: Direction.East,
};

// --- Raw JSON (verbatim from the pack) -------------------------------------------

/** A texture slot value: a sprite id / `#ref` string, or the 26.x object form (glass panes). */
export type RawTexture = string | { sprite?: string; force_translucent?: boolean };

export interface RawFace {
  /** `[u1,v1,u2,v2]` in 0..16; omitted → derived from element from/to. */
  uv?: [number, number, number, number];
  /** Sprite ref: `#slot` or a literal sprite id. */
  texture: string;
  /** Face this quad is culled against; absent → never culled. */
  cullface?: string;
  /** 0/90/180/270 — rotates which UV corner maps to each vertex. */
  rotation?: number;
  /** Tint source index; absent/−1 → untinted. */
  tintindex?: number;
}

export interface RawElement {
  from: [number, number, number];
  to: [number, number, number];
  rotation?: { origin: [number, number, number]; axis: "x" | "y" | "z"; angle: number; rescale?: boolean };
  /** Per-element directional shade; default true. */
  shade?: boolean;
  faces: Partial<Record<FaceName, RawFace>>;
}

export interface RawBlockModel {
  parent?: string;
  ambientocclusion?: boolean;
  textures?: Record<string, RawTexture>;
  elements?: RawElement[];
}

/** One `{model, x, y, z, uvlock, weight}` entry (blockstate variant / multipart apply). */
export interface RawVariant {
  model: string;
  x?: number;
  y?: number;
  z?: number;
  uvlock?: boolean;
  weight?: number;
}

/** Multipart `when`: AND-of-keys (value `a|b` = OR, leading `!` = negate) or explicit OR/AND. */
export type RawWhen = { OR: RawWhen[] } | { AND: RawWhen[] } | Record<string, string>;

export interface RawMultipartCase {
  when?: RawWhen;
  apply: RawVariant | RawVariant[];
}

export interface RawBlockState {
  variants?: Record<string, RawVariant | RawVariant[]>;
  multipart?: RawMultipartCase[];
}

// --- Resolved (post parent-merge + #ref resolution) ------------------------------

/** A fully-resolved sprite reference: no namespace, no `#`. */
export interface ResolvedTexture {
  /** Sprite id, e.g. "block/oak_planks". */
  sprite: string;
  /** Drives the TRANSLUCENT render layer regardless of pixel alpha (TRAP 7.C). */
  forceTranslucent: boolean;
}

export interface ResolvedModel {
  ambientocclusion: boolean;
  /** slot → resolved sprite (all `#refs` followed to a literal). */
  textures: Record<string, ResolvedTexture>;
  elements: RawElement[];
}

/** A blockstate part to bake: a model id + its 90° rotation + uvlock. */
export interface VariantPart {
  model: string;
  x: number;
  y: number;
  z: number;
  uvlock: boolean;
}

/** Strip a "minecraft:" (or any) namespace prefix from an id. */
export function stripNamespace(id: string): string {
  const colon = id.indexOf(":");
  return colon >= 0 ? id.slice(colon + 1) : id;
}
