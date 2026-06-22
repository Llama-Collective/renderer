// Simulation-accurate entity bounding boxes (the axis-aligned collision AABB), for the wireframe outline
// drawn around every entity. RENDERER_PLAN.md §18.
//
// The renderer is intentionally decoupled from `simulation/` (see EntityScene.ts), and the sim does NOT
// serialize per-entity width/height — so the dimensions are mirrored here, keyed by entity type string.
// They MUST match the sim's EntityDimensions constants:
//   minecart  0.98 × 0.7     simulation/entity/minecart.ts   (MINECART_DIMENSIONS)
//   boat/raft 1.375 × 0.5625 simulation/entity/boat.ts       (BOAT_DIMENSIONS)
//   sheep     0.9 × 1.3      simulation/entity/sheep.ts      (SHEEP_DIMENSIONS)
//   pig       0.9 × 0.9      simulation/entity/pig.ts        (PIG_DIMENSIONS)
//   cow       0.9 × 1.4      simulation/entity/cow.ts        (COW_DIMENSIONS)
//   tnt       0.98 × 0.98    simulation/entity/primedTnt.ts  (type "minecraft:tnt")
//   falling   0.98 × 0.98    simulation/entity/fallingBlock.ts
//   item      0.25 × 0.25    simulation/entity/itemEntity.ts
//   arrow     0.5 × 0.5      simulation/entity/arrow.ts
// All match vanilla EntityType.java (reference_code) `sized(width, height)`.
//
// CRITICAL: the box is always AXIS-ALIGNED and is built from the entity's position alone — it is NEVER
// rotated/tilted with the visual model. A minecart on a slope tilts its rendered body (xRot/yRot), but
// its collision AABB stays axis-aligned at the entity position, so this function takes no rotation input
// by design (matching the old renderer, which explicitly cancelled the model rotation on the outline).

import type { Vec3 } from "../../types";

export interface EntityAabbSize {
  /** Footprint side length (x and z). */
  width: number;
  /** Box height (y from feet up). */
  height: number;
}

/** Per-type AABB footprint, mirroring the sim's EntityDimensions (see file header). */
export function entityAabbSize(type: string): EntityAabbSize {
  const t = type.replace(/^minecraft:/, "");
  if (t === "item") return { width: 0.25, height: 0.25 };
  if (t === "tnt" || t === "primed_tnt") return { width: 0.98, height: 0.98 }; // sim type is "tnt"
  if (t === "falling_block") return { width: 0.98, height: 0.98 };
  if (t === "sheep") return { width: 0.9, height: 1.3 };
  if (t === "pig") return { width: 0.9, height: 0.9 };
  if (t === "cow") return { width: 0.9, height: 1.4 };
  if (t === "minecart" || t.endsWith("_minecart")) return { width: 0.98, height: 0.7 };
  if (t.endsWith("_boat") || t.endsWith("_raft")) return { width: 1.375, height: 0.5625 };
  if (t === "arrow" || t === "spectral_arrow" || t.endsWith("_arrow")) return { width: 0.5, height: 0.5 };
  // NOTE: item_frame / glow_item_frame are NOT square/feet-centred — entityAabb() handles them directly
  // (oriented thin box via itemFrameAabb); they never fall through to this square-footprint helper.
  return { width: 0.6, height: 1.8 }; // generic-mob fallback (vanilla Entity default footprint)
}

/**
 * The entity's axis-aligned bounding box from its FEET-CENTRE position — exactly the sim's
 * `EntityDimensions.makeBoundingBox` (x/z centred, y at feet, spanning to y+height). Never rotated.
 *
 * Item frames are the exception: a HangingEntity is NOT feet-centred and NOT a square footprint. Its box
 * is CENTRED on the entity position (which is itself the block-centre shifted toward the wall) and ORIENTED
 * by facing — thin (0.0625) along the facing axis, square on the other two: 0.75 normally, 1.0 for a framed
 * map (ItemFrame.createBoundingBox, ItemFrame.java:117-126). `props` carries the facing/map.
 */
export function entityAabb(type: string, position: Vec3, props?: Record<string, string>): { min: Vec3; max: Vec3 } {
  const t = type.replace(/^minecraft:/, "");
  if (t === "item_frame" || t === "glow_item_frame") return itemFrameAabb(position, props);
  const { width, height } = entityAabbSize(type);
  const hw = width / 2;
  const [x, y, z] = position;
  return { min: [x - hw, y, z - hw], max: [x + hw, y + height, z + hw] };
}

/** ItemFrame.createBoundingBox: centred on `position`, thin (0.0625) on the facing axis, 0.75/1.0 square on
 *  the other two. The entity position is already the block-centre shifted by -0.46875 toward the wall (the
 *  sim stores the vanilla recalculated pos), so the box is simply centred there — never rotated/tilted. */
function itemFrameAabb(position: Vec3, props?: Record<string, string>): { min: Vec3; max: Vec3 } {
  const facing = props?.facing ?? "south";
  const half = (props?.map === "true" ? 1.0 : 0.75) / 2; // WIDTH/HEIGHT 0.75 (1.0 framed map)
  const thin = 0.0625 / 2; // DEPTH 0.0625 along the facing axis
  const hx = facing === "east" || facing === "west" ? thin : half;
  const hy = facing === "up" || facing === "down" ? thin : half;
  const hz = facing === "north" || facing === "south" ? thin : half;
  const [x, y, z] = position;
  return { min: [x - hx, y - hy, z - hz], max: [x + hx, y + hy, z + hz] };
}

/** Outline colour for entity bounding boxes (golden — matches the old three.js renderer). 0xRRGGBB. */
export const ENTITY_BOUNDS_COLOR = 0xffd166;
