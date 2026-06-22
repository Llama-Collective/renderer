// Chest family (single / double / copper / ender / trapped). RENDERER_PLAN.md §18, Phase 4.5c.
// Vanilla ChestModel + ChestRenderer: bottom + lid + lock; per-FACING rotation about block centre;
// lid+lock open with the cubic ease 1-(1-o)^3 → xRot = -ease·π/2 (ChestModel.setupAnim).

import type { ModelPartDef } from "../../../mesh/entity/ModelPart";
import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";
import { Direction } from "../../../types";
import { boxBE, strip, type BlockEntityDef } from "./registry";
import { facingRot } from "./transforms";

const HINGE = Math.PI / 2;

// Double-chest seam-face exclusion (vanilla ChestModel.createDoubleBodyLeft/RightLayer): each half's box
// spans 15px and OMITS the interior X-face where the two halves meet (LEFT drops WEST/−X, RIGHT drops
// EAST/+X) so the join is seamless. `undefined` faces = a full 6-face box (single chest).
const NO_WEST = [Direction.Down, Direction.Up, Direction.North, Direction.South, Direction.East];
const NO_EAST = [Direction.Down, Direction.Up, Direction.North, Direction.South, Direction.West];

/** Chest parts (lid group hinged at z=1, y=9). `bodyX`/`width` = body x-origin/span; lock at `lockX` (width
 *  `lockW`); `faces` (if set) is the per-cube visible-face allow-list that drops the double-chest seam face. */
function chestParts(bodyX: number, width: number, lockX: number, lockW: number, faces?: Direction[]): ModelPartDef[] {
  return [
    { name: "bottom", pivot: [0, 0, 0], cubes: [{ origin: [bodyX, 0, 1], size: [width, 10, 14], uv: [0, 19], faces }] },
    { name: "lid", pivot: [0, 9, 1], cubes: [{ origin: [bodyX, 0, 0], size: [width, 5, 14], uv: [0, 0], faces }] },
    { name: "lock", pivot: [0, 9, 1], cubes: [{ origin: [lockX, -2, 14], size: [lockW, 4, 1], uv: [0, 0], faces }] },
  ];
}

const SINGLE = chestParts(1, 14, 7, 2);
/** Single-chest part tree ([0,1] block space, lid closed at rest) — reused by the chest minecart, which
 *  has no terrain block model and must render this BE box model inside the cart. */
export const SINGLE_CHEST_PARTS = SINGLE;
const LEFT = chestParts(0, 15, 0, 1, NO_WEST);
const RIGHT = chestParts(1, 15, 15, 1, NO_EAST);

const VARIANT: Record<string, string> = {
  chest: "normal",
  trapped_chest: "trapped",
  ender_chest: "ender",
  copper_chest: "copper",
  exposed_copper_chest: "copper_exposed",
  weathered_copper_chest: "copper_weathered",
  oxidized_copper_chest: "copper_oxidized",
};
const HAS_DOUBLE = new Set(["normal", "trapped", "copper", "copper_exposed", "copper_weathered", "copper_oxidized"]);

function baseVariant(rec: BlockEntityRecord): string {
  const t = strip(rec.type).replace(/^waxed_/, "");
  return VARIANT[t] ?? "normal";
}

function chestTexture(rec: BlockEntityRecord): string {
  const v = baseVariant(rec);
  const ct = rec.props.type;
  const suffix = (ct === "left" || ct === "right") && HAS_DOUBLE.has(v) ? `_${ct}` : "";
  return `entity/chest/${v}${suffix}`;
}

function selectParts(rec: BlockEntityRecord): ModelPartDef[] {
  return rec.props.type === "left" ? LEFT : rec.props.type === "right" ? RIGHT : SINGLE;
}

// Vanilla ChestRenderer: openNess = blockEntity.getOpenNess(partialTick); the lid lifts with the cubic ease
// xRot = -(1-(1-o)³)·π/2 (ChestModel.setupAnim). `o` is the container's lid openness ∈ [0,1] — driven by the
// open-viewer count (the editor's container GUI), integrated to a smooth value by the baker. 0 = lid shut.
function lidPose(openness: number): Record<string, { rotation: [number, number, number] }> {
  const o = openness <= 0 ? 0 : openness >= 1 ? 1 : openness;
  const eased = 1 - (1 - o) * (1 - o) * (1 - o);
  const x = -eased * HINGE;
  return { lid: { rotation: [x, 0, 0] }, lock: { rotation: [x, 0, 0] } };
}

export const CHEST: BlockEntityDef = boxBE({
  types: [
    "chest", "trapped_chest", "ender_chest",
    "copper_chest", "exposed_copper_chest", "weathered_copper_chest", "oxidized_copper_chest",
    "waxed_copper_chest", "waxed_exposed_copper_chest", "waxed_weathered_copper_chest", "waxed_oxidized_copper_chest",
  ],
  texWidth: 64,
  texHeight: 64,
  parts: selectParts,
  textures: [
    "entity/chest/normal", "entity/chest/normal_left", "entity/chest/normal_right",
    "entity/chest/trapped", "entity/chest/trapped_left", "entity/chest/trapped_right",
    "entity/chest/ender",
    "entity/chest/copper", "entity/chest/copper_left", "entity/chest/copper_right",
    "entity/chest/copper_exposed", "entity/chest/copper_weathered", "entity/chest/copper_oxidized",
  ],
  texture: chestTexture,
  transform: (rec) => facingRot(rec.props.facing, 0),
  // OPENABLE: the lid lift is driven by the container's openness (open-viewer count), not a free clock loop.
  // The baker integrates `ctx.openness` toward the BE's `open` target and keeps a closed chest in the static
  // section bake (lid shut, BBB-cullable); an opening/open/closing one draws per-frame.
  openable: true,
  animate: (_rec, _clock, ctx) => lidPose(ctx.openness ?? 0),
});
