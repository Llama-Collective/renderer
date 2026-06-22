// Box-model entities (the ModelPart path). RENDERER_PLAN.md §18, Phase 4.5b.
//
// Exact vanilla `ModelPart` definitions (cubes in model pixels, pivots, ZYX rotation) for the sim's
// box-model entities, + a `BoxModelSource` that bakes geometry and returns per-entity draw LAYERS:
// sheep = body layer (untinted) + wool overlay layer (dye-tinted via uniform — bake-once, recolour by
// uniform, §18). A layer that re-poses per frame (walk swing) is flagged `dynamic`, so the renderer
// re-encodes it into a reused buffer (TRAP 18.C) rather than churning GPU buffers. Base transform =
// vanilla scale(-1,-1,1)·translate(0,-1.501,0) (model space Y-down → world Y-up, feet at origin).

import type { GpuTextureHandle } from "../../core/GraphicsDevice";
import { bakeModel, bakeModelInto, type BakeOptions, type ModelPartDef, type PartPose } from "../../mesh/entity/ModelPart";
import { DEG, Mat4Frame, identity, mul, rotationY, scaling, translation, type Mat4 } from "../../mesh/entity/mat4";
import { unpackRgb } from "../color";
import { SINGLE_CHEST_PARTS } from "./blockentities/chest";
import { TerrainPass, type SpriteUv } from "../../types";
import type { RawVertex } from "../../mesh/VertexFormat";
import type { EntityVertsByPass } from "./blockDisplay";
import type { BoxModelLayer, BoxModelSource } from "./EntityModelFactory";
import type { RenderEntity } from "./EntityScene";
import { quadrupedMesh, quadrupedWalk } from "./entitymodels/quadruped";

/** A complete box model: texture key + logical size + model→entity-local base + part roots. */
export interface EntityModelDef {
  texture: string;
  texWidth: number;
  texHeight: number;
  base: Mat4;
  roots: ModelPartDef[];
}

/** Vanilla mob base transform: flip + lift feet to y=0. */
export const MOB_BASE: Mat4 = mul(scaling(-1, -1, 1), translation(0, -1.501, 0));

// ── Sheep (QuadrupedModel base + SheepFurModel wool layer) ──────────────────────

function leg(name: string, px: number, pz: number, len: number, uv: [number, number], inflate?: number): ModelPartDef {
  return { name, pivot: [px, 12, pz], cubes: [{ origin: [-2, 0, -2], size: [4, len, 4], uv, inflate }] };
}

export const SHEEP_BASE: EntityModelDef = {
  texture: "entity/sheep/sheep",
  texWidth: 64,
  texHeight: 32,
  base: MOB_BASE,
  roots: [
    { name: "head", pivot: [0, 6, -8], cubes: [{ origin: [-3, -4, -6], size: [6, 6, 8], uv: [0, 0] }] },
    { name: "body", pivot: [0, 5, 2], rotation: [Math.PI / 2, 0, 0], cubes: [{ origin: [-4, -10, -7], size: [8, 16, 6], uv: [28, 8] }] },
    leg("right_hind_leg", -3, 7, 12, [0, 16]),
    leg("left_hind_leg", 3, 7, 12, [0, 16]),
    leg("right_front_leg", -3, -5, 12, [0, 16]),
    leg("left_front_leg", 3, -5, 12, [0, 16]),
  ],
};

export const SHEEP_WOOL: EntityModelDef = {
  texture: "entity/sheep/sheep_wool",
  texWidth: 64,
  texHeight: 32,
  base: MOB_BASE,
  roots: [
    { name: "head", pivot: [0, 6, -8], cubes: [{ origin: [-3, -4, -4], size: [6, 6, 6], uv: [0, 0], inflate: 0.6 }] },
    { name: "body", pivot: [0, 5, 2], rotation: [Math.PI / 2, 0, 0], cubes: [{ origin: [-4, -10, -7], size: [8, 16, 6], uv: [28, 8], inflate: 1.75 }] },
    leg("right_hind_leg", -3, 7, 6, [0, 16], 0.5),
    leg("left_hind_leg", 3, 7, 6, [0, 16], 0.5),
    leg("right_front_leg", -3, -5, 6, [0, 16], 0.5),
    leg("left_front_leg", 3, -5, 6, [0, 16], 0.5),
  ],
};

// ── Pig (generic QuadrupedModel + snout) ────────────────────────────────────────
export const PIG: EntityModelDef = {
  texture: "entity/pig/pig_temperate",
  texWidth: 64,
  texHeight: 64,
  base: MOB_BASE,
  roots: quadrupedMesh({
    legSize: 6,
    mirrorLeftLeg: true,
    head: { name: "head", pivot: [0, 12, -6], cubes: [
      { origin: [-4, -4, -8], size: [8, 8, 8], uv: [0, 0] },
      { origin: [-2, 0, -9], size: [4, 3, 1], uv: [16, 16] }, // snout
    ] },
    body: { name: "body", pivot: [0, 11, 2], rotation: [Math.PI / 2, 0, 0], cubes: [{ origin: [-5, -10, -7], size: [10, 16, 8], uv: [28, 8] }] },
  }),
};

// ── Cow (custom body + horns + udder) ───────────────────────────────────────────
export const COW: EntityModelDef = {
  texture: "entity/cow/cow_temperate",
  texWidth: 64,
  texHeight: 64,
  base: MOB_BASE,
  roots: quadrupedMesh({
    legSize: 12,
    legX: 4,
    mirrorLeftLeg: true,
    head: { name: "head", pivot: [0, 4, -8], cubes: [
      { origin: [-4, -4, -6], size: [8, 8, 6], uv: [0, 0] },
      { origin: [-3, 1, -7], size: [6, 3, 1], uv: [1, 33] }, // snout
      { origin: [-5, -5, -5], size: [1, 3, 1], uv: [22, 0] }, // right horn
      { origin: [4, -5, -5], size: [1, 3, 1], uv: [22, 0] }, // left horn
    ] },
    body: { name: "body", pivot: [0, 5, 2], rotation: [Math.PI / 2, 0, 0], cubes: [
      { origin: [-6, -10, -7], size: [12, 18, 10], uv: [18, 4] },
      { origin: [-2, 2, -8], size: [4, 6, 1], uv: [52, 0] }, // udder
    ] },
  }),
};

// ── Minecart (MinecartModel.createBodyLayer) ────────────────────────────────────

const SIDE = (uv: [number, number] = [0, 0]) => ({ origin: [-8, -9, -1] as [number, number, number], size: [16, 8, 2] as [number, number, number], uv });

export const MINECART_BODY: EntityModelDef = {
  texture: "entity/minecart/minecart",
  texWidth: 64,
  texHeight: 32,
  // Just the model-space flip; the +0.375 lift now lives in `cartModelMatrix` so the rail-slope PITCH
  // (Axis.ZP) can be inserted between yaw and flip in vanilla's exact order (the lift commutes with the
  // Y-yaw but NOT with the Z-pitch, so it must be the outermost transform, not folded into `base`).
  base: scaling(-1, -1, 1),
  roots: [
    { name: "bottom", pivot: [0, 4, 0], rotation: [Math.PI / 2, 0, 0], cubes: [{ origin: [-10, -8, -1], size: [20, 16, 2], uv: [0, 10] }] },
    { name: "front", pivot: [-9, 4, 0], rotation: [0, Math.PI * 1.5, 0], cubes: [SIDE()] },
    { name: "back", pivot: [9, 4, 0], rotation: [0, Math.PI / 2, 0], cubes: [SIDE()] },
    { name: "left", pivot: [0, 4, -7], rotation: [0, Math.PI, 0], cubes: [SIDE()] },
    { name: "right", pivot: [0, 4, 7], rotation: [0, 0, 0], cubes: [SIDE()] },
  ],
};

// The block a chest minecart carries has NO terrain block model (a chest is a block entity), so the cart
// renders this BE single-chest box model inside its frame — the parts are already in [0,1] block space
// (lid closed at rest), so the same inner placement the terrain-baked contents use positions it correctly.
export const MINECART_CHEST: EntityModelDef = {
  texture: "entity/chest/normal",
  texWidth: 64,
  texHeight: 64,
  base: identity(),
  roots: SINGLE_CHEST_PARTS,
};

// ── Boats (BoatModel / RaftModel + chest variants) ──────────────────────────────
//
// Vanilla AbstractBoatRenderer base transform (BoatModel parts in model pixels, 128-wide texture):
//   translate(0, 0.375, 0) · Ry(180−yRot) · [hurt/bubble wobble — skipped] · scale(-1,-1,1) · Ry(90°)
// The +0.375 lift commutes with the Y-yaw, so it's folded into `base` (the boat carries NO Z-pitch, unlike
// the minecart). The sim serializes no boat yaw, so `yRot` defaults to 0 — every boat faces the default
// orientation, matching the OLD renderer (which rendered boats with identity rotation). Geometry is
// vanilla-exact (BoatModel.java / RaftModel.java), so it orients correctly under this base.

const BOAT_BASE: Mat4 = mul(translation(0, 0.375, 0), scaling(-1, -1, 1), rotationY(Math.PI / 2));

/** Vanilla BoatModel.createChildren: hull (bottom + 4 walls) + 2 paddles, at rest pose. */
function boatHullParts(): ModelPartDef[] {
  return [
    { name: "bottom", pivot: [0, 3, 1], rotation: [Math.PI / 2, 0, 0], cubes: [{ origin: [-14, -9, -3], size: [28, 16, 3], uv: [0, 0] }] },
    { name: "back", pivot: [-15, 4, 4], rotation: [0, Math.PI * 1.5, 0], cubes: [{ origin: [-13, -7, -1], size: [18, 6, 2], uv: [0, 19] }] },
    { name: "front", pivot: [15, 4, 0], rotation: [0, Math.PI / 2, 0], cubes: [{ origin: [-8, -7, -1], size: [16, 6, 2], uv: [0, 27] }] },
    { name: "right", pivot: [0, 4, -9], rotation: [0, Math.PI, 0], cubes: [{ origin: [-14, -7, -1], size: [28, 6, 2], uv: [0, 35] }] },
    { name: "left", pivot: [0, 4, 9], cubes: [{ origin: [-14, -7, -1], size: [28, 6, 2], uv: [0, 43] }] },
    { name: "left_paddle", pivot: [3, -5, 9], rotation: [0, 0, 0.19634955], cubes: [
      { origin: [-1, 0, -5], size: [2, 2, 18], uv: [62, 0] },
      { origin: [-1.001, -3, 8], size: [1, 6, 7], uv: [62, 0] },
    ] },
    { name: "right_paddle", pivot: [3, -5, -9], rotation: [0, Math.PI, 0.19634955], cubes: [
      { origin: [-1, 0, -5], size: [2, 2, 18], uv: [62, 20] },
      { origin: [0.001, -3, 8], size: [1, 6, 7], uv: [62, 20] },
    ] },
  ];
}

/** Vanilla RaftModel.createChildren (bamboo): a flat double-slab hull + 2 paddles. */
function raftParts(): ModelPartDef[] {
  return [
    { name: "bottom", pivot: [0, -2.1, 1], rotation: [Math.PI / 2, 0, 0], cubes: [
      { origin: [-14, -11, -4], size: [28, 20, 4], uv: [0, 0] },
      { origin: [-14, -9, -8], size: [28, 16, 4], uv: [0, 0] },
    ] },
    { name: "left_paddle", pivot: [3, -4, 9], rotation: [0, 0, 0.19634955], cubes: [
      { origin: [-1, 0, -5], size: [2, 2, 18], uv: [0, 24] },
      { origin: [-1.001, -3, 8], size: [1, 6, 7], uv: [0, 24] },
    ] },
    { name: "right_paddle", pivot: [3, -4, -9], rotation: [0, Math.PI, 0.19634955], cubes: [
      { origin: [-1, 0, -5], size: [2, 2, 18], uv: [40, 24] },
      { origin: [0.001, -3, 8], size: [1, 6, 7], uv: [40, 24] },
    ] },
  ];
}

/** Vanilla ChestBoatModel / ChestRaftModel chest parts. `dy` shifts the chest down for the lower raft deck. */
function chestParts(dy: number): ModelPartDef[] {
  return [
    { name: "chest_bottom", pivot: [-2, -5 + dy, -6], rotation: [0, -Math.PI / 2, 0], cubes: [{ origin: [0, 0, 0], size: [12, 8, 12], uv: [0, 76] }] },
    { name: "chest_lid", pivot: [-2, -9 + dy, -6], rotation: [0, -Math.PI / 2, 0], cubes: [{ origin: [0, 0, 0], size: [12, 4, 12], uv: [0, 59] }] },
    { name: "chest_lock", pivot: [-1, -6 + dy, -1], rotation: [0, -Math.PI / 2, 0], cubes: [{ origin: [0, 0, 0], size: [2, 4, 1], uv: [0, 59] }] },
  ];
}

/** Build a boat/raft EntityModelDef. Chest variants use a 128-tall texture (boat hull in the top 64px). */
function boatDef(texture: string, raft: boolean, chest: boolean): EntityModelDef {
  const hull = raft ? raftParts() : boatHullParts();
  const roots = chest ? [...hull, ...chestParts(raft ? -5.1 : 0)] : hull;
  return { texture, texWidth: 128, texHeight: chest ? 128 : 64, base: BOAT_BASE, roots };
}

/** Wood types that render with BoatModel (bamboo is a raft, handled separately). */
const BOAT_WOODS = ["oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry", "pale_oak"];

/** Stripped entity type → boat model def (e.g. "oak_boat", "spruce_chest_boat", "bamboo_raft"). */
const BOATS: Record<string, EntityModelDef> = (() => {
  const m: Record<string, EntityModelDef> = {};
  for (const w of BOAT_WOODS) {
    m[`${w}_boat`] = boatDef(`entity/boat/${w}`, false, false);
    m[`${w}_chest_boat`] = boatDef(`entity/chest_boat/${w}`, false, true);
  }
  m["bamboo_raft"] = boatDef("entity/boat/bamboo", true, false);
  m["bamboo_chest_raft"] = boatDef("entity/chest_boat/bamboo", true, true);
  return m;
})();

// ── Sheep wool dye colours (DyeColor → 0xRRGGBB) ────────────────────────────────

const DYE_RGB: Record<string, number> = {
  white: 0xe9ecec, orange: 0xf07613, magenta: 0xbd44b3, light_blue: 0x3aafd9,
  yellow: 0xf8c627, lime: 0x70b919, pink: 0xed8dac, gray: 0x3e4447,
  light_gray: 0x8e8e86, cyan: 0x158991, purple: 0x792aac, blue: 0x35399d,
  brown: 0x724728, green: 0x546d1b, red: 0xa12722, black: 0x141519,
};

function dyeTint(color: string | undefined): readonly [number, number, number, number] {
  const rgb = DYE_RGB[color ?? "white"] ?? DYE_RGB.white;
  const [r, g, b] = unpackRgb(rgb);
  return [r, g, b, 1];
}

// ── BoxModelSource ──────────────────────────────────────────────────────────────

/** Provides per-texture atlas rects so baked [0,1] UVs remap into the entity atlas. */
export interface AtlasRects {
  uvFor(name: string): SpriteUv;
  has(name: string): boolean;
}

/** Per-entity world placement from interpolated position + body yaw (degrees). Built into the per-frame
 *  arena `m` (A3) — the result is consumed by EntityRenderer this same frame. */
export function entityModelMatrix(e: RenderEntity, yawDeg: number, m: Mat4Frame): Mat4 {
  const p = e.position;
  return m.mul(m.Tw(p[0], p[1], p[2]), m.Ry((180 - yawDeg) * DEG)); // M6: `Tw` anchors the world placement in double
}

/**
 * Minecart placement with rail-slope PITCH, matching vanilla AbstractMinecartRenderer's order exactly:
 *   T(pos) · T(0,0.375,0) · Ry(180−yaw) · Zp(−pitch)   [· scale(-1,-1,1) lives in MINECART_BODY.base]
 * The +0.375 lift is the OUTERMOST transform (it commutes with the yaw but not the pitch). Yaw prefers the
 * rail-tangent `yRotRail` (correct for parked carts) over the movement-derived `yRot`; pitch is `xRot`.
 */
export function cartModelMatrix(e: RenderEntity, yawDeg: number, pitchDeg: number, m: Mat4Frame): Mat4 {
  const p = e.position;
  return m.mul(
    m.Tw(p[0], p[1], p[2]), // M6: outermost world placement anchored in double
    m.T(0, 0.375, 0),
    m.Ry((180 - yawDeg) * DEG),
    m.Rz(-pitchDeg * DEG),
  );
}

/** Yaw (deg) for a minecart: rail-tangent if present (parked/slope), else movement-derived. */
export function cartYaw(e: RenderEntity): number {
  return num(e.properties.yRotRail ?? e.properties.yRot, 0);
}

/** Pitch (deg) for a minecart: rail-slope tilt, 0 on flat rails. */
export function cartPitch(e: RenderEntity): number {
  return num(e.properties.xRot, 0);
}

function cutout(verts: RawVertex[]): EntityVertsByPass {
  return { [TerrainPass.Cutout]: verts };
}

// ── Mob registry (name → texture layers + animator) ─────────────────────────────
// Each mob is one or more texture LAYERS (base + optional tinted overlay) over a shared family mesh;
// adding a quadruped is now a registry entry, not a new `if` branch (ENTITY_RENDERING_FINDINGS.md).
type MobTint = (e: RenderEntity) => readonly [number, number, number, number];
interface MobDef {
  layers: { def: EntityModelDef; tint?: MobTint }[];
  walk?: (e: RenderEntity, clock: number) => Record<string, PartPose>;
}

const MOBS: Record<string, MobDef> = {
  sheep: { layers: [{ def: SHEEP_BASE }, { def: SHEEP_WOOL, tint: (e) => dyeTint(e.properties.color) }], walk: quadrupedWalk },
  pig: { layers: [{ def: PIG }], walk: quadrupedWalk },
  cow: { layers: [{ def: COW }], walk: quadrupedWalk },
};

/** Every entity-model texture the box-model engine may stitch (mobs + minecart + boats) — for the atlas. */
export function entityModelTextures(): string[] {
  const mob = Object.values(MOBS).flatMap((m) => m.layers.map((l) => l.def.texture));
  const boats = Object.values(BOATS).map((d) => d.texture);
  return [...new Set([...mob, MINECART_BODY.texture, ...boats])];
}

/** The box-model engine: dispatches registered mobs (sheep/pig/cow/…) + minecart. `atlas` = entity rects. */
export function createBoxModelSource(atlas: AtlasRects, texture: GpuTextureHandle): BoxModelSource {
  const bakeOpts = (def: EntityModelDef, poses?: Record<string, PartPose>): BakeOptions =>
    ({ texWidth: def.texWidth, texHeight: def.texHeight, uvRect: atlas.uvFor(def.texture), base: def.base, poses, pass: TerrainPass.Cutout, shade: true });
  const bake = (def: EntityModelDef, poses?: Record<string, PartPose>): RawVertex[] => bakeModel(def.roots, bakeOpts(def, poses));

  return {
    texture,
    build(e, clock, mat): BoxModelLayer[] | null {
      const type = e.type.replace(/^minecraft:/, "");
      if (type === "minecart" || type.endsWith("_minecart")) {
        if (!atlas.has(MINECART_BODY.texture)) return null;
        const frame = cartModelMatrix(e, cartYaw(e), cartPitch(e), mat);
        const layers: BoxModelLayer[] = [{ geomKey: "minecart:body", verts: () => cutout(bake(MINECART_BODY)), model: frame }];
        // Chest cart: render the BE chest box model (no terrain model exists) in the cart, using the SAME
        // inner placement the terrain-baked contents use — chest display offset 8 → no extra Y shift.
        if (type === "chest_minecart" && atlas.has(MINECART_CHEST.texture)) {
          const inner = mat.mul(frame, mat.S(0.75, 0.75, 0.75), mat.T(-0.5, 0, 0.5), mat.Ry(Math.PI / 2));
          layers.push({ geomKey: "minecart:chest", verts: () => cutout(bake(MINECART_CHEST)), model: inner });
        }
        return layers;
      }
      const boat = BOATS[type];
      if (boat) {
        if (!atlas.has(boat.texture)) return null;
        // Boats carry no sim yaw → entityModelMatrix(e, 0); BOAT_BASE supplies the constant flip/turn/lift.
        return [{ geomKey: `boat:${type}`, verts: () => cutout(bake(boat)), model: entityModelMatrix(e, num(e.properties.yRot, 0), mat) }];
      }
      const mob = MOBS[type];
      if (mob) {
        const layers = mob.layers.filter((l) => atlas.has(l.def.texture));
        if (layers.length === 0) return null;
        const yaw = num(e.properties.yBodyRot ?? e.properties.yRot, 0);
        const poses = mob.walk?.(e, clock) ?? {};
        const dynamic = Object.keys(poses).length > 0;
        const model = entityModelMatrix(e, yaw, mat);
        const sfx = dynamic ? `#${e.id}` : ":rest";
        // Dynamic (walking) layers bake straight into the packed sink — no per-frame RawVertex[] (A4).
        return layers.map((layer, i): BoxModelLayer => ({
          geomKey: `${type}:${i}${sfx}`,
          verts: () => cutout(bake(layer.def, poses)),
          bakeInto: dynamic ? (sink) => bakeModelInto(layer.def.roots, bakeOpts(layer.def, poses), sink) : undefined,
          model, tint: layer.tint?.(e), dynamic,
        }));
      }
      return null;
    },
  };
}

function num(s: string | undefined, dflt: number): number {
  const n = Number.parseFloat(s ?? "");
  return Number.isFinite(n) ? n : dflt;
}
