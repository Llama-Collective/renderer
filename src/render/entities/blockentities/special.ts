// Special (non-box) block entities. RENDERER_PLAN.md §18, Phase 4.5f/4.5g.
// These return `BESpecialDraw` descriptors the scene routes to dedicated renderers:
//   - embedded ITEMS (vault reward, brushable item, shelf/campfire held items) → the real item renderer
//     (`kind:"item"` → `ItemGeometry`: generated sprite or block model). The mob/trial spawner embeds the
//     spawned ENTITY, so it stays a block placeholder until the 4.6 entity-model path;
//   - beacon beam → the additive beam renderer; end portal/gateway → the portal renderer;
//   - structure / jigsaw / test blocks → a wireframe box.

import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";
import { dyeRgb } from "./dyes";
import { WOOD_TYPES, type BESpecialDraw, type BlockEntityDef } from "./registry";
import { mul, rotationX, rotationY, scaling, translation, toYRot, DEG } from "./transforms";

/** Direction.get2DDataValue(): south=0, west=1, north=2, east=3 (the horizontal-facing 2D index). */
const FACING_2D: Record<string, number> = { south: 0, west: 1, north: 2, east: 3 };

const num = (s: string | undefined, d: number): number => {
  const n = Number.parseFloat(s ?? "");
  return Number.isFinite(n) ? n : d;
};

/** A small spinning thing embedded at the block centre. `kind` = "item" (vault reward) or "blockModel"
 *  (spawner = a block placeholder for the spawned ENTITY, until the 4.6 entity-model path embeds it). */
function spinning(rec: BlockEntityRecord, id: string, clock: number, scale: number, y = 0.4, kind: "blockModel" | "item" = "blockModel"): BESpecialDraw {
  const model = mul(translation(rec.x + 0.5, rec.y + y, rec.z + 0.5), rotationY(clock), scaling(scale, scale, scale), translation(-0.5, -0.5, -0.5));
  return kind === "item" ? { kind: "item", item: id, model } : { kind: "blockModel", block: id, model };
}

// ── Embedded item (real item geometry — generated sprite or block model — via the item system) ──

export const SPAWNER: BlockEntityDef = {
  types: ["spawner", "trial_spawner"],
  textures: [],
  animated: true,
  // The spinning miniature is the spawned ENTITY; a block placeholder until the 4.6 entity-model path.
  special: (rec, clock) => [spinning(rec, "minecraft:slime_block", clock * 1.0, 0.42)],
};

export const VAULT: BlockEntityDef = {
  types: ["vault"],
  textures: [],
  animated: true,
  special: (rec, clock) => [spinning(rec, rec.props.item ?? "minecraft:diamond", clock * 0.8, 0.4, 0.4, "item")],
};

export const BRUSHABLE: BlockEntityDef = {
  types: ["suspicious_sand", "suspicious_gravel"],
  textures: [],
  animated: false,
  special: (rec) => {
    // Item slides out by `dusted` (0..3) along hit_direction (default up).
    const t = (num(rec.props.dusted, 0) / 3) * 0.5;
    const model = mul(translation(rec.x + 0.5, rec.y + 0.3 + t, rec.z + 0.5), scaling(0.4, 0.4, 0.4), translation(-0.5, -0.5, -0.5));
    return [{ kind: "item", item: rec.props.item ?? "minecraft:emerald", model }];
  },
};

export const SHELF: BlockEntityDef = {
  // Shelf uses WOOD_TYPES order-independently (no textures; the type list is registry-keyed) — the shared
  // list replaces the local copy that diverged only in order. See registry.WOOD_TYPES.
  types: WOOD_TYPES.map((w) => `${w}_shelf`),
  textures: [],
  animated: false,
  // Vanilla ShelfBlockEntity holds 3 items (NonNullList.withSize(3)). ShelfRenderer: yRot = -facing.toYRot,
  // slot X = (slot-1)*0.3125, Z = -0.25, scale 0.25. Only filled slots render — but at their OWN slot
  // index, so the `items` prop is comma-split WITHOUT dropping empties ("" = an empty slot keeps slot 1
  // empty while slots 0 and 2 still draw in place); itemList would collapse the gaps and misplace items.
  special: (rec) => {
    const slots = (rec.props.items ?? "").split(",");
    const yaw = -toYRot(rec.props.facing) * DEG;
    return slots.slice(0, 3).flatMap((raw, i) => {
      const item = raw.trim();
      if (!item) return [];
      return [{
        kind: "item" as const,
        item,
        model: mul(translation(rec.x + 0.5, rec.y + 0.5, rec.z + 0.5), rotationY(yaw), translation((i - 1) * 0.3125, 0, -0.25), scaling(0.25, 0.25, 0.25), translation(-0.5, -0.5, -0.5)),
      }];
    });
  },
};

export const CAMPFIRE: BlockEntityDef = {
  types: ["campfire", "soul_campfire"],
  textures: [],
  animated: false,
  // Cooking items only when the campfire holds food (vanilla); an empty campfire is just the (animated)
  // block model — rendering placeholders unconditionally clutters + z-fights the logs.
  special: (rec) => {
    // Slot-indexed: comma-split WITHOUT dropping empties so each item keeps its OWN slot (the per-slot
    // rotation below depends on the index), exactly as the shelf does.
    const slots = (rec.props.items ?? "").split(",");
    const facing2D = FACING_2D[rec.props.facing ?? "north"] ?? 0;
    // Vanilla CampfireRenderer: translate(0.5, 0.44921875, 0.5) · rotateY(-dir.toYRot()) · rotateX(90) ·
    // translate(-0.3125, -0.3125, 0) · scale(0.375), where dir = from2DDataValue((slot+facing2D)%4) and
    // from2DDataValue(n).toYRot() = n·90. The rotateX(90) lays each item FLAT on the fire (AUDIT M6).
    // Trailing translate(-0.5) centres this renderer's [0,1]³ item geometry, as the other item draws do.
    return slots.slice(0, 4).flatMap((raw, i) => {
      const item = raw.trim();
      if (!item) return [];
      const yawDeg = -(((i + facing2D) % 4) * 90);
      return [{
        kind: "item" as const,
        item,
        model: mul(
          translation(rec.x + 0.5, rec.y + 0.44921875, rec.z + 0.5),
          rotationY(yawDeg * DEG),
          rotationX(Math.PI / 2),
          translation(-0.3125, -0.3125, 0),
          scaling(0.375, 0.375, 0.375),
          translation(-0.5, -0.5, -0.5),
        ),
      }];
    });
  },
};

// ── Beacon beam ──────────────────────────────────────────────────────────────

export const BEACON: BlockEntityDef = {
  types: ["beacon"],
  textures: ["entity/beacon_beam"],
  animated: true,
  // The beam shows only when the beacon is ACTIVATED (a base pyramid) and its column is clear — the app
  // computes that from the world (`app/beaconBeam.ts`) and sets `props.active`/`beamColor`/`height`. A beacon
  // with `active==="false"` draws no beam (vanilla `getBeamSections()` is empty when `levels==0`). When
  // `active` is absent (e.g. a standalone harness with no world scan) the beam shows, as before. `beamColor`
  // is a glass-tinted 0xRRGGBB hex; absent ⇒ fall back to the `color` dye name.
  special: (rec) => {
    if (rec.props.active === "false") return [];
    const color = rec.props.beamColor !== undefined ? parseInt(rec.props.beamColor, 16) : dyeRgb(rec.props.color);
    return [{ kind: "beam", pos: [rec.x, rec.y + 1, rec.z], color, height: num(rec.props.height, 24) }];
  },
};

// ── End portal / gateway ──────────────────────────────────────────────────────

export const END_PORTAL: BlockEntityDef = {
  types: ["end_portal", "end_gateway"],
  textures: [],
  animated: true,
  special: (rec) => [{ kind: "portal", pos: [rec.x, rec.y, rec.z], gateway: rec.type.includes("gateway") }],
};

// ── Wireframe-box dev blocks ──────────────────────────────────────────────────

export const BOUNDING_BOX: BlockEntityDef = {
  types: ["structure_block", "jigsaw", "test_block", "test_instance_block"],
  textures: [],
  animated: false,
  special: (rec) => {
    const sx = num(rec.props.sizeX, 1), sy = num(rec.props.sizeY, 1), sz = num(rec.props.sizeZ, 1);
    return [{ kind: "lineBox", min: [rec.x, rec.y, rec.z], max: [rec.x + sx, rec.y + sy, rec.z + sz], color: 0xe6e6e6 }];
  },
};

export const SPECIAL_DEFS = [SPAWNER, VAULT, BRUSHABLE, SHELF, CAMPFIRE, BEACON, END_PORTAL, BOUNDING_BOX];
