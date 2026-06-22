// CPU picking for the editor: a screen ray vs entity AABBs, per-block collision shapes (outline boxes,
// so thin blocks like string/dust are clickable across their footprint), and the ground plane.
// Reimplements the old three.js Raycaster path (EditableWorldRenderer.pick) without three.js.
// RENDERER_INTEGRATION_PLAN Step 5.

import type { Vec3 } from "../types";
import { AIR } from "./WorldModel";
import type { EditableBlock, EditableEntity, PickResult, Vec3Tuple } from "../../../schematic-io/types";

export interface Ray {
  origin: Vec3;
  /** Normalized direction. */
  dir: Vec3;
}

/** A collision box for a block, as `[minX, minY, minZ, maxX, maxY, maxZ]` (block-local 0..1). */
export type OutlineBox = readonly [number, number, number, number, number, number];

const UNIT: OutlineBox[] = [[0, 0, 0, 1, 1, 1]];

export interface PickInputs {
  ray: Ray;
  /** Packed block id at world coords; 0 = air. */
  getBlock: (x: number, y: number, z: number) => number;
  /** Palette lookup for the hit block (name + properties). */
  blockAt: (x: number, y: number, z: number) => EditableBlock | null;
  /** Per-block collision shape; null ⇒ unit cube. */
  outlineProvider: ((x: number, y: number, z: number) => OutlineBox[] | null) | null;
  /** Current entities (for AABB picking). */
  entities: readonly EditableEntity[];
  /** Entity collision box size [w,h,d] (feet on its y). */
  boundsOf: (e: EditableEntity) => Vec3Tuple;
  maxDist?: number;
}

/** Ray vs AABB (slab method); returns entry distance + outward face normal, or null. */
function rayAabb(o: Vec3, d: Vec3, lo: Vec3, hi: Vec3): { t: number; n: [number, number, number] } | null {
  let tmin = 0;
  let tmax = Infinity;
  let axis = -1;
  for (let a = 0; a < 3; a++) {
    const oa = o[a];
    const da = d[a];
    if (Math.abs(da) < 1e-9) {
      if (oa < lo[a] || oa > hi[a]) return null;
      continue;
    }
    const inv = 1 / da;
    let t1 = (lo[a] - oa) * inv;
    let t2 = (hi[a] - oa) * inv;
    if (t1 > t2) {
      const tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    if (t1 > tmin) {
      tmin = t1;
      axis = a;
    }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return null;
  }
  if (axis < 0) return null; // origin inside the box → no face entry
  const n: [number, number, number] = [0, 0, 0];
  n[axis] = d[axis] > 0 ? -1 : 1; // outward normal of the entered face
  return { t: tmin, n };
}

export function pickWorld(inp: PickInputs): PickResult | null {
  const o = inp.ray.origin;
  const d = inp.ray.dir;
  const maxDist = inp.maxDist ?? 320;

  // ── Entities: nearest AABB the ray enters ──────────────────────────────────
  let nearestEntity: EditableEntity | null = null;
  let entityDist = Infinity;
  for (const e of inp.entities) {
    const [bw, bh, bd] = inp.boundsOf(e);
    const [ex, ey, ez] = e.position;
    const hit = rayAabb(o, d, [ex - bw / 2, ey, ez - bd / 2], [ex + bw / 2, ey + bh, ez + bd / 2]);
    if (hit && hit.t < entityDist) {
      entityDist = hit.t;
      nearestEntity = e;
    }
  }

  // ── Blocks: voxel DDA over the model, testing each cell's collision shape ───
  let best = Infinity;
  let blockResult: { x: number; y: number; z: number; n: [number, number, number] } | null = null;
  let x = Math.floor(o[0]);
  let y = Math.floor(o[1]);
  let z = Math.floor(o[2]);
  const stepX = d[0] > 0 ? 1 : -1;
  const stepY = d[1] > 0 ? 1 : -1;
  const stepZ = d[2] > 0 ? 1 : -1;
  const tDeltaX = d[0] !== 0 ? Math.abs(1 / d[0]) : Infinity;
  const tDeltaY = d[1] !== 0 ? Math.abs(1 / d[1]) : Infinity;
  const tDeltaZ = d[2] !== 0 ? Math.abs(1 / d[2]) : Infinity;
  let tMaxX = d[0] !== 0 ? (d[0] > 0 ? x + 1 - o[0] : o[0] - x) * tDeltaX : Infinity;
  let tMaxY = d[1] !== 0 ? (d[1] > 0 ? y + 1 - o[1] : o[1] - y) * tDeltaY : Infinity;
  let tMaxZ = d[2] !== 0 ? (d[2] > 0 ? z + 1 - o[2] : o[2] - z) * tDeltaZ : Infinity;
  let t = 0;
  let guard = 0;
  while (t <= maxDist && guard++ < 200000) {
    if (inp.getBlock(x, y, z) !== AIR) {
      const boxes = inp.outlineProvider?.(x, y, z) ?? UNIT;
      for (const b of boxes) {
        const hit = rayAabb(o, d, [x + b[0], y + b[1], z + b[2]], [x + b[3], y + b[4], z + b[5]]);
        if (hit && hit.t < best) {
          best = hit.t;
          blockResult = { x, y, z, n: hit.n };
        }
      }
    }
    if (tMaxX < tMaxY && tMaxX < tMaxZ) {
      x += stepX;
      t = tMaxX;
      tMaxX += tDeltaX;
    } else if (tMaxY < tMaxZ) {
      y += stepY;
      t = tMaxY;
      tMaxY += tDeltaY;
    } else {
      z += stepZ;
      t = tMaxZ;
      tMaxZ += tDeltaZ;
    }
    if (blockResult && t > best) break; // no closer voxel possible
  }

  // ── Resolve nearest of entity / block / ground ────────────────────────────
  // Entity wins only if its box is at/in front of the nearest block (matches old behavior).
  if (nearestEntity && (!blockResult || entityDist <= best)) {
    return {
      type: "entity",
      position: [Math.floor(nearestEntity.position[0]), Math.floor(nearestEntity.position[1]), Math.floor(nearestEntity.position[2])],
      normal: [0, 1, 0],
      entity: nearestEntity,
    };
  }

  if (blockResult) {
    const block = inp.blockAt(blockResult.x, blockResult.y, blockResult.z);
    if (block) {
      return {
        type: "block",
        position: [blockResult.x, blockResult.y, blockResult.z],
        normal: blockResult.n,
        block,
        hitPoint: [o[0] + d[0] * best, o[1] + d[1] * best, o[2] + d[2] * best],
      };
    }
  }

  // Ground plane y=0 (the editor floor).
  if (Math.abs(d[1]) > 1e-9) {
    const tg = (0 - o[1]) / d[1];
    if (tg > 0 && tg <= maxDist) {
      const px = o[0] + d[0] * tg;
      const pz = o[2] + d[2] * tg;
      return { type: "ground", position: [Math.floor(px), 0, Math.floor(pz)], normal: [0, 1, 0], hitPoint: [px, 0, pz] };
    }
  }
  return null;
}

/** Vanilla entity collision box size (ported from EditableWorldRenderer.entityBoundsSize). */
export function entityBoundsSize(e: EditableEntity): Vec3Tuple {
  const type = e.type.replace(/^minecraft:/, "");
  if (type === "item") return [0.25, 0.25, 0.25];
  if (type === "tnt") return [0.98, 0.98, 0.98];
  if (type === "falling_block") return [0.98, 0.98, 0.98];
  if (type === "armor_stand") return [0.5, 1.975, 0.5];
  if (type === "sheep") return [0.9, 1.3, 0.9];
  if (type === "minecart" || type.endsWith("_minecart")) return [0.98, 0.7, 0.98];
  if (type.endsWith("_boat") || type.endsWith("_raft") || type.endsWith("_chest_boat")) return [1.375, 0.5625, 1.375];
  return [0.6, 1.8, 0.6];
}
