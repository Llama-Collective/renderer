// Per-FACING / placement transform helpers for block-entity models. RENDERER_PLAN.md §18.
//
// Block-entity geometry is baked in model space (model pixels / 16 = block units) and placed into the
// block cell [0,1]³ by a transform that rotates about the block centre per FACING and (sometimes) lays
// the model flat / scales it. These mirror the vanilla `poseStack` op sequences (Chest/Shulker/Bed/etc).

import { DEG, identity, mul, rotationAxis, rotationX, rotationY, rotationZ, scaling, translation, type Mat4 } from "../../../mesh/entity/mat4";

export { DEG }; // re-export the canonical constant — the block-entity defs import DEG from here

/** Direction.toYRot(): south=0, west=90, north=180, east=270. */
const FACING_YROT: Record<string, number> = { south: 0, west: 90, north: 180, east: 270, up: 0, down: 0 };

export function toYRot(facing: string | undefined): number {
  return FACING_YROT[facing ?? "north"] ?? 0;
}

/** north↔south, east↔west. The single home of the cardinal-flip the BE renderers share. A non-cardinal
 *  `facing` (or undefined) yields `fallback` — the call sites differ deliberately (copper-golem statue
 *  falls back to "north", a skull passes the facing through), so the fallback is explicit, not baked in. */
const OPPOSITE_FACING: Record<string, string> = { north: "south", south: "north", east: "west", west: "east" };
export function oppositeFacing(facing: string | undefined, fallback: string): string {
  const o = facing === undefined ? undefined : OPPOSITE_FACING[facing];
  return o ?? fallback;
}

/** Section-local base transform: place a BE record at its block (rec − sectionOrigin), then apply the
 *  per-block `transform`. The shared form behind every hand-built BE `bake()` (chest/banner/sign/skull/pot). */
export function sectionLocal(rec: { x: number; y: number; z: number }, origin: readonly [number, number, number], transform: Mat4): Mat4 {
  return mul(translation(rec.x - origin[0], rec.y - origin[1], rec.z - origin[2]), transform);
}

/** Rotate `deg` about the vertical axis through the block centre (Y of the pivot is irrelevant). */
export function rotAroundY(deg: number, cx = 0.5, cz = 0.5): Mat4 {
  return mul(translation(cx, 0, cz), rotationY(deg * DEG), translation(-cx, 0, -cz));
}

/** Chest / decorated-pot style: rotate by `baseDeg - toYRot(facing)` about block centre.
 *
 * BBE opt #6 (cached per-`(type, FACING)` transform tables): the result depends ONLY on `(baseDeg, facing)`
 * — a tiny finite set (≤ a handful of base angles × 6 facings) — so it's memoized. EVERY caller feeds the
 * result into `mul(...)` (via `sectionLocal`/`bannerTransform`/etc.), and `mul`→`multiply` allocates a NEW
 * matrix and never mutates its inputs (mat4.ts) — so handing out a shared cached `Mat4` is safe. Treat the
 * returned matrix as IMMUTABLE; never `multiplyInto`/mutate it in place (that would poison the cache). */
const FACING_ROT_CACHE = new Map<string, Mat4>();
export function facingRot(facing: string | undefined, baseDeg = 0): Mat4 {
  const key = `${baseDeg}|${facing ?? "north"}`;
  let m = FACING_ROT_CACHE.get(key);
  if (!m) FACING_ROT_CACHE.set(key, (m = rotAroundY(baseDeg - toYRot(facing))));
  return m;
}

export { identity, mul, rotationAxis, rotationX, rotationY, rotationZ, scaling, translation };
export type { Mat4 };
