// Block-entity renderer registry — registers every BE def + the registry-backed public API.
// RENDERER_PLAN.md §18, Phase 4.5c. Replaces the old placeholder `blockEntityModels.ts` example.

import type { RawVertex } from "../../../mesh/VertexFormat";
import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";
import { registerBE, getBEDef, hasBEDef, type BEBakeContext, type BESpecialDraw } from "./registry";
import { CHEST } from "./chest";
import { SHULKER } from "./shulker";
import { BED } from "./bed";
import { SKULL } from "./skull";
import { BELL } from "./bell";
import { COPPER_GOLEM_STATUE } from "./copperGolem";
import { BOOK } from "./book";
import { SIGN } from "./signs";
import { BANNER } from "./banner";
import { DECORATED_POT } from "./pot";
import { CONDUIT } from "./conduit";
import { SPECIAL_DEFS } from "./special";

// ── Register every block-entity renderer ──────────────────────────────────────
const ALL = [CHEST, SHULKER, BED, SKULL, BELL, COPPER_GOLEM_STATUE, BOOK, SIGN, BANNER, DECORATED_POT, CONDUIT, ...SPECIAL_DEFS];
for (const def of ALL) registerBE(def);

export { getBEDef, hasBEDef, allBETextures, allBEDefs, registeredBETypes, boxBE } from "./registry";
export type { BlockEntityDef, BoxBEConfig, BEBakeContext, BESpecialDraw } from "./registry";

/** Bake a BE's section-local geometry via its registered renderer (null = no model). */
export function bakeBlockEntity(rec: BlockEntityRecord, ctx: BEBakeContext, clock: number, animating: boolean): RawVertex[] | null {
  return getBEDef(rec.type)?.bake?.(rec, ctx, clock, animating) ?? null;
}

/** A BE's per-frame special draws (beam / portal / wireframe / embedded), or []. */
export function blockEntitySpecials(rec: BlockEntityRecord, clock: number): BESpecialDraw[] {
  return getBEDef(rec.type)?.special?.(rec, clock) ?? [];
}

/** A type has any registered renderer (box or special). */
export function hasBlockEntityModel(type: string): boolean {
  return hasBEDef(type);
}

/**
 * A type whose BE box model REPLACES the block's terrain appearance, so terrain must suppress it (else a
 * wrong/empty cube double-draws): chest/shulker/sign/skull/banner/pot/conduit (their terrain model is empty).
 * Excluded:
 *  - Special-only BEs (spawner cage, vault frame, campfire logs, shelf planks, beacon, brushable sand) have
 *    NO box `bake`: the block is a normal TERRAIN model the BE only ADDS items/effects to — terrain must KEEP
 *    drawing them (suppressing left only the floating item — the spawner/campfire bug).
 *  - HYBRID box BEs (`hybridTerrain`): bell (posts/bars) + lectern/enchanting_table (stand/cube) have a REAL
 *    terrain frame the BE only hangs the bell/book onto — terrain must KEEP drawing the frame (else it vanishes,
 *    leaving a floating bell/book). So a box `bake` alone does not imply suppression.
 */
export function hasBlockEntityBoxModel(type: string): boolean {
  const def = getBEDef(type);
  return !!def?.bake && !def.hybridTerrain;
}

/**
 * Hybrid classifier: split a section's BEs into the IDLE set (baked into static section geometry) and
 * the ANIMATING set (re-posed per frame). A BE is in exactly ONE set. Only BEs whose def has box
 * geometry participate (special-only BEs draw every frame via `blockEntitySpecials`).
 *
 * A box BE is per-frame iff it's `animated` AND (its record's `animating` toggle is on OR its def is an
 * `idleLoop` — banner wave / conduit spin, the continuous motion vanilla always shows; BBE §5). So an
 * idle-loop BE animates BY DEFAULT (no editor toggle), while event-driven BEs (chest/bell/shulker/pot)
 * bake idle and animate only when toggled. Idle-loop BEs are never in the static bake → no double-draw.
 */
export function splitSectionBEs(records: readonly BlockEntityRecord[]): { idle: BlockEntityRecord[]; animating: BlockEntityRecord[] } {
  const idle: BlockEntityRecord[] = [];
  const animating: BlockEntityRecord[] = [];
  for (const r of records) {
    const def = getBEDef(r.type);
    if (!def?.bake) continue;
    (def.animated && (r.animating || def.idleLoop) ? animating : idle).push(r);
  }
  return { idle, animating };
}

/**
 * Will this BE be drawn by a PER-FRAME (animating) path this frame? — true for an idle-loop box motion
 * (banner wave / conduit spin), a user-toggled box preview, OR an always-animated special-only BE
 * (spinning spawner/vault, end portal/gateway, beacon beam). The render gate uses this to keep drawing
 * while any BE animates (else idle-loop/special motion freezes under render-on-demand). Mirrors the
 * `splitSectionBEs` box predicate so the two never diverge.
 */
export function blockEntityAnimatesPerFrame(rec: BlockEntityRecord): boolean {
  const def = getBEDef(rec.type);
  if (!def?.animated) return false;
  if (def.bake) return !!(rec.animating || def.idleLoop); // box: idle-loop or user-toggled
  return !!def.special; // special-only animated (spawner/vault spin, portal shader, beam)
}
