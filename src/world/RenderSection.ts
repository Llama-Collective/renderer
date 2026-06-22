// Per-16³ section presentation state — the most important type in the project.
// RENDERER_PLAN.md §4, §5.
//
// The renderer draws ONLY from `presented`. A section can simultaneously hold a presented
// mesh AND a pending result. Starting/cancelling a job NEVER touches `presented`
// (TRAP 4.A, 4.B). `presented` changes only at atomic commit (§5 / SectionStore.commit).

import type { Generation } from "../types";
import { parseSectionKey, type SectionKey } from "./SectionKey";
import type { CommittedSection } from "./SectionUploader";
import type { SectionBuildOutput } from "../workers/BuildOutput";
import type { SortOutput } from "../workers/SortOutput";
import { VIS_ALL, type SectionVisibility } from "./SectionVisibility";

export enum DirtyFlag {
  None = 0,
  Geometry = 1 << 0,
  Sort = 1 << 1,
  Light = 1 << 2,
}

/**
 * The kind of pending rebuild a section is waiting for (AUDIT N1 — `ChunkUpdateTypes`). Ordered
 * by cost/priority so a coalescing OR-join keeps the STRONGEST pending kind: a section dirtied as Sort
 * and then Rebuild stays Rebuild; an Initial build outranks a Rebuild. The scheduler enqueues a section
 * at most once (idempotent), and the type is cleared only when the job is dispatched (`clear-on-dispatch`).
 */
export enum UpdateType {
  None = 0,
  Sort = 1, // index-only re-sort (cheapest)
  Light = 2, // P6/LM-2 — light-only re-bake: re-mesh for new light, in-place reupload (no realloc/BFS)
  Rebuild = 3, // full geometry remesh
  Initial = 4, // first build (most important)
}

/** OR-join two pending update kinds, keeping the stronger (idempotent). `upgradePendingUpdate`. */
export function upgradeUpdateType(current: UpdateType, next: UpdateType): UpdateType {
  return next > current ? next : current;
}

/** Whether a section is live or being torn down. `Disposing` blocks accept/commit. */
export enum Lifecycle {
  Active = "active",
  Disposing = "disposing",
}

export class RenderSection {
  readonly key: SectionKey;
  /** Numeric section coords, parsed ONCE here from `key` (P1.2). Hot loops — the camera BFS bounds, the
   *  upload-priority Chebyshev distance, region add/remove — read these instead of re-splitting the string
   *  key per section per frame (the churn the audit flagged). */
  readonly sx: number;
  readonly sy: number;
  readonly sz: number;

  // --- Occlusion graph state (OCC-1; set at commit from the build's visibilityData) ---
  /** Face-to-face connectivity of the presented geometry (DirectionalVisGraph). VIS_ALL until
   *  built (empty space is fully traversable — OCC-6). Read by the camera BFS (world/OcclusionCuller). */
  visibility: SectionVisibility = VIS_ALL;
  /** OCC-2 (default-off): the per-perspective DIRECTION_SETS (`computeSectionVisibilitySets`), 4 masks whose
   *  `[0]` equals `visibility` (the symmetric union). `undefined` until built with `perPerspectiveVisibility`
   *  (and on every off-path build) — the camera BFS then joins these by camera quadrant, falling back to the
   *  symmetric `visibility` word whenever this is absent (a mixed/legacy build can never under-draw). Written
   *  ONLY by `SectionStore.commit` (generation-matched), read-only at BFS time (T-PRES). */
  visibilitySets?: Uint32Array;
  /** Does `presented` carry drawable geometry? (Empty/cleared sections are traversed but not drawn.) */
  hasGeometry = false;

  // --- Presentation state (drawn from `presented` only) ---
  presented: CommittedSection | null = null;
  pendingBuild: SectionBuildOutput | null = null;
  pendingSort: SortOutput | null = null;

  // --- Versioning (TRAP 4.C: match current generation, not "newer in time") ---
  generation: Generation = 0;
  lastBuiltGeneration: Generation = -1;

  dirtyFlags: number = DirtyFlag.None;
  lifecycle: Lifecycle = Lifecycle.Active;

  // --- Scheduling (AUDIT N1: coalesce N re-dirties into ONE queued job) ---
  /** The strongest pending rebuild kind queued for this section, or None if not queued. Set on dirty
   *  (OR-joined via upgradeUpdateType), cleared when the scheduler dispatches the job. */
  pendingUpdateType: UpdateType = UpdateType.None;
  /** True while this section sits in the scheduler's queue (so a re-dirty coalesces instead of re-queuing). */
  queued = false;

  constructor(key: SectionKey) {
    this.key = key;
    const [sx, sy, sz] = parseSectionKey(key);
    this.sx = sx;
    this.sy = sy;
    this.sz = sz;
  }
}
