// Phase P instrumentation (RENDERER_OPTIMIZATION_PLAN.md §Phase P). Opt-in, near-zero-overhead counters
// for the audit-named hot paths + the per-tick remesh job mix. DISABLED by default → every `bump*` is a
// single boolean branch, so it is free in production and in the unit suite (which never enables it); the
// sim-storm harness flips `enabled = true` and reads per-frame DELTAS off the cumulative counters.
//
// These are deliberately COARSE: one bump per operation carrying the element count (not per-element), so
// the meshing/render inner loops stay clean and the instrumentation can't itself distort the numbers.

/** Per-category allocation churn (the GC-pressure the audits target; the single-pass sink + the A/F-series
 *  pools drive these → ~0 in steady state, so a per-frame delta > 0 flags a regression). Armed at the real
 *  allocation/growth sites (X5). */
export interface AllocCounts {
  /** `RawVertex` objects emitted by a bake. Terrain (`mesh/SectionMesher`) is 0 — single-pass inline encode.
   *  Entity/BE model bakes (`EntityModelFactory.getMesh`/`dynamicMesh`) bump it on a cache miss / re-pose;
   *  the A4 dynamic-mob SINK path emits ZERO RawVertex, so a sink-baked walking mob keeps this at 0. */
  rawVertex: number;
  /** `SectionDraw` objects re-resolved by the F3 incremental draw list (`app/scene.DrawList`). 0 when no
   *  section committed this frame; a spike marks a full rebuild (first build / arena relocation). */
  drawList: number;
  /** Frustum instances allocated (`camera/Frustum` ctor). F4 reuses one per renderer ⇒ 0/frame steady state. */
  frustum: number;
  /** Entity per-frame pool growth (`EntityModelFactory` mat4 arena + draw/sort pools, A3/A4). 0 once the
   *  pools reach the scene's high-water mark — a per-frame delta means a pool is leaking/growing unbounded. */
  entityEmit: number;
}

/** Per-tick remesh job mix (the key dynamic metric for P1). */
export interface JobCounts {
  /** Sections marked dirty by an edit batch. */
  dirtied: number;
  /** `meshSection` calls dispatched. */
  meshed: number;
  /** Builds dropped by `acceptBuild` on a generation mismatch (≈0 until off-thread meshing — Phase 5). */
  discarded: number;
  /** OCC-1: actual camera occlusion-BFS runs (`computeVisibleSections`). With the `occBfsCache` flag OFF
   *  this equals the camera-move frame count (status quo); ON it drops to ≈ home-section crossings +
   *  graphRevision bumps (≈0 across a pure-rotation/orbit segment). The metric that proves the cache win. */
  occBfsRuns: number;
  /** OCC-1: total sections walked by those BFS runs (dequeued nodes). Pairs with `occBfsRuns` so a frame's
   *  per-node traversal cost is visible; 0 on frames the cache reuses the reachable set. */
  occVisited: number;
}

class Instrument {
  /** Off in production + tests; the perf harness turns it on. */
  enabled = false;

  /** Cumulative since enable; callers snapshot DELTAS per frame/tick (uniform with `passCount`). */
  readonly alloc: AllocCounts = { rawVertex: 0, drawList: 0, frustum: 0, entityEmit: 0 };
  readonly job: JobCounts = { dirtied: 0, meshed: 0, discarded: 0, occBfsRuns: 0, occVisited: 0 };

  bumpAlloc(category: keyof AllocCounts, n = 1): void {
    if (this.enabled) this.alloc[category] += n;
  }

  bumpJob(category: keyof JobCounts, n = 1): void {
    if (this.enabled) this.job[category] += n;
  }

  /** Test/harness reset to a clean baseline. */
  reset(): void {
    this.alloc.rawVertex = this.alloc.drawList = this.alloc.frustum = this.alloc.entityEmit = 0;
    this.job.dirtied = this.job.meshed = this.job.discarded = 0;
    this.job.occBfsRuns = this.job.occVisited = 0;
  }
}

/** Process-wide singleton — the hot paths import this and bump it; the façade snapshots it per frame. */
export const instrument = new Instrument();
