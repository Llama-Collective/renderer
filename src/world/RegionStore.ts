// MEM-1 — RegionStore: the 256-section (8×4×8) RenderRegion tier (DEFAULT-OFF).
//
// Maintains, per packed region key, an INCLUSIVE section-coord AABB over the populated sections it holds
// plus a live-section count. Fed EXACTLY at the two `SectionStore` mutation sites — `getOrCreate` (add)
// and `dispose` (remove) — behind a constructor-injected optional `regions?: RegionStore`; the default
// `new SectionStore(uploader)` passes none, so every hook is a `?.` no-op and the shipped path is
// byte-identical (TRAP MEM-1.B: membership is coord-derived from the two presence sites, NEVER from a
// pending build).
//
// Two wins it enables (both default-off):
//   1. Region-level cull PRE-PASS (scene.ts::occlusionVisibleSet, `viewerRegionCullEnabled`): the
//      per-region AABBs let a region box be frustum-rejected as a unit, then the survivors' union is fed
//      to the UNCHANGED per-section BFS as a tightened `bounds`. The visible SET is identical (out-of-
//      frustum sections already dead-end the BFS); only the traversal domain shrinks (TRAP MEM-1.C).
//   2. Per-region CONTIGUOUS reclaim (GpuSectionUploader, `regionArenas`): when a region's last live
//      section is removed, `onRegionEmpty(regionKey)` lets the uploader destroy that region's per-pass
//      arenas as whole buffers — O(1) `disposeRegion` instead of N per-section `free()` that fragment a
//      global arena. The hook fires EXACTLY ONCE on count→0, from inside `SectionStore.dispose` (the
//      existing presented-mutation site, presented already nulled — TRAP MEM-1.B / T-PRES 2.B).
//
// Honest scope (TRAP MEM-1.A): WebGPU has no multidraw, so the region tier does NOT reduce draw calls —
// `encodeDraws` still emits one drawIndexed per section-pass. The only wins are coarser cull + contiguous
// reclaim. The headline verifiable property is the PARITY GATE: with both flags on, the occlusion-visible
// set and the flat draw list are equal element-for-element to the flags-off run for the same camera.

import { packRegionOfSection, REGION_SHIFT_X, REGION_SHIFT_Y, REGION_SHIFT_Z } from "./RegionKey";

/** Inclusive section-coord AABB over the populated sections of one region, plus a live-section count. */
export interface RegionAABB {
  /** Min section coords (inclusive). */
  loX: number; loY: number; loZ: number;
  /** Max section coords (inclusive). */
  hiX: number; hiY: number; hiZ: number;
  /** Sections currently registered in this region (drives the count→0 reclaim hook). */
  liveCount: number;
  /** Sections in this region that currently carry drawable geometry (drives the `regionRev` bump). */
  drawableCount: number;
}

/** A region's last live section was removed — the uploader frees its per-pass arenas as whole buffers
 *  (contiguous O(1) reclaim). Fired ONLY from `SectionStore.dispose` (presented already nulled). */
export type RegionEmptyHook = (regionKey: number) => void;

export class RegionStore {
  private readonly regions = new Map<number, RegionAABB>();
  /** Bumped ONLY when a region's DRAWABLE presence changes (a section gained/lost geometry, or the first/
   *  last drawable section of a region appeared/vanished) — the mirror of `SectionStore.graphRev`. A
   *  connectivity-neutral remesh that keeps the same drawable-presence does NOT bump it. */
  private rev = 0;

  /** Optional reclaim hook — set by the uploader-aware wiring; absent ⇒ no per-region reclaim. */
  constructor(private readonly onRegionEmpty?: RegionEmptyHook) {}

  /** Current region revision (bumps on drawable-presence change only). */
  get regionRev(): number {
    return this.rev;
  }

  /** Number of non-empty regions (live count > 0). Diagnostic / test. */
  get regionCount(): number {
    return this.regions.size;
  }

  /** The per-region AABB + counts, or undefined. Read-only view for the cull pre-pass / tests. */
  get(regionKey: number): Readonly<RegionAABB> | undefined {
    return this.regions.get(regionKey);
  }

  /** Iterate the live regions (for the cull pre-pass). */
  entries(): IterableIterator<[number, RegionAABB]> {
    return this.regions.entries();
  }

  /** A section coordinate now EXISTS in the store (SectionStore.getOrCreate). Registers it into its region,
   *  extending the region's AABB to include it. Drawable presence is folded in separately via
   *  `setDrawable` at commit/dispose time — so a freshly-created (not-yet-meshed) section is counted as
   *  live (for reclaim) but not drawable (no `regionRev` bump until it actually presents geometry). */
  addSection(sx: number, sy: number, sz: number): void {
    const key = packRegionOfSection(sx, sy, sz);
    let r = this.regions.get(key);
    if (!r) {
      r = { loX: sx, loY: sy, loZ: sz, hiX: sx, hiY: sy, hiZ: sz, liveCount: 0, drawableCount: 0 };
      this.regions.set(key, r);
    } else {
      if (sx < r.loX) r.loX = sx;
      if (sy < r.loY) r.loY = sy;
      if (sz < r.loZ) r.loZ = sz;
      if (sx > r.hiX) r.hiX = sx;
      if (sy > r.hiY) r.hiY = sy;
      if (sz > r.hiZ) r.hiZ = sz;
    }
    r.liveCount++;
  }

  /** A section coordinate was REMOVED from the store (SectionStore.dispose). Decrements the region's live
   *  count; on count→0 drops the region AND fires `onRegionEmpty` EXACTLY ONCE (contiguous reclaim).
   *  `wasDrawable` lets dispose tell us whether a drawn section vanished (so the rev bumps + the drawable
   *  count stays consistent without a second pass). The AABB is NOT shrunk on removal — it is a
   *  conservative traversal bound (a slightly-too-large box never drops a reachable section, TRAP MEM-1.C),
   *  and it is recomputed from scratch on the region's next `addSection` rebuild only via emptying. */
  removeSection(sx: number, sy: number, sz: number, wasDrawable: boolean): void {
    const key = packRegionOfSection(sx, sy, sz);
    const r = this.regions.get(key);
    if (!r) return;
    if (wasDrawable && r.drawableCount > 0) {
      r.drawableCount--;
      this.rev++; // a drawn section vanished — the drawable presence of this region changed
    }
    r.liveCount--;
    if (r.liveCount <= 0) {
      this.regions.delete(key);
      this.onRegionEmpty?.(key); // contiguous reclaim — fires once, region's last section gone
    }
  }

  /** A section's DRAWABLE presence changed (commit folded geometry in or a clear removed it). Bumps
   *  `regionRev` only on an actual presence flip — a connectivity-neutral remesh that keeps geometry is a
   *  no-op (mirrors `SectionStore.graphRev`). Coord-derived; never reads a pending build (TRAP MEM-1.B). */
  setDrawable(sx: number, sy: number, sz: number, drawable: boolean): void {
    const key = packRegionOfSection(sx, sy, sz);
    const r = this.regions.get(key);
    if (!r) return; // the section must have been added first (getOrCreate runs before commit)
    if (drawable) {
      r.drawableCount++;
      this.rev++; // a section gained geometry — drawable presence changed
    } else if (r.drawableCount > 0) {
      r.drawableCount--;
      this.rev++; // a section lost geometry — drawable presence changed
    }
  }
}

// Re-export the region span so the cull pre-pass can size a region's world AABB without re-deriving it.
export { REGION_SHIFT_X, REGION_SHIFT_Y, REGION_SHIFT_Z };
