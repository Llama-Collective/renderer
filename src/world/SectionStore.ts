// Owns all RenderSections + generation bumping + the atomic commit that enforces the
// Stable Presentation Invariant. RENDERER_PLAN.md §4, §5.
//
// The accept/commit/discard methods here are the ones GATE 4.1 / T-PRES-1..5 test. This
// module holds NO GPU types — it drives the GPU through an injected SectionUploader.

import type { Generation, Vec3i } from "../types";
import { DirtyReason } from "../types";
import { RenderSection, DirtyFlag, Lifecycle } from "./RenderSection";
import type { SectionKey } from "./SectionKey";
import { packSectionCoord } from "./OcclusionCuller";
import type { SectionBuildOutput } from "../workers/BuildOutput";
import type { SortOutput } from "../workers/SortOutput";
import type { CommittedSection, SectionUploader } from "./SectionUploader";
import { VIS_ALL } from "./SectionVisibility";
import type { RegionStore } from "./RegionStore";

/** Outcome of staging a worker output. `DiscardedStale` means the output was disposed. */
export enum AcceptResult {
  Accepted = "accepted",
  DiscardedStale = "discarded-stale",
}

export class SectionStore {
  private readonly sections = new Map<SectionKey, RenderSection>();
  /** Packed-coord mirror of `sections` for the camera occlusion BFS — its `getByCoords` per-node lookup
   *  keys off the same packed int the BFS already computes, so it allocates NO `"sx,sy,sz"` string per
   *  traversed section (S1). Kept in sync at the ONLY two `sections` mutation sites: `getOrCreate`/`dispose`. */
  private readonly byCoords = new Map<number, RenderSection>();
  /** OCC-1/N4 change-mask: bumped whenever a commit/dispose changes the occlusion GRAPH (a section's
   *  face connectivity or its has-geometry presence) — NOT on connectivity-neutral remeshes. The camera
   *  occlusion BFS re-runs only when this (or the camera's home section) changes (TRAP 3.A). */
  private graphRev = 0;

  /**
   * `regions` (MEM-1, DEFAULT-OFF): the optional 256-section RegionStore tier. When undefined (the shipped
   * default) every region hook below is a `?.` no-op, so `getOrCreate`/`dispose`/`commit` run UNCHANGED and
   * the render is byte-identical. When injected, the store feeds the RegionStore at exactly these two
   * mutation sites (getOrCreate add / dispose remove) plus the commit drawable-flip, so region membership
   * is coord-derived from presence — never from a pending build (TRAP MEM-1.B).
   */
  constructor(
    private readonly uploader: SectionUploader,
    private readonly regions?: RegionStore,
  ) {}

  /** The MEM-1 RegionStore tier, or undefined on the default (single-arena, section-granular) path. */
  get regionStore(): RegionStore | undefined {
    return this.regions;
  }

  /** Current occlusion-graph revision (see `graphRev`). */
  get graphRevision(): number {
    return this.graphRev;
  }

  // ── Populated-region AABB (occlusion BFS traversal bound) ─────────────────────────────────────────
  // Cached against `graphRev`, which bumps exactly when a section's has-geometry presence changes — so the
  // box is recomputed only when the drawable-section set changes, NOT per frame (a rotating camera over a
  // static world recomputes nothing here). This is the finite domain the camera BFS is clamped to (
  // loaded-section set + render-distance bound), without which OCC-6 air would let the flood escape the
  // populated region across the whole frustum (the empty-world regression).
  private boundsRev = -1;
  private boundsLo: Vec3i | null = null;
  private boundsHi: Vec3i | null = null;

  // ── F3 incremental draw-list membership ───────────────────────────────────────────────────────────
  // commit() and dispose() are the SOLE writers of `presented` (the Stable Presentation Invariant), so
  // recording the touched key HERE gives a complete, leak-free change set the incremental draw-list cache
  // drains each committing frame — O(changed sections) instead of an O(world) `collectDraws` rebuild per
  // commit. Paired with `uploadLayoutRevision()` (arena relocation) it's TRAP-4.A-safe (DrawList re-resolves
  // everyone when an arena grew/compacted). Single consumer (the DrawList); it clears after draining.
  private readonly presentedDirty = new Set<SectionKey>();

  /** Sections whose `presented` changed since the last `clearPresentedChanges()` (F3). Read-only. */
  get presentedChanges(): ReadonlySet<SectionKey> {
    return this.presentedDirty;
  }

  /** Drop the recorded change set — the incremental draw-list cache calls this after draining it. */
  clearPresentedChanges(): void {
    this.presentedDirty.clear();
  }

  /** Relocation counter of the GPU buffers backing committed sections (F3 — see SectionUploader). 0 if the
   *  uploader has no relocating storage. Changes ⇒ cached (buffer, offset) draw snapshots are stale. */
  uploadLayoutRevision(): number {
    return this.uploader.layoutRevision?.() ?? 0;
  }

  /** The wrapped uploader's own per-frame byte budget (INFRA-5), or undefined if it doesn't expose one.
   *  The UploadScheduler reads this to assert it is the single budget authority (uploader must be unbounded). */
  uploaderBudgetBytes(): number | undefined {
    return this.uploader.budgetBytes;
  }

  /** Inclusive section-coord AABB over all sections that currently carry drawable geometry, or null if
   *  none. Air gaps BETWEEN drawable sections fall inside this box (so they stay traversable — OCC-6);
   *  anything beyond it is provably empty and need not be searched. */
  sectionBounds(): { lo: Vec3i; hi: Vec3i } | null {
    if (this.boundsRev !== this.graphRev) {
      this.boundsRev = this.graphRev;
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (const s of this.sections.values()) {
        if (!s.hasGeometry) continue; // only drawable sections define the box (empty ones add nothing to draw)
        const { sx, sy, sz } = s; // P1.2: cached numeric coords — no per-section string split in this per-frame loop
        if (sx < minX) minX = sx;
        if (sy < minY) minY = sy;
        if (sz < minZ) minZ = sz;
        if (sx > maxX) maxX = sx;
        if (sy > maxY) maxY = sy;
        if (sz > maxZ) maxZ = sz;
      }
      if (minX === Infinity) { this.boundsLo = null; this.boundsHi = null; }
      else { this.boundsLo = [minX, minY, minZ]; this.boundsHi = [maxX, maxY, maxZ]; }
    }
    return this.boundsLo && this.boundsHi ? { lo: this.boundsLo, hi: this.boundsHi } : null;
  }

  get(key: SectionKey): RenderSection | undefined {
    return this.sections.get(key);
  }

  /** Lookup by section coords without a string key — the camera BFS's per-node read (S1). Backed by the
   *  `byCoords` mirror so a traversal over thousands of sections allocates no template literals. */
  getByCoords(sx: number, sy: number, sz: number): RenderSection | undefined {
    return this.byCoords.get(packSectionCoord(sx, sy, sz));
  }

  getOrCreate(key: SectionKey): RenderSection {
    let s = this.sections.get(key);
    if (!s) {
      s = new RenderSection(key);
      this.sections.set(key, s);
      const { sx, sy, sz } = s; // P1.2: RenderSection parsed the coords once in its constructor
      this.byCoords.set(packSectionCoord(sx, sy, sz), s);
      this.regions?.addSection(sx, sy, sz); // MEM-1: the ONLY region "add" site (coord-derived presence)
    }
    return s;
  }

  /**
   * Mark a section dirty and bump its generation. Border edits dirty BOTH sides — callers
   * resolve affected sections via DirtyTracking and call this for each (TRAP 6.C).
   * Any staged-but-uncommitted output for an older generation becomes stale here and is
   * disposed so a later commit can't apply it.
   */
  markDirty(key: SectionKey, reason: DirtyReason): Generation {
    const s = this.getOrCreate(key);
    s.generation++;
    // LM-2: a Light dirty sets ONLY the Light flag; a section dirtied Light AND Geometry OR-joins to
    // Geometry|Light and is treated as a full rebuild at commit. The flag drives the commit path choice
    // (pure Light ⇒ in-place light reupload + no BFS); geometry-bearing reasons take the full upload.
    s.dirtyFlags |= reason === DirtyReason.Light ? DirtyFlag.Light : DirtyFlag.Geometry;
    if (s.pendingBuild && s.pendingBuild.generation !== s.generation) {
      s.pendingBuild.dispose();
      s.pendingBuild = null;
    }
    if (s.pendingSort && s.pendingSort.generation !== s.generation) {
      s.pendingSort.dispose();
      s.pendingSort = null;
    }
    return s.generation;
  }

  /**
   * Stage a build output. Accept IFF it matches the section's CURRENT generation (TRAP 4.C)
   * and is newer than what's presented; otherwise discard + dispose, leaving `presented`
   * untouched (Invariant rules 1–2). Does NOT upload — that happens in commit().
   */
  acceptBuild(out: SectionBuildOutput): AcceptResult {
    const s = this.sections.get(out.sectionKey);
    if (!s || s.lifecycle === Lifecycle.Disposing) {
      out.dispose();
      return AcceptResult.DiscardedStale;
    }
    if (out.generation !== s.generation || out.generation <= s.lastBuiltGeneration) {
      out.dispose();
      return AcceptResult.DiscardedStale;
    }
    if (s.pendingBuild) s.pendingBuild.dispose(); // superseded by a newer accepted build
    s.pendingBuild = out;
    return AcceptResult.Accepted;
  }

  /** Stage a sort output (index-only). Stale → discard without clearing (Invariant 5). */
  acceptSort(out: SortOutput): AcceptResult {
    const s = this.sections.get(out.sectionKey);
    if (!s || s.lifecycle === Lifecycle.Disposing) {
      out.dispose();
      return AcceptResult.DiscardedStale;
    }
    if (out.generation !== s.generation) {
      out.dispose();
      return AcceptResult.DiscardedStale;
    }
    if (s.pendingSort) s.pendingSort.dispose();
    s.pendingSort = out;
    return AcceptResult.Accepted;
  }

  /**
   * Attempt to upload staged work and atomically swap `presented`. Honors the uploader's
   * per-frame budget: if upload returns null, keeps old data and leaves the pending output
   * for a later frame (Invariant rule 4). The ONLY place `presented` changes. Returns true
   * iff `presented` changed.
   */
  commit(section: RenderSection): boolean {
    if (section.lifecycle === Lifecycle.Disposing) return false;
    let changed = false;

    if (section.pendingBuild) {
      const old = section.presented;
      // LM-2: a PURE light dirty (Light flag, no Geometry) with existing presented geometry takes the
      // in-place light-byte reupload — same allocations, no relocation, no draw-list realloc. A
      // `"mismatch"` (geometry actually differs) falls through to the full upload; `null` is over-budget.
      let next: CommittedSection | null = null;
      let lightInPlace = false;
      if (section.dirtyFlags === DirtyFlag.Light && old && this.uploader.uploadLight) {
        const r = this.uploader.uploadLight(old, section.pendingBuild);
        if (r === null) return false; // over budget — keep everything, retry next frame
        if (r !== "mismatch") {
          next = r; // in-place light update succeeded — `next === old` (allocations reused)
          lightInPlace = true;
        }
      }
      if (!lightInPlace) {
        // Pass the current presented data so the uploader can reuse unchanged buffers (TRAP 12.D); on
        // reuse it moves them out of `old`, so the free(old) below won't destroy them.
        next = this.uploader.upload(section.pendingBuild, old ?? undefined);
        if (next === null) return false; // over budget — keep everything, retry next frame
      }
      // OCC-1/N4: fold the build's visibility graph into the section and bump the graph revision ONLY
      // when the connectivity or has-geometry presence actually changed (connectivity-neutral remeshes —
      // a lamp toggling within a still-solid section — must NOT re-trigger the camera BFS; TRAP 3.A). A
      // light-only change recomputes the SAME visibility word ⇒ no bump ⇒ the BFS does not re-run (LM-2).
      const visData = section.pendingBuild.info.visibilityData;
      const newVis = visData?.[0] ?? VIS_ALL;
      const newHasGeom = section.pendingBuild.approxBytes > 0;
      // OCC-1/N4/TRAP OCC-2.C: bump the graph revision (⇒ camera BFS re-run) ONLY on the symmetric UNION word
      // or has-geometry presence change — NEVER on the raw 4-element sets array. A connectivity-neutral remesh
      // (a lamp toggling in a still-solid section) recomputes the same union ⇒ no bump ⇒ the BFS is skipped,
      // exactly as in the off-path (the sets refine which sections draw, they don't change reachability bumps).
      if (newVis !== section.visibility || newHasGeom !== section.hasGeometry) this.graphRev++;
      // MEM-1: fold the DRAWABLE-presence flip into the RegionStore (bumps `regionRev` only on an actual
      // has-geometry change — a connectivity-neutral remesh that stays drawable is a no-op, exactly like
      // graphRev). Coord-derived; the section was already added at getOrCreate, so the region exists.
      if (this.regions && newHasGeom !== section.hasGeometry) {
        this.regions.setDrawable(section.sx, section.sy, section.sz, newHasGeom); // P1.2: cached coords
      }
      section.visibility = newVis;
      // OCC-2 (default-off): retain the raw per-perspective DIRECTION_SETS for the camera-quadrant join. A
      // 1-element off-path build leaves this undefined ⇒ the BFS falls back to the symmetric `visibility` word.
      section.visibilitySets = visData && visData.length > 1 ? visData : undefined;
      section.hasGeometry = newHasGeom;
      section.presented = next;
      section.lastBuiltGeneration = section.pendingBuild.generation;
      section.pendingBuild.dispose();
      section.pendingBuild = null;
      // The light path reuses `old`'s allocations IN PLACE (next === old) — freeing it would destroy the
      // buffers we just updated and are presenting. Only the full-upload path frees the superseded data.
      if (old && !lightInPlace) this.uploader.free(old);
      // A pending sort that predates the just-applied geometry is meaningless now.
      if (section.pendingSort && section.pendingSort.generation !== section.generation) {
        section.pendingSort.dispose();
        section.pendingSort = null;
      }
      changed = true;
    }

    if (section.pendingSort) {
      if (section.presented) {
        const next = this.uploader.uploadSort(section.presented, section.pendingSort);
        if (next === null) return changed; // budget — keep sort for next frame
        section.presented = next; // vertices reused; uploader freed old index internally
        section.pendingSort.dispose();
        section.pendingSort = null;
        changed = true;
      } else {
        // Nothing to sort against — drop it.
        section.pendingSort.dispose();
        section.pendingSort = null;
      }
    }

    if (changed) {
      section.dirtyFlags = DirtyFlag.None;
      this.presentedDirty.add(section.key); // F3: this section's draws changed — re-resolve it incrementally
    }
    return changed;
  }

  dispose(key: SectionKey): void {
    const s = this.sections.get(key);
    if (!s) return;
    s.lifecycle = Lifecycle.Disposing;
    if (s.pendingBuild) {
      s.pendingBuild.dispose();
      s.pendingBuild = null;
    }
    if (s.pendingSort) {
      s.pendingSort.dispose();
      s.pendingSort = null;
    }
    if (s.presented) {
      this.uploader.free(s.presented);
      s.presented = null;
      if (s.hasGeometry) this.graphRev++; // a drawn section vanished — the occlusion graph changed (OCC-1)
    }
    this.presentedDirty.add(key); // F3: drop this section's draws on the next drain (store.get → undefined)
    this.sections.delete(key);
    const { sx, sy, sz } = s; // P1.2: cached coords (s is the section being disposed)
    this.byCoords.delete(packSectionCoord(sx, sy, sz)); // keep the BFS mirror in sync (S1)
    // MEM-1: the ONLY region "remove" site. `presented` is already nulled above, so when this drops the
    // region's last live section the RegionStore fires `onRegionEmpty` → the uploader frees the region's
    // per-pass arenas as whole buffers — never against a live presented draw (TRAP MEM-1.B / T-PRES 2.B).
    // `wasDrawable` (the section's has-geometry at removal) keeps the region's drawable count + regionRev
    // consistent in one pass, mirroring the graphRev bump above.
    this.regions?.removeSection(sx, sy, sz, s.hasGeometry);
  }

  *all(): IterableIterator<RenderSection> {
    yield* this.sections.values();
  }
}
