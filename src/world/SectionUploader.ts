// GPU-side abstraction for committing a section, so world/ stays free of GPU types
// (RENDERER_PLAN.md §23). The render layer implements this with a GraphicsDevice (and,
// later, BufferArena). SectionStore drives the Stable Presentation Invariant through it.

import type { Generation } from "../types";
import type { SectionBuildOutput } from "../workers/BuildOutput";
import type { SortOutput } from "../workers/SortOutput";

/**
 * Opaque committed GPU data for one section — what the renderer draws from. Its concrete
 * shape (which buffers / arena allocations) lives in the render layer; world/ only needs
 * the generation it was built from and the ability to hand it back to be freed.
 */
export interface CommittedSection {
  readonly generation: Generation;
}

export interface SectionUploader {
  /**
   * Upload a build output's geometry; returns committed GPU data, or `null` if the
   * per-frame upload budget is exhausted (caller keeps old data, retries next frame —
   * Invariant rule 4). An output with no geometry yields a valid EMPTY CommittedSection
   * (i.e. clears the section). Does NOT dispose the output — the caller owns its lifecycle.
   *
   * `prev` (the section's current presented data) lets the uploader reuse unchanged buffers — e.g.
   * moving identical translucent geometry across instead of re-uploading (TRAP 12.D). When buffers
   * are moved out of `prev`, they are removed from it so the subsequent `free(prev)` won't destroy them.
   */
  upload(out: SectionBuildOutput, prev?: CommittedSection): CommittedSection | null;

  /**
   * Apply an index-only sort onto already-committed data: reuses vertex buffers, replaces
   * only the translucent index buffer (Invariant rule 5 — never touches vertex data).
   * Frees the old index buffer internally, so the caller must NOT also free(prev).
   * Returns updated committed data, or `null` if over budget.
   */
  uploadSort(prev: CommittedSection, out: SortOutput): CommittedSection | null;

  /**
   * P6/LM-2 — apply a LIGHT-ONLY rebuild onto already-committed data: the geometry is identical (a pure
   * light change never moves a vertex), so reuse every existing vertex/index allocation and update its
   * bytes IN PLACE (same size → no realloc, no buffer relocation, no draw-list/F1 churn beyond the
   * commit). Returns the updated (same-identity) committed data on success; `"mismatch"` if the geometry
   * actually differs (caller must fall back to a full `upload`); `null` if over the per-frame budget
   * (caller keeps old data + retries). Like `uploadSort`, this is a partial reupload — the achievable
   * local-light win (LM-5); only GLOBAL brightness is truly free (the LUT rewrite). Optional: a uploader
   * may omit it (callers treat absence as `"mismatch"` ⇒ always full-upload).
   */
  uploadLight?(prev: CommittedSection, out: SectionBuildOutput): CommittedSection | "mismatch" | null;

  /** Release all GPU resources held by committed data. */
  free(c: CommittedSection): void;

  /**
   * MEM-1 (optional): contiguous per-region reclaim. When the uploader shards into per-region arenas
   * (`regionArenas` on) and a region's LAST live section is removed, the RegionStore's `onRegionEmpty` hook
   * calls this to destroy that region's per-pass arenas as WHOLE buffers (O(1), zero compaction copies)
   * instead of fragmenting a global arena via N per-section `free()`. A single-arena uploader / test mock may
   * omit it (then per-section `free()` reclaims as today). Fired ONLY from `SectionStore.dispose` after every
   * section in the region has had its `presented` freed + nulled — never against a live presented draw
   * (TRAP MEM-1.B / T-PRES 2.B). No-op (or absent) in single-arena mode.
   */
  disposeRegion?(regionKey: number): void;

  /**
   * Monotonic revision that bumps whenever the GPU buffers backing committed sections were RELOCATED
   * (an arena grow/compaction → new buffer handle + changed byte offsets). An incremental draw-list
   * cache (F3) re-resolves every section's (buffer, offset) when this changes, since its cached snapshots
   * went stale (TRAP 4.A — the relocation the once-per-commit full rebuild used to hide). Optional: a
   * uploader with no relocating storage (a test mock) may omit it; callers treat absence as a constant 0.
   */
  layoutRevision?(): number;

  /**
   * The uploader's OWN per-frame byte budget, if it self-limits (Infinity / absent ⇒ unbounded). INFRA-5:
   * when an UploadScheduler is the SINGLE budget authority it asserts this is Infinity, so two budgets can't
   * silently both gate throughput (the uploader's reuse-adjusted byte check interacting with the scheduler's
   * gross-byte budget). A STANDALONE uploader (screenshot / item-slot, no scheduler) may legitimately set a
   * finite budget — it simply never reaches that assertion. Optional: test mocks omit it (treated unbounded).
   */
  readonly budgetBytes?: number;
}
