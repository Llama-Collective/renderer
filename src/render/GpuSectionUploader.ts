// Concrete SectionUploader backed by a GraphicsDevice. RENDERER_PLAN.md §5, §11.
//
// This is the render-layer GPU side of the Stable Presentation Invariant. Section VERTICES are
// sub-allocated from ONE shared BufferArena per pass (§11) instead of one device buffer per section
// — collapsing the live-buffer count from O(sections × passes) to O(passes). The committed section
// stores a STABLE `AllocationId` (not a buffer handle); the live buffer + byte offset are resolved at
// DRAW time (collectDraws), so an arena grow/compaction (fresh buffer, relocated bytes) is invisible
// to `presented` — the id never changes, and collectDraws re-runs the same frame any commit grows it.
//
// Section INDICES (translucent sorted order) are likewise sub-allocated from ONE shared index BufferArena
// (P2 follow-up — collapses the per-section index buffers, the remaining live-buffer bulk, to one arena).
// A re-sort updates the index allocation IN PLACE when the size is unchanged (U6); the DYNAMIC trigger
// writes directly into the section's arena slice at its byte offset. Indices stay 0-based (per-section),
// the byte offset locates the slice — exactly like vertices.
//
// Budget: tracks remaining bytes for the current frame. upload()/uploadSort() return null when a
// request would exceed the remaining budget, so the caller keeps old data and retries next frame
// (Invariant rule 4). Call beginFrame() once per frame to reset.

import type { GraphicsDevice } from "../core/GraphicsDevice";
import { BufferUsage } from "../core/GraphicsDevice";
import { BufferArena } from "../core/BufferArena";
import type { AllocationId, Generation, TerrainPass } from "../types";
import { TERRAIN_PASSES, TerrainPass as Pass } from "../types";
import type { CommittedSection, SectionUploader } from "../world/SectionUploader";
import type { SectionBuildOutput } from "../workers/BuildOutput";
import type { SortOutput } from "../workers/SortOutput";
import { SortType } from "../mesh/SortTypes";
import type { TQuad } from "../mesh/TranslucentCollector";
import { parseSectionKey } from "../world/SectionKey";
import { packRegionOfSection } from "../world/RegionKey";

/** Initial per-pass vertex-arena capacity; grows (doubling) on demand. 256 KiB ≈ a few thousand quads. */
const INITIAL_ARENA_BYTES = 1 << 18;
/** Initial index-arena capacity (indices are translucent-only + ~⅙ the vertex bytes). 64 KiB. */
const INITIAL_INDEX_ARENA_BYTES = 1 << 16;
/** MEM-1 (TRAP MEM-1.D): per-region arenas must start SMALL and GROW — a large initial size × 256 regions
 *  × passes would pre-allocate hundreds of MB of mostly-empty buffers (a sparse world has many thin
 *  regions). 16 KiB vertex / 4 KiB index seeds, doubling on demand exactly like the global arenas. */
const INITIAL_REGION_ARENA_BYTES = 1 << 14;
const INITIAL_REGION_INDEX_ARENA_BYTES = 1 << 12;
/** U8: reclaim a grown-then-emptied arena's high-water mark once ≥512 KiB is reclaimable AND it's mostly
 *  empty. Big enough that the GPU copy is worth it and it rarely fires (only after heavy churn frees). */
const RECLAIM_WASTE_BYTES = 1 << 19;

/** 4-byte align (matches BufferArena's internal ALIGN) — to compare an allocation's size to new data. */
function align4(n: number): number {
  return (n + 3) & ~3;
}

/**
 * Render-layer concrete committed data. Vertices AND indices are SUB-ALLOCATIONS in shared per-pass
 * arenas (§11): `vertexAlloc[pass]`/`indexAlloc[pass]` are stable ids into `vertexArena[pass]`/
 * `indexArena[pass]`; resolve the live buffer + byte offset at draw time so an arena grow/compaction is
 * invisible to `presented` (the id never changes; collectDraws re-resolves the same frame any commit grows it).
 */
export interface GpuCommittedSection extends CommittedSection {
  generation: Generation;
  /** Stable per-pass vertex allocation id within `vertexArena[pass]`. */
  vertexAlloc: Partial<Record<TerrainPass, AllocationId>>;
  /** The shared arena holding each pass's vertices — resolve `gpuBuffer`/`rangeOf` at draw time. */
  vertexArena: Partial<Record<TerrainPass, BufferArena>>;
  /** Stable per-pass index allocation id within `indexArena[pass]` (translucent sorted sections only).
   *  Absent ⇒ the pass uses the shared quad EBO. Resolve buffer+offset at draw time, exactly like vertices. */
  indexAlloc: Partial<Record<TerrainPass, AllocationId>>;
  indexArena: Partial<Record<TerrainPass, BufferArena>>;
  quadCount: Partial<Record<TerrainPass, number>>;
  /** FS-2/FS-3: per-pass facing VERTEX-count runs (opaque/cutout only) when the section was meshed with
   *  `partitionFacing`. Canonical facing order 0..6; the concatenated vertex runs follow the same order.
   *  Vertex-relative ⇒ survives arena relocation. Absent for a pass ⇒ one undifferentiated draw. */
  facingVertexCounts?: Partial<Record<TerrainPass, Uint32Array>>;
  /** Translucent sort classification (§12); absent ⇒ no translucent geometry. */
  sortType?: SortType;
  /** Per-quad geometry for DYNAMIC topological re-sort (TRAP 12.A); absent for static sorts. */
  sortQuads?: readonly TQuad[];
  /** FNV hash of the translucent geometry, for reuse on rebuild (TRAP 12.D). */
  quadHash?: number;
  /** Retained CPU translucent vertices (packed, 20B stride). Serves as the section-unique identity for the
   *  DYNAMIC re-sort trigger (a per-section ArrayBuffer — the shared arena handles can't identify a section,
   *  §11). Translucent geometry is a small fraction of a section, so this costs little. Absent ⇒ no
   *  translucent pass. */
  translucentVertexData?: ArrayBuffer;
}

export class GpuSectionUploader implements SectionUploader {
  private remainingBytes: number;
  /** One shared vertex arena per pass — the §11 pooling that collapses the live-buffer count.
   *  Created LAZILY on a pass's first allocation, so a world never pays for a pass it doesn't use.
   *  (MEM-1 single-arena mode — `regionArenas` off, the shipped default.) */
  private readonly arenas: Partial<Record<TerrainPass, BufferArena>> = {};
  /** One shared index arena (translucent indices only) — created lazily on first sorted-translucent commit. */
  private indexArena: BufferArena | null = null;

  /** MEM-1 (DEFAULT-OFF): per-region arenas, keyed by `packRegionOfSection`. When `regionArenas` is on,
   *  `arenaFor(pass, regionKey)` shards into one SMALL arena per (region, pass) instead of one global arena
   *  per pass — so a whole region's GPU memory can be freed CONTIGUOUSLY (`disposeRegion`, one destroyBuffer
   *  per pass) when its last section is removed, instead of N per-section `free()` fragmenting a global arena.
   *  No draw-layer change: the committed section stores the resolved arena REFERENCE per pass (TRAP MEM-1.A —
   *  this does NOT reduce draw calls; WebGPU has no multidraw). Off ⇒ these maps stay empty, byte-identical. */
  private readonly regionVertexArenas = new Map<number, Partial<Record<TerrainPass, BufferArena>>>();
  private readonly regionIndexArenas = new Map<number, BufferArena>();

  constructor(
    private readonly device: GraphicsDevice,
    private readonly budgetBytesPerFrame: number = Infinity,
    /** MEM-1: shard vertices/indices into per-region arenas (default OFF ⇒ the global single-arena path). */
    private readonly regionArenas = false,
  ) {
    this.remainingBytes = budgetBytesPerFrame;
  }

  /** Resolve the vertex arena backing `pass` for the section in `regionKey`. Single-arena mode ignores
   *  `regionKey` and returns the one global per-pass arena (byte-identical to pre-MEM-1). */
  private arenaFor(pass: TerrainPass, regionKey: number): BufferArena {
    if (!this.regionArenas) {
      return (this.arenas[pass] ??= new BufferArena(this.device, INITIAL_ARENA_BYTES, BufferUsage.Vertex));
    }
    let byPass = this.regionVertexArenas.get(regionKey);
    if (!byPass) {
      byPass = {};
      this.regionVertexArenas.set(regionKey, byPass);
    }
    return (byPass[pass] ??= new BufferArena(this.device, INITIAL_REGION_ARENA_BYTES, BufferUsage.Vertex));
  }

  /** Resolve the index arena for the section in `regionKey` (translucent indices only). Single-arena mode
   *  returns the one shared index arena. */
  private indexArenaFor(regionKey: number): BufferArena {
    if (!this.regionArenas) {
      return (this.indexArena ??= new BufferArena(this.device, INITIAL_INDEX_ARENA_BYTES, BufferUsage.Index));
    }
    let ia = this.regionIndexArenas.get(regionKey);
    if (!ia) {
      ia = new BufferArena(this.device, INITIAL_REGION_INDEX_ARENA_BYTES, BufferUsage.Index);
      this.regionIndexArenas.set(regionKey, ia);
    }
    return ia;
  }

  /** Every live vertex arena (global or per-region) — the shared spine for the compaction loop,
   *  `layoutRevision`, and teardown so each is written ONCE regardless of mode. */
  private *vertexArenas(): IterableIterator<BufferArena> {
    if (this.regionArenas) {
      for (const byPass of this.regionVertexArenas.values()) {
        for (const pass of TERRAIN_PASSES) {
          const a = byPass[pass];
          if (a) yield a;
        }
      }
    } else {
      for (const pass of TERRAIN_PASSES) {
        const a = this.arenas[pass];
        if (a) yield a;
      }
    }
  }

  /** Every live index arena (global or per-region). */
  private *indexArenas(): IterableIterator<BufferArena> {
    if (this.regionArenas) {
      yield* this.regionIndexArenas.values();
    } else if (this.indexArena) {
      yield this.indexArena;
    }
  }

  /** MEM-1 CONTIGUOUS RECLAIM: a region's last live section was removed (fired from SectionStore.dispose via
   *  the RegionStore `onRegionEmpty` hook). Destroy that region's per-pass vertex + index arenas as WHOLE
   *  buffers — O(1), one destroyBuffer per arena, ZERO `copyBufferBatch` (no compaction) — instead of the
   *  global arena's per-section `free()` + eventual compact. No-op in single-arena mode (the spec's "off ⇒
   *  per-section free runs exactly as today"). Safe: `dispose` already freed + nulled `presented` for every
   *  section in the region before the count hit 0, so this never destroys a live presented allocation
   *  (TRAP MEM-1.B / T-PRES 2.B). The freed allocations were within THESE arenas, so destroying them is the
   *  bulk reclaim of the same bytes. */
  disposeRegion(regionKey: number): void {
    if (!this.regionArenas) return; // single-arena mode: per-section free() already reclaimed the bytes
    const byPass = this.regionVertexArenas.get(regionKey);
    if (byPass) {
      for (const pass of TERRAIN_PASSES) byPass[pass]?.dispose();
      this.regionVertexArenas.delete(regionKey);
    }
    const ia = this.regionIndexArenas.get(regionKey);
    if (ia) {
      ia.dispose();
      this.regionIndexArenas.delete(regionKey);
    }
  }

  /** The uploader's OWN per-frame byte budget (Infinity ⇒ unbounded). INFRA-5: when an UploadScheduler is
   *  the budget authority, this MUST be Infinity so throughput isn't gated in two places. */
  get budgetBytes(): number {
    return this.budgetBytesPerFrame;
  }

  /** Reset the per-frame upload budget. Optionally override for this frame (tests/tuning). */
  beginFrame(budgetBytes: number = this.budgetBytesPerFrame): void {
    this.remainingBytes = budgetBytes;
  }

  upload(out: SectionBuildOutput, prev?: CommittedSection): CommittedSection | null {
    // TRAP 12.D (oldDataMatches): if the rebuilt translucent geometry hashes identical to the
    // previous section's, KEEP the old translucent vertex allocation + index buffer (and the carried
    // quads + sort/trigger state) instead of re-uploading. Opaque-only/non-geometric rebuilds pay
    // nothing extra for translucent.
    const p = prev as GpuCommittedSection | undefined;
    const reuseT = !!(p && out.translucent && p.vertexAlloc[Pass.Translucent] !== undefined && p.quadHash !== undefined && p.quadHash === out.translucent.quadHash);
    const tBytes = out.translucent ? out.translucent.vertexData.byteLength + (out.translucent.indexData?.byteLength ?? 0) : 0;

    const cost = out.approxBytes - (reuseT ? tBytes : 0);
    if (cost > this.remainingBytes) return null; // over budget — allocate NOTHING, retry next frame
    this.remainingBytes -= cost;

    // MEM-1: the section's region key (single-arena mode ignores it). Coord-derived from `out.sectionKey`
    // (already present) — no plumbing of a new field. Off ⇒ `arenaFor`/`indexArenaFor` ignore it.
    const [sx, sy, sz] = parseSectionKey(out.sectionKey);
    const regionKey = packRegionOfSection(sx, sy, sz);

    const vertexAlloc: Partial<Record<TerrainPass, AllocationId>> = {};
    const vertexArena: Partial<Record<TerrainPass, BufferArena>> = {};
    const indexAlloc: Partial<Record<TerrainPass, AllocationId>> = {};
    const indexArena: Partial<Record<TerrainPass, BufferArena>> = {};
    const quadCount: Partial<Record<TerrainPass, number>> = {};
    const facingVertexCounts: Partial<Record<TerrainPass, Uint32Array>> = {}; // FS-3 (opaque/cutout only)

    for (const pass of TERRAIN_PASSES) {
      let vertexData: ArrayBuffer | undefined;
      let qCount = 0;
      let indexData: ArrayBuffer | undefined;

      if (pass === Pass.Translucent) {
        if (reuseT) continue; // handled below by moving the old allocation across
        if (out.translucent) {
          vertexData = out.translucent.vertexData;
          qCount = out.translucent.quadCount;
          indexData = out.translucent.indexData;
        }
      } else {
        const part = out.parts[pass];
        if (part) {
          vertexData = part.vertexData;
          qCount = part.quadCount;
          if (part.facingVertexCounts) facingVertexCounts[pass] = part.facingVertexCounts; // FS-2 facing runs
        }
      }

      if (!vertexData) continue;
      const arena = this.arenaFor(pass, regionKey);
      vertexAlloc[pass] = arena.allocate(new Uint8Array(vertexData));
      vertexArena[pass] = arena;
      quadCount[pass] = qCount;
      if (indexData) {
        const ia = this.indexArenaFor(regionKey);
        indexAlloc[pass] = ia.allocate(new Uint8Array(indexData));
        indexArena[pass] = ia;
      }
    }

    const committed: GpuCommittedSection = { generation: out.generation, vertexAlloc, vertexArena, indexAlloc, indexArena, quadCount };
    for (const _ in facingVertexCounts) { committed.facingVertexCounts = facingVertexCounts; break; } // attach iff any pass partitioned
    if (reuseT && p && out.translucent) {
      // Transfer ownership of the unchanged translucent allocation + index buffer from prev → next (so
      // free(prev) skips them) and keep prev's carried quads (geometry is identical, still valid).
      vertexAlloc[Pass.Translucent] = p.vertexAlloc[Pass.Translucent];
      vertexArena[Pass.Translucent] = p.vertexArena[Pass.Translucent];
      indexAlloc[Pass.Translucent] = p.indexAlloc[Pass.Translucent];
      indexArena[Pass.Translucent] = p.indexArena[Pass.Translucent];
      quadCount[Pass.Translucent] = p.quadCount[Pass.Translucent];
      delete p.vertexAlloc[Pass.Translucent];
      delete p.vertexArena[Pass.Translucent];
      delete p.indexAlloc[Pass.Translucent];
      delete p.indexArena[Pass.Translucent];
      committed.sortType = p.sortType;
      committed.sortQuads = p.sortQuads;
      committed.quadHash = p.quadHash;
      committed.translucentVertexData = p.translucentVertexData;
    } else if (out.translucent) {
      committed.sortType = out.translucent.sortType;
      committed.sortQuads = out.translucent.quads; // CPU objects, owned by the renderer after commit
      committed.quadHash = out.translucent.quadHash;
      committed.translucentVertexData = out.translucent.vertexData;
    }
    // U8 reclaim — SAFE only here: a successful upload means this commit changes `presented`, so the viewer
    // rebuilds the draw list THIS frame and re-resolves every section's (relocated) arena buffer + offset.
    // Compacting outside a commit would strand the cached draw list against a destroyed buffer (TRAP 2.B).
    // INFRA-1 cross-arena: coalesce every arena's compaction copy into ONE encoder + submit this commit (a
    // multi-pass reclaim relocates several arenas). The batch also defers each old buffer's destroy until after
    // the submit, so a copy source outlives the GPU read. No-op on backends without an encoder model (WebGL2).
    this.device.beginCopyBatch?.();
    for (const a of this.vertexArenas()) if (a.shouldReclaim(RECLAIM_WASTE_BYTES)) a.compact();
    for (const ia of this.indexArenas()) if (ia.shouldReclaim(RECLAIM_WASTE_BYTES)) ia.compact();
    this.device.endCopyBatch?.();
    return committed;
  }

  /**
   * P6/LM-2 — light-only in-place reupload. A pure light change re-meshes to identical geometry with new
   * lightmap bytes; rather than free+alloc (which would relocate the section, churning the draw list and
   * the F1 bundle and growing the arena), this UPDATES each pass's existing vertex allocation in place
   * (same size → `arena.update`, no relocation → `layoutRevision` unchanged). Indices, sort state, and the
   * carried CPU quads are untouched (geometry is identical). Returns the same-identity `prev` on success;
   * `"mismatch"` if any pass's geometry actually differs (caller full-uploads); `null` if over budget.
   */
  uploadLight(prev: CommittedSection, out: SectionBuildOutput): CommittedSection | "mismatch" | null {
    const p = prev as GpuCommittedSection;
    // Geometry must be byte-for-byte the same SHAPE (same quad count + an allocation to update) per pass —
    // else this isn't a pure light change and the caller must take the full path.
    let cost = 0;
    for (const pass of TERRAIN_PASSES) {
      const data = pass === Pass.Translucent ? out.translucent?.vertexData : out.parts[pass]?.vertexData;
      const newQ = pass === Pass.Translucent ? (out.translucent?.quadCount ?? 0) : (out.parts[pass]?.quadCount ?? 0);
      const oldQ = p.quadCount[pass] ?? 0;
      if (newQ !== oldQ) return "mismatch";
      if (newQ > 0 && (p.vertexAlloc[pass] === undefined || !data)) return "mismatch";
      if (data) cost += data.byteLength;
    }
    if (cost > this.remainingBytes) return null; // over budget — keep old light, retry next frame
    this.remainingBytes -= cost;

    for (const pass of TERRAIN_PASSES) {
      const data = pass === Pass.Translucent ? out.translucent?.vertexData : out.parts[pass]?.vertexData;
      const alloc = p.vertexAlloc[pass];
      const arena = p.vertexArena[pass];
      if (data && alloc !== undefined && arena) arena.update(alloc, new Uint8Array(data)); // same size → in place
    }
    // The transient moving-glass merger reads the retained CPU translucent vertices — refresh them so it
    // pulls the NEW light. quadHash MUST refresh too: the lightmap bytes are part of the hashed vertex
    // stream (fnv1a over the whole 20B vertex), so the in-place light update CHANGED the hashed contents.
    // Leaving the old hash would let a later TRAP-12.D reuse (`upload` reuseT = quadHash match) keep this
    // stale-light buffer when a rebuild happens to revert to the prior light level — rendering wrong light.
    // geometry/quads/sortType are unchanged (light never moves a vertex), so those stay valid as-is.
    if (out.translucent) {
      // P1.5: write the new light bytes IN PLACE into the existing translucent CPU buffer, PRESERVING its
      // identity. That buffer is `cpuVertex` — the key for the DYNAMIC sort-trigger WeakMaps
      // (TerrainRenderer.sortState / SchematicViewer.sortTrigger). A pure-light change never moves a vertex,
      // so the sort planes + order are unchanged; preserving the key keeps the cached trigger state and avoids
      // a spurious unconditional "first-sight" re-sort on every block-light flicker near glass/water (redstone,
      // lava glow, per-block day-night). The full geometry `upload` path still mints a NEW buffer, which
      // correctly resets the trigger state when the geometry (and thus the sort order) actually changes.
      const old = p.translucentVertexData;
      const next = out.translucent.vertexData;
      if (old && old.byteLength === next.byteLength) new Uint8Array(old).set(new Uint8Array(next)); // same shape → in place
      else p.translucentVertexData = next; // size changed unexpectedly → replace (resets the state; still correct)
      p.quadHash = out.translucent.quadHash;
    }
    p.generation = out.generation;
    return p; // same identity (allocations reused) → no relocation, draw (buffer, offset) unchanged
  }

  uploadSort(prev: CommittedSection, out: SortOutput): CommittedSection | null {
    // Index-only (Invariant rule 5). A re-sort reorders the SAME quads → the new index is the same byte
    // length, so update the existing index-arena allocation IN PLACE (U6) — no free/alloc, vertices
    // untouched. (Defensive fallback to free+alloc if a size ever differs.)
    if (out.indexData.byteLength > this.remainingBytes) return null;
    this.remainingBytes -= out.indexData.byteLength;

    const p = prev as GpuCommittedSection;
    const ia = p.indexAlloc[Pass.Translucent];
    const arena = p.indexArena[Pass.Translucent];
    const data = new Uint8Array(out.indexData);
    if (ia !== undefined && arena && arena.rangeOf(ia).sizeBytes === align4(data.byteLength)) {
      arena.update(ia, data); // U6 — same size, in place
    } else {
      if (ia !== undefined && arena) arena.free(ia);
      // MEM-1: re-allocate into the SAME region's index arena (single-arena mode ignores the key). The
      // fallback is rare (a re-sort that changed index size); deriving the region from the section key
      // keeps the slice in its region so a later disposeRegion reclaims it.
      const [sx, sy, sz] = parseSectionKey(out.sectionKey);
      const dst = this.indexArenaFor(packRegionOfSection(sx, sy, sz));
      p.indexAlloc[Pass.Translucent] = dst.allocate(data);
      p.indexArena[Pass.Translucent] = dst;
    }
    return p; // vertex allocations reused unchanged
  }

  free(c: CommittedSection): void {
    const g = c as GpuCommittedSection;
    for (const pass of TERRAIN_PASSES) {
      const a = g.vertexAlloc[pass];
      const arena = g.vertexArena[pass];
      if (a !== undefined && arena) arena.free(a);
      const ia = g.indexAlloc[pass];
      const iarena = g.indexArena[pass];
      if (ia !== undefined && iarena) iarena.free(ia);
    }
  }

  /** Sum of every arena's relocation counter (F3). Bumps when any pass/index arena grew or compacted —
   *  i.e. when cached (buffer, offset) draw snapshots went stale (TRAP 4.A). Summing is monotonic: each
   *  arena's counter only increases, so the sum changes iff at least one arena relocated since last read. */
  layoutRevision(): number {
    let rev = 0;
    for (const a of this.vertexArenas()) rev += a.layoutRevision;
    for (const ia of this.indexArenas()) rev += ia.layoutRevision;
    return rev;
  }

  /** Release the shared arenas (whole-renderer teardown). Per-section frees go through free(); per-region
   *  contiguous reclaim goes through disposeRegion(). This is the final catch-all over every live arena. */
  dispose(): void {
    for (const a of this.vertexArenas()) a.dispose();
    for (const ia of this.indexArenas()) ia.dispose();
  }
}
