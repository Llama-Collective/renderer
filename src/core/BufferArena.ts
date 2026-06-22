// Per-region/pass GPU buffer arena. RENDERER_PLAN.md §11.
//
// Sub-allocates inside ONE large buffer instead of one buffer per section (TRAP 11.B). Free
// segments merge with their neighbors; fragmentation/overflow grows or compacts the buffer
// GPU-side via device.copyBuffer (src !== dst, into a fresh buffer — TRAP 2.B/11.A), never a CPU
// mirror. Vertex and index data use SEPARATE arenas so translucent re-sorting can swap indices
// without touching vertices (TRAP 11.C).
//
// Allocation is BEST-FIT over the free segments (`GlBufferArena.findFree` — less fragmentation
// than first-fit under mixed-size churn). The per-allocation offset/range lookups (`rangeOf`, the
// per-draw hot path in collectDraws) are O(1) via `byId`, NOT a linear block scan — otherwise a single
// global arena makes collectDraws O(sections²) (audit). `compact()` packs live data to the front of a
// SHRUNK fresh buffer (used + slack), reclaiming the high-water mark (U8). Allocation/free touch the
// block list (per-commit, not per-frame), so they stay simple linear merges.

import type { AllocationId, ByteRange } from "../types";
import type { BufferCopy, BufferUsage, GpuBufferHandle, GraphicsDevice } from "./GraphicsDevice";

export type { AllocationId };

export interface ArenaStats {
  capacityBytes: number;
  usedBytes: number;
  freeSegments: number;
  largestFreeBytes: number;
}

/** A contiguous span of the arena. `alloc === null` means free. Blocks tile [0, capacity). */
interface Block {
  offset: number;
  size: number;
  alloc: AllocationId | null;
}

const ALIGN = 4; // WebGPU copy/offset alignment; also keeps writes 4-byte aligned.

function alignUp(n: number): number {
  return (n + (ALIGN - 1)) & ~(ALIGN - 1);
}

/** INFRA-2: dev/test gate for the O(blocks) structural audit. Default OFF so production pays nothing —
 *  each allocate/free/grow/compact does ONE boolean test (per-commit, not per-draw), and the walk itself
 *  is never entered. A true const would be statically dead-code-eliminable but un-toggleable for tests;
 *  the runtime guard is the practical equivalent (the byId O(1) hot path is untouched either way). */
let ARENA_ASSERTS = false;
/** Test hook to enable/disable {@link BufferArena.checkInvariants} auto-running after each mutation. */
export function __setArenaAsserts(on: boolean): void {
  ARENA_ASSERTS = on;
}

export class BufferArena {
  private buffer: GpuBufferHandle;
  private capacity: number;
  private blocks: Block[];
  /** id → its (allocated) block, for O(1) `rangeOf`/`offsetOf` (the per-draw hot path). Kept in sync with
   *  `blocks` on every allocate/free/compact/grow. */
  private readonly byId = new Map<AllocationId, Block>();
  private readonly liveSize = new Map<AllocationId, number>();
  /** Live (allocated) bytes, tracked incrementally so `shouldReclaim` is O(1) (not a block scan). */
  private usedTotal = 0;
  private nextId = 1;
  /** Bumps on grow()/compact() — the ONLY ops that relocate existing allocations (new buffer handle AND
   *  changed offsets). A plain allocate/free/update never moves an existing allocation, so it doesn't bump.
   *  Lets an incremental draw-list cache (F3) detect when its cached (buffer, offset) snapshots went stale
   *  and force a full re-resolve (TRAP 4.A — the relocation the once-per-commit full rebuild used to hide). */
  private layoutRev = 0;

  constructor(
    private readonly device: GraphicsDevice,
    initialCapacityBytes: number,
    private readonly usage: BufferUsage,
  ) {
    this.capacity = Math.max(ALIGN, alignUp(initialCapacityBytes));
    this.buffer = device.createBuffer({ sizeBytes: this.capacity, usage, label: `arena:${usage}` });
    this.blocks = [{ offset: 0, size: this.capacity, alloc: null }];
  }

  /** Reserve a range and upload `data`. Returns a stable id usable in draw commands. */
  allocate(data: ArrayBufferView): AllocationId {
    const size = alignUp(data.byteLength);
    let bi = this.bestFitIndex(size);
    if (bi < 0) {
      this.grow(this.usedBytes() + size);
      bi = this.bestFitIndex(size);
      if (bi < 0) throw new Error("BufferArena: grow failed to make room");
    }
    const id = this.nextId++ as AllocationId;
    this.placeInto(bi, size, id);
    this.liveSize.set(id, size);
    this.usedTotal += size;
    this.device.writeBuffer(this.buffer, this.offsetOf(id), data);
    this.audit();
    return id;
  }

  /** Overwrite an allocation's contents (must be the same byte length as allocated). U6. */
  update(id: AllocationId, data: ArrayBufferView): void {
    const size = this.liveSize.get(id);
    if (size === undefined) throw new Error(`BufferArena.update: unknown allocation ${id}`);
    if (alignUp(data.byteLength) !== size) throw new Error("BufferArena.update: size mismatch");
    this.device.writeBuffer(this.buffer, this.offsetOf(id), data);
  }

  free(id: AllocationId): void {
    const b = this.byId.get(id);
    if (!b) return;
    b.alloc = null;
    this.byId.delete(id);
    this.usedTotal -= this.liveSize.get(id) ?? b.size;
    this.liveSize.delete(id);
    this.mergeFreeAround(this.blocks.indexOf(b));
    this.audit();
  }

  /** True iff compaction would reclaim at least `minWaste` bytes AND the arena is mostly empty — the U8
   *  reclaim heuristic. O(1). After a compaction the capacity shrinks to fit, so this won't re-fire until
   *  the arena grows then empties again (no thrash). */
  shouldReclaim(minWaste: number): boolean {
    return this.capacity - this.usedTotal >= minWaste && this.usedTotal * 2 < this.capacity;
  }

  /** Byte range of an allocation, for building draw commands. O(1). Reads off the block in hand — for an
   *  allocated block `b.size` always equals the stored `liveSize` (both set to the aligned request in
   *  `placeInto`; `update` forbids a resize and grow/compact copy `b.size`), so no second map lookup. */
  rangeOf(id: AllocationId): ByteRange {
    const b = this.byId.get(id);
    if (!b) throw new Error(`BufferArena.rangeOf: unknown allocation ${id}`);
    return { offsetBytes: b.offset, sizeBytes: b.size };
  }

  /** The underlying buffer (rebound after a compaction/grow — re-read after those). */
  get gpuBuffer(): GpuBufferHandle {
    return this.buffer;
  }

  /** Monotonic counter bumped whenever existing allocations were relocated (grow/compact) — see `layoutRev`. */
  get layoutRevision(): number {
    return this.layoutRev;
  }

  /**
   * Pack live allocations to the front of a FRESH, SHRUNK buffer (src !== dst is required by WebGPU
   * copyBufferToBuffer — §4), preserving ids and contents, then drop the old buffer. Eliminates
   * fragmentation AND reclaims the high-water mark (capacity → used + ~25% slack) so a long churn-heavy
   * session doesn't hold a permanently-grown buffer (U8). `gpuBuffer`/`rangeOf` change — only safe to
   * call when the caller will re-resolve draws this frame (the uploader does, gated to commit frames).
   */
  compact(): void {
    const live = this.blocks.filter((b) => b.alloc !== null);
    const used = this.usedBytes();
    const newCap = Math.max(ALIGN, alignUp(used + (used >> 2))); // shrink to fit + 25% slack
    const dst = this.device.createBuffer({ sizeBytes: newCap, usage: this.usage, label: `arena:${this.usage}` });
    this.byId.clear();
    const newBlocks: Block[] = [];
    // Accumulate every live-block relocation, then copy them all in ONE submit (INFRA-1) — preserving the
    // exact per-block dstOffset==cursor ordering so the packed bytes are identical to K separate copies.
    const copies: BufferCopy[] = [];
    let cursor = 0;
    for (const b of live) {
      if (b.alloc === null) continue; // narrows b.alloc → AllocationId (the filter already excluded nulls; grow() does the same)
      const id = b.alloc;
      copies.push({ srcOffset: b.offset, size: b.size, dstOffset: cursor });
      const nb: Block = { offset: cursor, size: b.size, alloc: id };
      newBlocks.push(nb);
      this.byId.set(id, nb);
      cursor += b.size;
    }
    this.device.copyBufferBatch(this.buffer, copies, dst); // before destroyBuffer(this.buffer) — src still alive
    if (cursor < newCap) newBlocks.push({ offset: cursor, size: newCap - cursor, alloc: null });
    this.device.destroyBuffer(this.buffer);
    this.buffer = dst;
    this.capacity = newCap;
    this.blocks = newBlocks.length ? newBlocks : [{ offset: 0, size: newCap, alloc: null }];
    this.layoutRev++; // allocations relocated → stale any cached (buffer, offset) snapshot (F3/TRAP 4.A)
    this.audit();
  }

  stats(): ArenaStats {
    let used = 0;
    let freeSegments = 0;
    let largestFree = 0;
    for (const b of this.blocks) {
      if (b.alloc === null) {
        freeSegments++;
        largestFree = Math.max(largestFree, b.size);
      } else {
        used += b.size;
      }
    }
    return { capacityBytes: this.capacity, usedBytes: used, freeSegments, largestFreeBytes: largestFree };
  }

  dispose(): void {
    this.device.destroyBuffer(this.buffer);
    this.blocks = [];
    this.byId.clear();
    this.liveSize.clear();
    this.usedTotal = 0;
  }

  // --- internals ------------------------------------------------------------------
  private usedBytes(): number {
    return this.usedTotal; // O(1) — kept in sync on allocate/free
  }

  private offsetOf(id: AllocationId): number {
    const b = this.byId.get(id);
    if (!b) throw new Error(`BufferArena.offsetOf: unknown allocation ${id}`);
    return b.offset;
  }

  /** Smallest free block that fits `size` (best-fit — findFree), or -1. */
  private bestFitIndex(size: number): number {
    let best = -1;
    let bestSize = Infinity;
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.alloc === null && b.size >= size && b.size < bestSize) {
        best = i;
        bestSize = b.size;
        if (b.size === size) break; // exact fit — can't do better
      }
    }
    return best;
  }

  /** Claim `size` bytes at the front of free block `i`, splitting off any remainder. */
  private placeInto(i: number, size: number, id: AllocationId): void {
    const b = this.blocks[i];
    if (b.size !== size) {
      const remainder: Block = { offset: b.offset + size, size: b.size - size, alloc: null };
      b.size = size;
      this.blocks.splice(i + 1, 0, remainder);
    }
    b.alloc = id;
    this.byId.set(id, b);
  }

  private mergeFreeAround(i: number): void {
    if (i < 0) return;
    // Merge with next, then previous, so adjacent free space coalesces.
    if (i + 1 < this.blocks.length && this.blocks[i + 1].alloc === null) {
      this.blocks[i].size += this.blocks[i + 1].size;
      this.blocks.splice(i + 1, 1);
    }
    if (i - 1 >= 0 && this.blocks[i - 1].alloc === null) {
      this.blocks[i - 1].size += this.blocks[i].size;
      this.blocks.splice(i, 1);
    }
  }

  /** Grow to hold at least `needed` live bytes, copying existing live data into a fresh buffer. */
  private grow(needed: number): void {
    const newCap = Math.max(alignUp(needed), this.capacity * 2);
    const dst = this.device.createBuffer({ sizeBytes: newCap, usage: this.usage, label: `arena:${this.usage}` });
    this.byId.clear();
    const newBlocks: Block[] = [];
    // One submit for all live-block relocations (INFRA-1); dstOffset==cursor ordering preserved byte-for-byte.
    const copies: BufferCopy[] = [];
    let cursor = 0;
    for (const b of this.blocks) {
      if (b.alloc === null) continue;
      copies.push({ srcOffset: b.offset, size: b.size, dstOffset: cursor });
      const nb: Block = { offset: cursor, size: b.size, alloc: b.alloc };
      newBlocks.push(nb);
      this.byId.set(b.alloc, nb);
      cursor += b.size;
    }
    this.device.copyBufferBatch(this.buffer, copies, dst); // before destroyBuffer(this.buffer) — src still alive
    newBlocks.push({ offset: cursor, size: newCap - cursor, alloc: null });
    this.device.destroyBuffer(this.buffer);
    this.buffer = dst;
    this.capacity = newCap;
    this.blocks = newBlocks;
    this.layoutRev++; // allocations relocated → stale any cached (buffer, offset) snapshot (F3/TRAP 4.A)
    this.audit();
  }

  /** INFRA-2: run the structural audit iff asserts are enabled. Off ⇒ one boolean test, no walk. */
  private audit(): void {
    if (ARENA_ASSERTS) this.checkInvariants();
  }

  /**
   * Dev/test structural audit (INFRA-2). Walks the block list ONCE and throws on the first
   * inconsistency a `placeInto`/`mergeFreeAround`/`grow`/`compact` splice bug could introduce:
   * blocks must tile `[0, capacity)` with no gap/overlap; no two adjacent free blocks (they must
   * have merged); every allocated block round-trips through `byId`; `liveSize` matches the block
   * size; and `usedTotal`/`byId.size` match the live total. Safe to call directly from tests; the
   * mutators only invoke it when `__setArenaAsserts(true)` (production never enters the walk).
   */
  checkInvariants(): void {
    let cursor = 0;
    let used = 0;
    let liveBlocks = 0;
    let prevFree = false;
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      if (b.offset !== cursor) throw new Error(`BufferArena invariant: block ${i} offset ${b.offset} != expected ${cursor} (gap/overlap)`);
      if (b.size <= 0) throw new Error(`BufferArena invariant: block ${i} has non-positive size ${b.size}`);
      if (b.alloc === null) {
        if (prevFree) throw new Error(`BufferArena invariant: adjacent free blocks at index ${i} (should have merged)`);
        prevFree = true;
      } else {
        prevFree = false;
        liveBlocks++;
        used += b.size;
        if (this.byId.get(b.alloc) !== b) throw new Error(`BufferArena invariant: byId[${b.alloc}] does not point at its block`);
        const ls = this.liveSize.get(b.alloc);
        if (ls !== b.size) throw new Error(`BufferArena invariant: liveSize ${ls} != block size ${b.size} for ${b.alloc}`);
      }
      cursor += b.size;
    }
    if (cursor !== this.capacity) throw new Error(`BufferArena invariant: blocks tile ${cursor} bytes != capacity ${this.capacity}`);
    if (used !== this.usedTotal) throw new Error(`BufferArena invariant: usedTotal ${this.usedTotal} != Σ live ${used}`);
    if (this.byId.size !== liveBlocks) throw new Error(`BufferArena invariant: byId.size ${this.byId.size} != live block count ${liveBlocks}`);
  }
}
