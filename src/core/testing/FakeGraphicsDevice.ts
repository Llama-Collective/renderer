// GPU-less GraphicsDevice for unit tests. RENDERER_PLAN.md §4 (GATE 4.1), §5.
//
// This one is implemented FOR REAL (lightweight, in-memory) — the data-model and
// Stable-Presentation-Invariant tests need a working device without a GPU. It records
// operations so tests can assert e.g. "presented buffer was never destroyed while a stale
// build was discarded".

import type { ByteRange } from "../../types";
import type {
  BindingsDesc,
  BindingsHandle,
  BufferCopy,
  BufferDesc,
  BundleEncoder,
  ClearDesc,
  DeviceFeatures,
  GpuBufferHandle,
  GpuTextureHandle,
  GraphicsDevice,
  PassEncoder,
  PipelineDesc,
  PipelineHandle,
  RenderBundleDesc,
  RenderBundleHandle,
  RenderTarget,
  SamplerDesc,
  SamplerHandle,
  ShaderModuleHandle,
  TextureDesc,
  TextureRegion,
} from "../GraphicsDevice";
import { BackendKind, BufferUsage } from "../GraphicsDevice";

interface FakeBuffer {
  id: number;
  data: Uint8Array;
  usage: BufferUsage;
  label?: string;
  /** True only after a REAL release (destroyBuffer with the pool off, or a pool eviction). `resolve()` throws
   *  on a destroyed buffer — the use-after-destroy guard. */
  destroyed: boolean;
  /** MEM-2: true while the buffer object is sitting IN the pool (returned but not destroyed). A DISTINCT state
   *  from `destroyed` so liveBufferCount() still counts it (pooled === retained GPU memory) yet `resolve()`
   *  rejects touching it while pooled. Cleared when it is recycled by a later createBuffer. */
  pooled: boolean;
}

// The one sanctioned handle-brand bridge for the Fake backend (STYLE_GUIDE §3 — handles are opaque brands
// whose runtime shape IS the backend's record; these are the only `as unknown as` casts allowed, kept in
// one pair instead of scattered through every create*/resolve method like the WebGPU backend's handleOf/recOf).
function handleOf<H>(rec: object): H {
  return rec as unknown as H;
}
function recOf<R>(handle: object): R {
  return handle as unknown as R;
}

export interface FakeDeviceLog {
  /** REAL buffer allocations (a pool hit does NOT bump this — same semantics as WebGPUDevice.createBufferCalls). */
  buffersCreated: number;
  /** REAL .destroy() calls (pool-return does NOT bump; an eviction DOES — mirrors WebGPU). */
  buffersDestroyed: number;
  // ── MEM-2 pool counters (all 0 when the pool is off) ──
  /** A createBuffer satisfied by a recycled object (no real alloc). */
  poolHits: number;
  /** A destroyBuffer that retained the buffer in the pool instead of destroying it. */
  poolReturns: number;
  /** A pooled buffer evicted (real .destroy()) because a usage bucket hit its count/byte cap. */
  poolEvictions: number;
  writes: number;
  /** Per-range GPU copies (one per relocated block). A batched copy of N blocks bumps this by N. */
  copies: number;
  /** queue.submit calls for copies (INFRA-1). copyBuffer ⇒ +1; copyBufferBatch ⇒ +1 regardless of N. */
  submits: number;
  drawCalls: number;
  /** F1: render bundles recorded + replayed (so tests can assert the adaptive record/replay state machine). */
  bundlesCreated: number;
  bundlesExecuted: number;
  /** Per-drawIndexed args recorded from the LIVE pass encoder (FS-4: assert per-facing-slice baseVertex/
   *  indexCount/firstInstance). Bundle-encoder draws aren't recorded here (they have no live args to check). */
  drawCmds: { indexCount: number; firstIndex: number; baseVertex: number; firstInstance: number }[];
  /** LIVE-pass `setVertexBuffer` calls (bind-once: assert the arena binds once per handle, not per draw). */
  vertexBinds: number;
  /** LIVE-pass pipeline LABEL bound for each `drawIndexed` (parallel to `drawCmds`). Lets a test assert the
   *  draw ORDER across pipelines — e.g. the depth-writing moving-translucent draw precedes terrain translucent. */
  drawPipelines: string[];
}

// MEM-2: caps mirror WebGPUDevice so the GPU-free pool behaves identically to the real one (bounded by count
// AND bytes; evict largest/oldest).
const POOL_MAX_COUNT_PER_USAGE = 8;
const POOL_MAX_BYTES_PER_USAGE = 64 * 1024 * 1024;

export class FakeGraphicsDevice implements GraphicsDevice {
  readonly backend = BackendKind.Fake;
  readonly features: DeviceFeatures = { multiDraw: true, timerQuery: false, gpuCopy: true };

  readonly log: FakeDeviceLog = {
    buffersCreated: 0,
    buffersDestroyed: 0,
    poolHits: 0,
    poolReturns: 0,
    poolEvictions: 0,
    writes: 0,
    copies: 0,
    submits: 0,
    drawCalls: 0,
    bundlesCreated: 0,
    bundlesExecuted: 0,
    drawCmds: [],
    vertexBinds: 0,
    drawPipelines: [],
  };

  private nextId = 1;
  private readonly buffers = new Map<number, FakeBuffer>();
  /** Pipeline descriptors by handle id — so the live pass can resolve a bound pipeline back to its label/state
   *  (e.g. assert the moving-translucent draw uses the depth-WRITE pipeline and is drawn before terrain). */
  private readonly pipelineDescs = new Map<number, PipelineDesc>();

  /** MEM-2: the same default-OFF device-level pool the WebGPU backend has, so the GPU-free arena/T-PRES tests
   *  exercise the real pool semantics. Bucketed by (usage, actual size). */
  private readonly bufferPoolEnabled: boolean;
  private readonly bufferPool = new Map<BufferUsage, FakeBuffer[]>();
  // INFRA-1 cross-arena copy batch (mirrors WebGPUDevice): while open, copies count toward ONE submit and
  // destroys defer until endCopyBatch (after the submit), so the `submits` gauge reflects the coalescing.
  private copyBatchActive = false;
  private copyBatchPendingSubmit = false;
  private readonly copyBatchDeferredDestroys: GpuBufferHandle[] = [];

  constructor(opts: { bufferPool?: boolean } = {}) {
    this.bufferPoolEnabled = opts.bufferPool ?? false;
  }

  /** Test helper: number of buffers still alive (created and not REALLY destroyed). A pooled buffer is still
   *  alive (retained GPU memory) — MEM-2's liveBuffers semantics: only a real destroy/eviction decrements. */
  liveBufferCount(): number {
    let n = 0;
    for (const b of this.buffers.values()) if (!b.destroyed) n++;
    return n;
  }

  /** Test helper: read back a buffer's bytes. */
  read(h: GpuBufferHandle): Uint8Array {
    return this.resolve(h).data;
  }

  /** Test helper: the most-recently-created live buffer with `label` (or null). Lets a test inspect an
   *  internal uniform (e.g. the terrain frame uniform) without the renderer exposing its private handle. */
  bufferByLabel(label: string): GpuBufferHandle | null {
    let found: number | null = null;
    for (const [id, b] of this.buffers) if (!b.destroyed && !b.pooled && b.label === label) found = id;
    return found === null ? null : handleOf<GpuBufferHandle>({ id: found });
  }

  private resolve(h: GpuBufferHandle): FakeBuffer {
    const b = this.buffers.get(recOf<{ id: number }>(h).id);
    if (!b) throw new Error("FakeGraphicsDevice: unknown buffer handle");
    if (b.destroyed) throw new Error("FakeGraphicsDevice: buffer used after destroy");
    // MEM-2: a pooled buffer has been handed back; touching it via a stale handle is a use-after-free of a
    // different flavor. Distinct message so the guard tests can tell the two apart.
    if (b.pooled) throw new Error("FakeGraphicsDevice: buffer used while pooled (returned to the buffer pool)");
    return b;
  }

  createBuffer(desc: BufferDesc): GpuBufferHandle {
    // MEM-2: best-fit pool hit — recycle a freed buffer of the SAME usage whose ACTUAL data length already
    // fits. No new FakeBuffer, no buffersCreated bump (mirrors the real device's createBufferCalls semantics).
    if (this.bufferPoolEnabled) {
      const reused = this.takeFromPool(desc.usage, desc.sizeBytes);
      if (reused) {
        reused.pooled = false; // back in service; resolve() works again
        reused.label = desc.label;
        // The buffer KEEPS its actual (>= requested) data — the arena writes its [0, requested) slice before
        // reading, so the over-large tail is harmless and never zeroed (TRAP: pooled bytes are NOT zeroed).
        this.log.poolHits++;
        return handleOf<GpuBufferHandle>({ id: reused.id });
      }
    }
    const id = this.nextId++;
    this.buffers.set(id, {
      id,
      data: new Uint8Array(desc.sizeBytes),
      usage: desc.usage,
      label: desc.label,
      destroyed: false,
      pooled: false,
    });
    this.log.buffersCreated++;
    return handleOf<GpuBufferHandle>({ id });
  }

  /** MEM-2: best-fit (smallest actual size >= requested) within the usage bucket, or null. */
  private takeFromPool(usage: BufferUsage, requested: number): FakeBuffer | null {
    const bucket = this.bufferPool.get(usage);
    if (!bucket || bucket.length === 0) return null;
    let bestIdx = -1;
    let bestSize = Infinity;
    for (let i = 0; i < bucket.length; i++) {
      const s = bucket[i].data.length;
      if (s >= requested && s < bestSize) {
        bestIdx = i;
        bestSize = s;
        if (s === requested) break;
      }
    }
    if (bestIdx < 0) return null;
    const [rec] = bucket.splice(bestIdx, 1);
    return rec;
  }

  writeBuffer(h: GpuBufferHandle, offsetBytes: number, data: ArrayBufferView): void {
    const b = this.resolve(h);
    const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    b.data.set(view, offsetBytes);
    this.log.writes++;
  }

  copyBuffer(src: GpuBufferHandle, srcRange: ByteRange, dst: GpuBufferHandle, dstOffset: number): void {
    const s = this.resolve(src);
    const d = this.resolve(dst);
    // Mirror the real WebGPU backend's guard (WEBGPU_FINDINGS §4 / TRAP 2.B): copyBufferToBuffer requires
    // src !== dst, so the GPU-free invariant suite catches the same misuse the real device forbids.
    if (s === d) {
      throw new Error("copyBuffer: source and destination must differ — WebGPU copyBufferToBuffer requires src !== dst");
    }
    d.data.set(s.data.subarray(srcRange.offsetBytes, srcRange.offsetBytes + srcRange.sizeBytes), dstOffset);
    this.log.copies++;
    if (this.copyBatchActive) this.copyBatchPendingSubmit = true; // folded into the one batch submit
    else this.log.submits++;
  }

  copyBufferBatch(src: GpuBufferHandle, copies: readonly BufferCopy[], dst: GpuBufferHandle): void {
    if (copies.length === 0) return; // no-op, no submit (mirrors the real backend)
    const s = this.resolve(src);
    const d = this.resolve(dst);
    if (s === d) {
      throw new Error("copyBufferBatch: source and destination must differ — WebGPU copyBufferToBuffer requires src !== dst");
    }
    for (const c of copies) {
      d.data.set(s.data.subarray(c.srcOffset, c.srcOffset + c.size), c.dstOffset);
      this.log.copies++; // one per relocated block — same accounting as N separate copyBuffer calls
    }
    if (this.copyBatchActive) this.copyBatchPendingSubmit = true; // several arenas fold into one batch submit
    else this.log.submits++; // …else ONE submit for this whole batch (the INFRA-1 win)
  }

  beginCopyBatch(): void {
    this.copyBatchActive = true;
    this.copyBatchPendingSubmit = false;
  }

  endCopyBatch(): void {
    this.copyBatchActive = false;
    if (this.copyBatchPendingSubmit) { this.log.submits++; this.copyBatchPendingSubmit = false; } // one submit for all arenas
    if (this.copyBatchDeferredDestroys.length > 0) {
      const pending = this.copyBatchDeferredDestroys.splice(0);
      for (const h of pending) this.destroyBuffer(h); // after the submit (mirrors WebGPUDevice src-lifetime rule)
    }
  }

  destroyBuffer(h: GpuBufferHandle): void {
    if (this.copyBatchActive) { this.copyBatchDeferredDestroys.push(h); return; } // defer past the batch submit
    const b = this.resolve(h);
    // MEM-2: pool ON ⇒ retain instead of destroy (no buffersDestroyed, no liveBufferCount change). Pool OFF ⇒
    // the exact pre-MEM-2 path: mark destroyed + bump buffersDestroyed.
    if (this.bufferPoolEnabled) {
      this.returnToPool(b);
      return;
    }
    b.destroyed = true;
    this.log.buffersDestroyed++;
  }

  /** MEM-2: retain into the usage bucket (capped by count AND bytes); evict largest-first (oldest on ties) with
   *  a REAL destroy so liveBufferCount only drops on a true release. */
  private returnToPool(b: FakeBuffer): void {
    b.pooled = true;
    let bucket = this.bufferPool.get(b.usage);
    if (!bucket) {
      bucket = [];
      this.bufferPool.set(b.usage, bucket);
    }
    bucket.push(b);
    this.log.poolReturns++;
    let bytes = 0;
    for (const r of bucket) bytes += r.data.length;
    while (bucket.length > POOL_MAX_COUNT_PER_USAGE || (bucket.length > 0 && bytes > POOL_MAX_BYTES_PER_USAGE)) {
      let evictIdx = 0;
      let evictSize = bucket[0].data.length;
      for (let i = 1; i < bucket.length; i++) {
        if (bucket[i].data.length > evictSize) {
          evictIdx = i;
          evictSize = bucket[i].data.length;
        }
      }
      const [evicted] = bucket.splice(evictIdx, 1);
      bytes -= evicted.data.length;
      evicted.pooled = false;
      evicted.destroyed = true; // real release ⇒ liveBufferCount drops here, not on the pool-return
      this.log.buffersDestroyed++;
      this.log.poolEvictions++;
    }
  }

  // Textures / pipelines / bindings: enough to satisfy the interface for tests.
  createTexture(_desc: TextureDesc): GpuTextureHandle {
    return handleOf<GpuTextureHandle>({ id: this.nextId++ });
  }
  writeTexture(_h: GpuTextureHandle, _data: ArrayBufferView | ImageBitmap, _mip?: number, _region?: TextureRegion): void {}
  destroyTexture(_h: GpuTextureHandle): void {}
  createSampler(_desc?: SamplerDesc): SamplerHandle {
    return handleOf<SamplerHandle>({ id: this.nextId++ });
  }

  createShaderModule(_code: string, _label?: string): ShaderModuleHandle {
    return handleOf<ShaderModuleHandle>({ id: this.nextId++ });
  }
  createPipeline(desc: PipelineDesc): PipelineHandle {
    const id = this.nextId++;
    this.pipelineDescs.set(id, desc);
    return handleOf<PipelineHandle>({ id });
  }

  /** Test helper: the descriptor a pipeline handle was created with (for asserting depth-write / blend state). */
  pipelineDesc(h: PipelineHandle): PipelineDesc | undefined {
    return this.pipelineDescs.get(recOf<{ id: number }>(h).id);
  }

  /** Test helper: the most-recently created pipeline descriptor with `label` (or undefined). */
  pipelineDescByLabel(label: string): PipelineDesc | undefined {
    let found: PipelineDesc | undefined;
    for (const d of this.pipelineDescs.values()) if (d.label === label) found = d;
    return found;
  }
  createBindings(_desc: BindingsDesc): BindingsHandle {
    return handleOf<BindingsHandle>({ id: this.nextId++ });
  }

  createRenderBundle(_desc: RenderBundleDesc, record: (enc: BundleEncoder) => void): RenderBundleHandle {
    // Run the record callback to count its logical draws (so replay reproduces the drawCalls metric), but
    // hold no real commands — the GPU-free tests only need the record/replay bookkeeping.
    let drawCount = 0;
    record({
      setPipeline() {},
      setBindings() {},
      setVertexBuffer() {},
      setIndexBuffer() {},
      drawIndexed() {
        drawCount++;
      },
    });
    this.log.bundlesCreated++;
    return handleOf<RenderBundleHandle>({ drawCount });
  }

  beginPass(_target: RenderTarget, _clear: ClearDesc): PassEncoder {
    const log = this.log;
    const descs = this.pipelineDescs;
    let curPipeline = "";
    return {
      setPipeline(p: PipelineHandle) { curPipeline = descs.get(recOf<{ id: number }>(p).id)?.label ?? ""; },
      setBindings() {},
      setVertexBuffer() { log.vertexBinds++; },
      setIndexBuffer() {},
      drawIndexed(indexCount: number, firstIndex: number, baseVertex: number, firstInstance = 0) {
        log.drawCalls++;
        log.drawCmds.push({ indexCount, firstIndex, baseVertex, firstInstance });
        log.drawPipelines.push(curPipeline);
      },
      multiDrawIndexed(counts) {
        log.drawCalls += counts.length;
        for (let i = 0; i < counts.length; i++) log.drawPipelines.push(curPipeline);
      },
      executeBundles(bundles) {
        for (const b of bundles) log.drawCalls += recOf<{ drawCount: number }>(b).drawCount;
        log.bundlesExecuted++;
      },
      end() {},
    };
  }

  destroy(): void {
    this.buffers.clear();
  }
}
