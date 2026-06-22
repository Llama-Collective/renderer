/// <reference types="@webgpu/types" />
// WebGPU backend — the primary concrete GraphicsDevice. RENDERER_PLAN.md §2; WEBGPU_FINDINGS.md.
//
// Maps the backend-neutral GraphicsDevice onto WebGPU. Opaque handles wrap the real GPU objects;
// the ONLY casts that bridge handle<->record live in `handleOf`/`recOf` below. The neutral enums
// are translated to WebGPU's vocabulary by the exhaustive `GPU_*` tables — add an enum member and
// the table stops compiling until you map it.
//
// Key spec-grounded choices (see WEBGPU_FINDINGS):
//  - Buffer usage: vertex/index buffers get COPY_DST|COPY_SRC so they can receive uploads AND act
//    as compaction sources; mappable staging is a separate concern (§3-4).
//  - copyBuffer requires src !== dst (§4) — enforced here.
//  - Color chain is linear: the canvas is configured non-srgb with an `-srgb` viewFormat, and
//    every frame renders through the `-srgb` view so blending is linear (§17.2). Pipelines must
//    use `colorFormat` (exposed below) as their color target.
//  - Packed vertex formats map 1:1 (uint32x2/unorm8x4/unorm16x2/uint8x4); color/UV auto-decode (§6).
//  - No core multi-draw: multiDrawIndexed loops drawIndexed; render bundles + the Chrome
//    `chromium-experimental-multi-draw-indirect` extension are the real cache/optimization (§9).
//  - The swapchain carries COPY_SRC so `readCanvasPixels()` can screenshot it for the smoke
//    harness self-check and the §22 pixel-diff suite.

import type { ByteRange } from "../../types";
import type {
  BindingLayoutEntry,
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
  VertexAttribute,
} from "../GraphicsDevice";
import {
  AddressMode,
  BackendKind,
  BufferUsage,
  CompareFn,
  CullMode,
  FilterMode,
  IndexFormat,
  PrimitiveTopology,
  SamplerBindingType,
  ShaderStage,
  TextureFormat,
  TextureSampleType,
  VertexScalarKind,
} from "../gpu-enums";

// --- Handle backing records (opaque handles ARE these objects at runtime) ----------
interface BufferRec {
  buffer: GPUBuffer;
  /** ACTUAL allocated size of the GPUBuffer (>= the requested size on a pool hit — MEM-2 TRAP-B). The pool
   *  buckets + best-fits on THIS, never the requested size, so a recycled buffer is never handed back smaller
   *  than asked for. */
  sizeBytes: number;
  /** The usage the buffer was created with — part of the pool bucket KEY (MEM-2 TRAP-A: a Vertex buffer's
   *  flags differ from an Index buffer's, so it can never be reused across usages). */
  usage: BufferUsage;
}
interface TextureRec {
  texture: GPUTexture;
  view: GPUTextureView;
  width: number;
  height: number;
  bytesPerTexel: number;
}
interface ShaderRec {
  module: GPUShaderModule;
}
interface SamplerRec {
  sampler: GPUSampler;
}
interface PipelineRec {
  pipeline: GPURenderPipeline;
}
interface BindingsRec {
  group: number;
  bindGroup: GPUBindGroup;
}
interface BundleRec {
  bundle: GPURenderBundle;
  /** Logical draws recorded — added to `drawCalls` on replay so the metric counts draws whether they
   *  were encoded directly or replayed from a bundle (F1). */
  drawCount: number;
}

/**
 * The two — and ONLY two — places the handle brand is bridged. `handleOf` hands a backing record
 * out as an opaque handle; `recOf` recovers it. Centralizing the `as unknown as` here keeps every
 * device method (and the pass encoder) cast-free (cleanup item 1).
 */
function handleOf<H>(rec: object): H {
  return rec as unknown as H;
}
function recOf<R>(handle: object): R {
  return handle as unknown as R;
}

// --- Neutral-enum -> WebGPU-vocabulary tables (exhaustive; a missing member won't compile) ------
const GPU_TEXTURE_FORMAT: Record<TextureFormat, GPUTextureFormat> = {
  [TextureFormat.Rgba8]: "rgba8unorm",
  [TextureFormat.Rgba8Srgb]: "rgba8unorm-srgb",
  [TextureFormat.R8]: "r8unorm",
  [TextureFormat.Bgra8]: "bgra8unorm",
  [TextureFormat.Bgra8Srgb]: "bgra8unorm-srgb",
  [TextureFormat.Depth24Plus]: "depth24plus",
  [TextureFormat.Depth32Float]: "depth32float",
};
const GPU_COMPARE: Record<CompareFn, GPUCompareFunction> = {
  [CompareFn.Less]: "less",
  [CompareFn.LessEqual]: "less-equal",
  [CompareFn.Greater]: "greater",
  [CompareFn.GreaterEqual]: "greater-equal",
  [CompareFn.Always]: "always",
};
const GPU_FILTER: Record<FilterMode, GPUFilterMode> = {
  [FilterMode.Nearest]: "nearest",
  [FilterMode.Linear]: "linear",
};
const GPU_ADDRESS: Record<AddressMode, GPUAddressMode> = {
  [AddressMode.Clamp]: "clamp-to-edge",
  [AddressMode.Repeat]: "repeat",
  [AddressMode.Mirror]: "mirror-repeat",
};
const GPU_CULL: Record<CullMode, GPUCullMode> = {
  [CullMode.None]: "none",
  [CullMode.Back]: "back",
  [CullMode.Front]: "front",
};
const GPU_INDEX: Record<IndexFormat, GPUIndexFormat> = {
  [IndexFormat.Uint16]: "uint16",
  [IndexFormat.Uint32]: "uint32",
};
const GPU_TOPOLOGY: Record<PrimitiveTopology, GPUPrimitiveTopology> = {
  [PrimitiveTopology.TriangleList]: "triangle-list",
  [PrimitiveTopology.LineList]: "line-list",
};
const GPU_SAMPLE_TYPE: Record<TextureSampleType, GPUTextureSampleType> = {
  [TextureSampleType.Float]: "float",
  [TextureSampleType.Uint]: "uint",
};
const GPU_SAMPLER_TYPE: Record<SamplerBindingType, GPUSamplerBindingType> = {
  [SamplerBindingType.Filtering]: "filtering",
  [SamplerBindingType.NonFiltering]: "non-filtering",
};
/** Base spelling of a vertex scalar; the full `GPUVertexFormat` appends `x{components}` if >1. */
const GPU_SCALAR: Record<VertexScalarKind, string> = {
  [VertexScalarKind.Uint8]: "uint8",
  [VertexScalarKind.Uint16]: "uint16",
  [VertexScalarKind.Uint32]: "uint32",
  [VertexScalarKind.Float32]: "float32",
  [VertexScalarKind.Snorm8]: "snorm8",
  [VertexScalarKind.Unorm8]: "unorm8",
  [VertexScalarKind.Snorm16]: "snorm16",
  [VertexScalarKind.Unorm16]: "unorm16",
};
// These two tables reference the WebGPU VALUE globals `GPUBufferUsage`/`GPUShaderStage`, which only
// exist in a browser. Build them LAZILY (on first use) so merely IMPORTING this module is safe under
// SSR/prerender in Node (Next renders the client page on the server) — see RENDERER_INTEGRATION_PLAN.
let _gpuBufferUsage: Record<BufferUsage, GPUBufferUsageFlags> | null = null;
function gpuBufferUsage(): Record<BufferUsage, GPUBufferUsageFlags> {
  return (_gpuBufferUsage ??= {
    // Vertex/index buffers double as arena compaction sources, hence COPY_SRC too.
    [BufferUsage.Vertex]: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    [BufferUsage.Index]: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    [BufferUsage.Uniform]: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    [BufferUsage.Storage]: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    [BufferUsage.Copy]: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}
let _gpuShaderStage: Record<ShaderStage, GPUShaderStageFlags> | null = null;
function gpuShaderStage(): Record<ShaderStage, GPUShaderStageFlags> {
  return (_gpuShaderStage ??= {
    [ShaderStage.Vertex]: GPUShaderStage.VERTEX,
    [ShaderStage.Fragment]: GPUShaderStage.FRAGMENT,
    [ShaderStage.Compute]: GPUShaderStage.COMPUTE,
  });
}

// --- MEM-2: freed-GPUBuffer-object pool caps (default-OFF; see WebGPUDevice.bufferPoolEnabled) -----
// Bound the pool in BOTH count AND bytes so a post-explosion giant buffer can't pin GPU memory forever
// (TRAP MEM-2.E). 8 is freeBuffers[8] number; the byte ceiling is per-usage.
const POOL_MAX_COUNT_PER_USAGE = 8;
const POOL_MAX_BYTES_PER_USAGE = 64 * 1024 * 1024; // 64 MiB of retained buffers per usage bucket

function isDepthFormat(f: TextureFormat): boolean {
  return f === TextureFormat.Depth24Plus || f === TextureFormat.Depth32Float;
}
function bytesPerTexel(f: TextureFormat): number {
  return f === TextureFormat.R8 ? 1 : 4;
}
function toGpuVertexFormat(a: VertexAttribute): GPUVertexFormat {
  const base = GPU_SCALAR[a.kind];
  return (a.components > 1 ? `${base}x${a.components}` : base) as GPUVertexFormat;
}

// --- Pass encoder -----------------------------------------------------------------
class WebGPUPassEncoder implements PassEncoder {
  constructor(
    private readonly gpu: GPUDevice,
    private readonly enc: GPUCommandEncoder,
    private readonly pass: GPURenderPassEncoder,
    /** Owning device — bumped per draw call for the Phase-P draw-call metric. */
    private readonly device: WebGPUDevice,
  ) {}

  setPipeline(p: PipelineHandle): void {
    this.pass.setPipeline(recOf<PipelineRec>(p).pipeline);
  }
  setBindings(b: BindingsHandle, dynamicOffsets?: number[]): void {
    const r = recOf<BindingsRec>(b);
    if (dynamicOffsets) this.pass.setBindGroup(r.group, r.bindGroup, dynamicOffsets);
    else this.pass.setBindGroup(r.group, r.bindGroup);
  }
  setVertexBuffer(slot: number, buf: GpuBufferHandle, offsetBytes = 0): void {
    this.pass.setVertexBuffer(slot, recOf<BufferRec>(buf).buffer, offsetBytes);
  }
  setIndexBuffer(buf: GpuBufferHandle, indexType: IndexFormat, offsetBytes = 0): void {
    this.pass.setIndexBuffer(recOf<BufferRec>(buf).buffer, GPU_INDEX[indexType], offsetBytes);
  }
  drawIndexed(indexCount: number, firstIndex: number, baseVertex: number, firstInstance = 0): void {
    this.device.drawCalls++;
    this.pass.drawIndexed(indexCount, 1, firstIndex, baseVertex, firstInstance);
  }
  multiDrawIndexed(counts: Int32Array, firstIndices: Int32Array, baseVertices: Int32Array): void {
    // No native multi-draw in core; emulate by looping. The render-bundle + Chrome
    // `chromium-experimental-multi-draw-indirect` path is the real optimization (§9).
    this.device.drawCalls += counts.length;
    for (let i = 0; i < counts.length; i++) {
      this.pass.drawIndexed(counts[i], 1, firstIndices[i], baseVertices[i], 0);
    }
  }
  executeBundles(bundles: RenderBundleHandle[]): void {
    // F1: replay pre-recorded draws. Count the bundle's logical draws so the drawCalls metric is the same
    // whether a frame encoded directly or replayed (the per-draw drawIndexed inside a bundle does NOT
    // touch drawCalls — it's counted here, at replay).
    const list: GPURenderBundle[] = [];
    for (const h of bundles) {
      const r = recOf<BundleRec>(h);
      list.push(r.bundle);
      this.device.drawCalls += r.drawCount;
    }
    this.pass.executeBundles(list);
  }
  end(): void {
    this.pass.end();
    this.gpu.queue.submit([this.enc.finish()]);
  }
}

/**
 * Records draw commands into a `GPURenderBundle` (F1). Shares the pass encoder's command vocabulary but
 * targets a bundle encoder; `drawIndexed` here does NOT bump `drawCalls` (replay does, in executeBundles).
 */
class WebGPURenderBundleEncoder implements BundleEncoder {
  drawCount = 0;
  constructor(private readonly enc: GPURenderBundleEncoder) {}
  setPipeline(p: PipelineHandle): void {
    this.enc.setPipeline(recOf<PipelineRec>(p).pipeline);
  }
  setBindings(b: BindingsHandle, dynamicOffsets?: number[]): void {
    const r = recOf<BindingsRec>(b);
    if (dynamicOffsets) this.enc.setBindGroup(r.group, r.bindGroup, dynamicOffsets);
    else this.enc.setBindGroup(r.group, r.bindGroup);
  }
  setVertexBuffer(slot: number, buf: GpuBufferHandle, offsetBytes = 0): void {
    this.enc.setVertexBuffer(slot, recOf<BufferRec>(buf).buffer, offsetBytes);
  }
  setIndexBuffer(buf: GpuBufferHandle, indexType: IndexFormat, offsetBytes = 0): void {
    this.enc.setIndexBuffer(recOf<BufferRec>(buf).buffer, GPU_INDEX[indexType], offsetBytes);
  }
  drawIndexed(indexCount: number, firstIndex: number, baseVertex: number, firstInstance = 0): void {
    this.drawCount++;
    this.enc.drawIndexed(indexCount, 1, firstIndex, baseVertex, firstInstance);
  }
}

// --- Device -----------------------------------------------------------------------
export class WebGPUDevice implements GraphicsDevice {
  readonly backend = BackendKind.WebGPU;
  readonly features: DeviceFeatures;

  private context: GPUCanvasContext | null = null;
  private canvasViewFormat: GPUTextureFormat = "bgra8unorm-srgb";
  private _colorFormat: TextureFormat = TextureFormat.Bgra8Srgb;
  private _depthFormat: TextureFormat = TextureFormat.Depth24Plus;
  private depthTexture: GPUTexture | null = null;
  private depthView: GPUTextureView | null = null;
  /** Render passes opened since construction (each = one command-buffer submit). Large-world metric. */
  passCount = 0;
  /** Draw calls issued since construction (Phase-P metric; callers read per-frame deltas). */
  drawCalls = 0;
  /** Live `GPUBuffer` objects (created − destroyed) — the O(sections×passes)→O(regions×passes) gauge.
   *  MEM-2 INVARIANT: still means "real GPUBuffers holding GPU memory" — it is bumped only on a real
   *  gpu.createBuffer and decremented only on a real .destroy() (a pooled-then-reused buffer never changes it). */
  liveBuffers = 0;
  /** MEM-2: cumulative count of REAL gpu.createBuffer calls (a pool hit does NOT bump it). Callers read the
   *  per-tick delta; with the pool on under churn this rises slower than `createBuffer` is logically called. */
  createBufferCalls = 0;
  /** MEM-2: cumulative count of pool hits (a `createBuffer` satisfied by a recycled object, no real alloc). */
  poolHits = 0;

  /** MEM-2: device-level freed-GPUBuffer-object pool, default OFF. Device-level ⇒ inherently cross-arena
   *  (every BufferArena in GpuSectionUploader shares this one device). Bucketed by (usage, actual size). When
   *  disabled, `createBuffer`/`destroyBuffer` run the EXACT pre-MEM-2 code. */
  private readonly bufferPoolEnabled: boolean;
  /** usage → recycled BufferRecs (each retains its ACTUAL sizeBytes). Best-fit on lookup, evict largest/oldest. */
  private readonly bufferPool = new Map<BufferUsage, BufferRec[]>();
  // INFRA-1 cross-arena copy batch (beginCopyBatch/endCopyBatch). While open, every copy encodes into this ONE
  // lazily-created encoder (one submit at end), and destroyBuffer is deferred so a copy SOURCE outlives the GPU
  // copy that reads it. Null encoder + inactive flag ⇒ the per-call submit path (byte-identical to before).
  private copyBatchActive = false;
  private copyBatchEncoder: GPUCommandEncoder | null = null;
  private readonly copyBatchDeferredDestroys: GpuBufferHandle[] = [];

  constructor(
    private readonly gpu: GPUDevice,
    adapter?: GPUAdapter,
    opts: { bufferPool?: boolean } = {},
  ) {
    this.bufferPoolEnabled = opts.bufferPool ?? false;
    this.features = {
      // multiDrawIndexed() loops drawIndexed; native multi-draw is a feature-detected optimization
      // layered on render bundles, not exposed through this flag yet.
      multiDraw: false,
      timerQuery: adapter ? adapter.features.has("timestamp-query") : false,
      gpuCopy: true,
    };
  }

  /** Request an adapter + device, configure the canvas, and return a ready device. */
  static async create(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    opts: {
      powerPreference?: GPUPowerPreference;
      requiredFeatures?: GPUFeatureName[];
      depthFormat?: TextureFormat;
      alphaMode?: GPUCanvasAlphaMode;
      /** MEM-2: enable the device-level freed-GPUBuffer-object pool (default OFF). */
      bufferPool?: boolean;
    } = {},
  ): Promise<WebGPUDevice> {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      throw new Error("WebGPU is not available in this environment");
    }
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: opts.powerPreference ?? "high-performance",
    });
    if (!adapter) throw new Error("No WebGPU adapter (requestAdapter returned null)");
    const requiredFeatures = (opts.requiredFeatures ?? []).filter((f) => adapter.features.has(f));
    const gpu = await adapter.requestDevice({ requiredFeatures });
    const device = new WebGPUDevice(gpu, adapter, { bufferPool: opts.bufferPool });
    device.configureCanvas(canvas, { depthFormat: opts.depthFormat, alphaMode: opts.alphaMode });
    return device;
  }

  /** Configure (or reconfigure) the canvas for rendering. Sets up the linear-color chain. */
  configureCanvas(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    opts: { depthFormat?: TextureFormat; alphaMode?: GPUCanvasAlphaMode; usage?: GPUTextureUsageFlags } = {},
  ): void {
    const ctx = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!ctx) throw new Error("Failed to acquire a 'webgpu' canvas context");
    this.context = ctx;

    const base = navigator.gpu.getPreferredCanvasFormat(); // 'bgra8unorm' | 'rgba8unorm'
    this.canvasViewFormat = `${base}-srgb` as GPUTextureFormat; // linear blending (§17.2)
    this._colorFormat = base === "rgba8unorm" ? TextureFormat.Rgba8Srgb : TextureFormat.Bgra8Srgb;
    ctx.configure({
      device: this.gpu,
      format: base,
      viewFormats: [this.canvasViewFormat],
      alphaMode: opts.alphaMode ?? "opaque",
      // COPY_SRC lets readCanvasPixels() screenshot the swapchain for the smoke harness and the §22
      // pixel-diff suite. It is cheap and universally supported; keep it on by default. (Parens
      // matter: `|` binds tighter than `??`, so this is `opts.usage ?? (RA | COPY_SRC)`.)
      usage: opts.usage ?? (GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC),
    });

    this._depthFormat = opts.depthFormat ?? TextureFormat.Depth24Plus;
    this.resize(canvas.width, canvas.height);
  }

  /** Recreate the depth attachment to match the drawing-buffer size. Call on canvas resize. */
  resize(width: number, height: number): void {
    this.depthTexture?.destroy();
    this.depthTexture = this.gpu.createTexture({
      size: { width: Math.max(1, width), height: Math.max(1, height) },
      format: GPU_TEXTURE_FORMAT[this._depthFormat],
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      label: "depth",
    });
    this.depthView = this.depthTexture.createView();
  }

  /** Color-target format pipelines must use (the `-srgb` canvas view format). */
  get colorFormat(): TextureFormat {
    return this._colorFormat;
  }
  /** Depth-target format pipelines must use. */
  get depthFormat(): TextureFormat {
    return this._depthFormat;
  }
  /** The underlying GPUDevice, for advanced/compute paths. */
  get raw(): GPUDevice {
    return this.gpu;
  }

  // --- Buffers --------------------------------------------------------------------
  createBuffer(desc: BufferDesc): GpuBufferHandle {
    // MEM-2: on a pool hit, recycle a freed buffer of the SAME usage whose ACTUAL size already fits — no real
    // gpu.createBuffer and no liveBuffers bump (the buffer never logically left GPU memory). Off ⇒ skipped.
    if (this.bufferPoolEnabled) {
      const reused = this.takeFromPool(desc.usage, desc.sizeBytes);
      if (reused) {
        this.poolHits++;
        return handleOf<GpuBufferHandle>(reused);
      }
    }
    const buffer = this.gpu.createBuffer({
      size: desc.sizeBytes,
      usage: gpuBufferUsage()[desc.usage],
      label: desc.label,
    });
    this.liveBuffers++;
    this.createBufferCalls++;
    return handleOf<GpuBufferHandle>({ buffer, sizeBytes: desc.sizeBytes, usage: desc.usage } satisfies BufferRec);
  }

  /** MEM-2: best-fit pull from the usage bucket — the SMALLEST pooled rec whose ACTUAL `sizeBytes >= requested`
   *  (TRAP-B: never hand back a buffer smaller than asked). Removes + returns it, or null on a miss. */
  private takeFromPool(usage: BufferUsage, requested: number): BufferRec | null {
    const bucket = this.bufferPool.get(usage);
    if (!bucket || bucket.length === 0) return null;
    let bestIdx = -1;
    let bestSize = Infinity;
    for (let i = 0; i < bucket.length; i++) {
      const s = bucket[i].sizeBytes;
      if (s >= requested && s < bestSize) {
        bestIdx = i;
        bestSize = s;
        if (s === requested) break; // exact fit — can't do better
      }
    }
    if (bestIdx < 0) return null;
    const rec = bucket[bestIdx];
    bucket.splice(bestIdx, 1);
    return rec;
  }

  writeBuffer(h: GpuBufferHandle, offsetBytes: number, data: ArrayBufferView): void {
    // Cast away the ArrayBufferLike/SharedArrayBuffer generic mismatch (TS 5.7+ vs @webgpu/types);
    // our data is always a regular ArrayBuffer-backed view.
    this.gpu.queue.writeBuffer(recOf<BufferRec>(h).buffer, offsetBytes, data as GPUAllowSharedBufferSource);
  }

  copyBuffer(src: GpuBufferHandle, srcRange: ByteRange, dst: GpuBufferHandle, dstOffset: number): void {
    const s = recOf<BufferRec>(src).buffer;
    const d = recOf<BufferRec>(dst).buffer;
    if (s === d) {
      throw new Error(
        "copyBuffer: source and destination must differ — WebGPU copyBufferToBuffer requires " +
          "src !== dst (compact into a fresh buffer, or use a compute shader; WEBGPU_FINDINGS §4)",
      );
    }
    // In a cross-arena batch, encode into the shared encoder (one submit at endCopyBatch); else own encoder+submit.
    const enc = this.copyBatchEnc();
    enc.copyBufferToBuffer(s, srcRange.offsetBytes, d, dstOffset, srcRange.sizeBytes);
    if (!this.copyBatchActive) this.gpu.queue.submit([enc.finish()]);
  }

  copyBufferBatch(src: GpuBufferHandle, copies: readonly BufferCopy[], dst: GpuBufferHandle): void {
    if (copies.length === 0) return; // no work ⇒ no submit
    const s = recOf<BufferRec>(src).buffer;
    const d = recOf<BufferRec>(dst).buffer;
    if (s === d) {
      throw new Error(
        "copyBufferBatch: source and destination must differ — WebGPU copyBufferToBuffer requires " +
          "src !== dst (compact into a fresh buffer; WEBGPU_FINDINGS §4)",
      );
    }
    // One encoder, one submit for ALL ranges (INFRA-1): arena grow/compact relocate every live block here. When a
    // cross-arena batch is open, encode into the SHARED encoder so several arenas collapse into one submit too.
    const enc = this.copyBatchEnc();
    for (const c of copies) {
      enc.copyBufferToBuffer(s, c.srcOffset, d, c.dstOffset, c.size);
    }
    if (!this.copyBatchActive) this.gpu.queue.submit([enc.finish()]);
  }

  /** The encoder to record a copy into: the shared cross-arena batch encoder (lazily created) when a batch is
   *  open, otherwise a throwaway per-call encoder the caller submits immediately. */
  private copyBatchEnc(): GPUCommandEncoder {
    if (!this.copyBatchActive) return this.gpu.createCommandEncoder();
    if (!this.copyBatchEncoder) this.copyBatchEncoder = this.gpu.createCommandEncoder();
    return this.copyBatchEncoder;
  }

  beginCopyBatch(): void {
    this.copyBatchActive = true; // encoder is created lazily on the first copy (no wasted submit if none relocate)
  }

  endCopyBatch(): void {
    this.copyBatchActive = false;
    if (this.copyBatchEncoder) {
      this.gpu.queue.submit([this.copyBatchEncoder.finish()]); // ONE submit for every arena relocated this commit
      this.copyBatchEncoder = null;
    }
    // Run the destroys deferred during the batch — AFTER the submit, so each copy SOURCE buffer was still alive
    // when the GPU read it (WebGPU frees a destroyed buffer only once the commands referencing it complete).
    if (this.copyBatchDeferredDestroys.length > 0) {
      const pending = this.copyBatchDeferredDestroys.splice(0);
      for (const h of pending) this.destroyBuffer(h);
    }
  }

  destroyBuffer(h: GpuBufferHandle): void {
    if (this.copyBatchActive) { this.copyBatchDeferredDestroys.push(h); return; } // defer past the batch submit
    const rec = recOf<BufferRec>(h);
    // MEM-2: with the pool ON, retain the buffer object instead of destroying it (no liveBuffers change — it
    // still holds GPU memory and may be reused). Over-cap, evict the largest/oldest with a REAL .destroy().
    // Off ⇒ the exact pre-MEM-2 path: real .destroy() + liveBuffers--.
    if (this.bufferPoolEnabled) {
      this.returnToPool(rec);
      return;
    }
    rec.buffer.destroy();
    this.liveBuffers--;
  }

  /** MEM-2: push a freed BufferRec into its usage bucket (capped by count AND bytes — TRAP-E). When adding it
   *  would exceed a cap, evict largest-first (and, ties aside, oldest by insertion order) with a real .destroy()
   *  + liveBuffers-- so the gauge keeps tracking true GPU memory (TRAP-C). */
  private returnToPool(rec: BufferRec): void {
    let bucket = this.bufferPool.get(rec.usage);
    if (!bucket) {
      bucket = [];
      this.bufferPool.set(rec.usage, bucket);
    }
    bucket.push(rec); // newest at the tail; eviction prefers the largest, breaking ties by oldest (head)
    let bytes = 0;
    for (const r of bucket) bytes += r.sizeBytes;
    while (bucket.length > POOL_MAX_COUNT_PER_USAGE || (bucket.length > 0 && bytes > POOL_MAX_BYTES_PER_USAGE)) {
      let evictIdx = 0;
      let evictSize = bucket[0].sizeBytes;
      for (let i = 1; i < bucket.length; i++) {
        if (bucket[i].sizeBytes > evictSize) {
          evictIdx = i;
          evictSize = bucket[i].sizeBytes;
        }
      }
      const [evicted] = bucket.splice(evictIdx, 1);
      bytes -= evicted.sizeBytes;
      evicted.buffer.destroy();
      this.liveBuffers--;
    }
  }

  // --- Textures / samplers --------------------------------------------------------
  createTexture(desc: TextureDesc): GpuTextureHandle {
    const depth = isDepthFormat(desc.format);
    let usage = depth
      ? GPUTextureUsage.RENDER_ATTACHMENT
      : GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        // RENDER_ATTACHMENT enables copyExternalImageToTexture + manual mip blits (§8).
        GPUTextureUsage.RENDER_ATTACHMENT;
    if (desc.renderable) usage |= GPUTextureUsage.RENDER_ATTACHMENT;

    const texture = this.gpu.createTexture({
      size: { width: desc.width, height: desc.height, depthOrArrayLayers: desc.layers ?? 1 },
      format: GPU_TEXTURE_FORMAT[desc.format],
      mipLevelCount: desc.mips ?? 1,
      usage,
      label: desc.label,
    });
    return handleOf<GpuTextureHandle>({
      texture,
      view: texture.createView(),
      width: desc.width,
      height: desc.height,
      bytesPerTexel: bytesPerTexel(desc.format),
    } satisfies TextureRec);
  }

  writeTexture(h: GpuTextureHandle, data: ArrayBufferView | ImageBitmap, mip = 0, region?: TextureRegion): void {
    const t = recOf<TextureRec>(h);
    // AUDIT #4: a `region` writes only that texel sub-rect (data is the tightly-packed sub-rect). No
    // 256-byte bytesPerRow alignment applies to queue.writeTexture (that rule is for buffer→texture copies).
    const w = region ? region.width : Math.max(1, t.width >> mip);
    const hgt = region ? region.height : Math.max(1, t.height >> mip);
    const origin = region ? { x: region.x, y: region.y } : undefined;
    if (ArrayBuffer.isView(data)) {
      this.gpu.queue.writeTexture(
        { texture: t.texture, mipLevel: mip, origin },
        data as GPUAllowSharedBufferSource,
        { bytesPerRow: w * t.bytesPerTexel, rowsPerImage: hgt },
        { width: w, height: hgt },
      );
    } else {
      // ImageBitmap (or other external image source); `region` is honored as the destination origin.
      this.gpu.queue.copyExternalImageToTexture(
        { source: data },
        { texture: t.texture, mipLevel: mip, origin },
        { width: w, height: hgt },
      );
    }
  }

  destroyTexture(h: GpuTextureHandle): void {
    recOf<TextureRec>(h).texture.destroy();
  }

  createSampler(desc: SamplerDesc = {}): SamplerHandle {
    const sampler = this.gpu.createSampler({
      magFilter: GPU_FILTER[desc.mag ?? FilterMode.Nearest],
      minFilter: GPU_FILTER[desc.min ?? FilterMode.Nearest],
      mipmapFilter: GPU_FILTER[desc.mip ?? FilterMode.Nearest],
      addressModeU: GPU_ADDRESS[desc.address ?? AddressMode.Clamp],
      addressModeV: GPU_ADDRESS[desc.address ?? AddressMode.Clamp],
      addressModeW: GPU_ADDRESS[desc.address ?? AddressMode.Clamp],
      label: desc.label,
    });
    return handleOf<SamplerHandle>({ sampler } satisfies SamplerRec);
  }

  // --- Shaders / pipelines / bindings ---------------------------------------------
  createShaderModule(code: string, label?: string): ShaderModuleHandle {
    return handleOf<ShaderModuleHandle>({ module: this.gpu.createShaderModule({ code, label }) } satisfies ShaderRec);
  }

  createPipeline(desc: PipelineDesc): PipelineHandle {
    const shader = recOf<ShaderRec>(desc.shader).module;

    const vertexLayout: GPUVertexBufferLayout = {
      arrayStride: desc.vertexLayout.strideBytes,
      stepMode: "vertex",
      attributes: desc.vertexLayout.attributes.map((a) => ({
        format: toGpuVertexFormat(a),
        offset: a.offsetBytes,
        shaderLocation: a.location,
      })),
    };

    const blend: GPUBlendState | undefined = desc.pass.blend
      ? {
          // Straight-alpha "over". (Premultiplied variant: color srcFactor "one".) §17.2
          color: { operation: "add", srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
          alpha: { operation: "add", srcFactor: "one", dstFactor: "one-minus-src-alpha" },
        }
      : undefined;

    const depthStencil: GPUDepthStencilState | undefined = desc.depthFormat
      ? {
          format: GPU_TEXTURE_FORMAT[desc.depthFormat],
          // NOTE: depthWriteEnabled is OPTIONAL in @webgpu/types but required by validation —
          // always set it explicitly (TRAP from WEBGPU_FINDINGS §18).
          depthWriteEnabled: desc.pass.depthWrite,
          depthCompare: desc.depthCompare
            ? GPU_COMPARE[desc.depthCompare]
            : desc.pass.depthTest
              ? "less"
              : "always",
        }
      : undefined;

    const layout: GPUPipelineLayout | GPUAutoLayoutMode = desc.bindingLayout
      ? this.gpu.createPipelineLayout({ bindGroupLayouts: [this.buildBindGroupLayout(desc.bindingLayout)] })
      : "auto";

    const pipeline = this.gpu.createRenderPipeline({
      label: desc.label,
      layout,
      vertex: { module: shader, entryPoint: desc.vertexEntry, buffers: [vertexLayout] },
      primitive: { topology: GPU_TOPOLOGY[desc.pass.topology ?? PrimitiveTopology.TriangleList], cullMode: GPU_CULL[desc.pass.cull ?? CullMode.Back], frontFace: "ccw" },
      depthStencil,
      // Single-sampled today. MSAA + alpha-to-coverage is a deferred backend decision (RENDERER_PLAN
      // §2.5 open decisions): sampleCount bakes into render bundles, so it must be chosen alongside the
      // Phase-5 bundle cache. The old `alphaToCoverage` knob was removed — it could never work with count:1.
      multisample: { count: 1 },
      fragment: {
        module: shader,
        entryPoint: desc.fragmentEntry,
        constants: desc.constants,
        targets: [{ format: GPU_TEXTURE_FORMAT[desc.colorFormat], blend, writeMask: 0xf }],
      },
    });
    return handleOf<PipelineHandle>({ pipeline } satisfies PipelineRec);
  }

  createBindings(desc: BindingsDesc): BindingsHandle {
    const pipeline = recOf<PipelineRec>(desc.pipeline).pipeline;
    const group = desc.group ?? 0;
    const layout = pipeline.getBindGroupLayout(group);

    const entries: GPUBindGroupEntry[] = desc.entries.map((e) => {
      const r = e.resource;
      if ("buffer" in r) {
        const buf = recOf<BufferRec>(r.buffer);
        return { binding: e.binding, resource: { buffer: buf.buffer, offset: r.offset ?? 0, size: r.size } };
      }
      if ("sampler" in r) {
        return { binding: e.binding, resource: recOf<SamplerRec>(r.sampler).sampler };
      }
      return { binding: e.binding, resource: recOf<TextureRec>(r.texture).view };
    });

    return handleOf<BindingsHandle>({ group, bindGroup: this.gpu.createBindGroup({ layout, entries }) } satisfies BindingsRec);
  }

  createRenderBundle(desc: RenderBundleDesc, record: (enc: BundleEncoder) => void): RenderBundleHandle {
    // sampleCount: 1 matches createPipeline's single-sampled pipelines (MSAA bakes into bundles — §2.5).
    // depthReadOnly defaults false ⇒ matches the terrain pass (depth-write on for opaque); the bundle's
    // formats must equal the replaying pass's attachments or executeBundles validation fails.
    const enc = this.gpu.createRenderBundleEncoder({
      colorFormats: [GPU_TEXTURE_FORMAT[desc.colorFormat]],
      depthStencilFormat: desc.depthFormat ? GPU_TEXTURE_FORMAT[desc.depthFormat] : undefined,
      sampleCount: 1,
      label: desc.label,
    });
    const rec = new WebGPURenderBundleEncoder(enc);
    record(rec);
    const bundle = enc.finish({ label: desc.label });
    return handleOf<RenderBundleHandle>({ bundle, drawCount: rec.drawCount } satisfies BundleRec);
  }

  private buildBindGroupLayout(entries: BindingLayoutEntry[]): GPUBindGroupLayout {
    return this.gpu.createBindGroupLayout({
      entries: entries.map((e): GPUBindGroupLayoutEntry => {
        let visibility = 0;
        for (const s of e.visibility) visibility |= gpuShaderStage()[s];
        const t = e.type;
        if (t.kind === "uniform-buffer") {
          return { binding: e.binding, visibility, buffer: { type: "uniform", hasDynamicOffset: t.dynamicOffset ?? false } };
        }
        if (t.kind === "storage-buffer") {
          return { binding: e.binding, visibility, buffer: { type: "read-only-storage" } };
        }
        if (t.kind === "sampler") {
          return { binding: e.binding, visibility, sampler: { type: GPU_SAMPLER_TYPE[t.sampler ?? SamplerBindingType.Filtering] } };
        }
        return { binding: e.binding, visibility, texture: { sampleType: GPU_SAMPLE_TYPE[t.sampleType ?? TextureSampleType.Float] } };
      }),
    });
  }

  // --- Frame ----------------------------------------------------------------------
  beginPass(target: RenderTarget, clear: ClearDesc): PassEncoder {
    if (target.id !== null) {
      throw new Error("WebGPUDevice.beginPass: only the canvas target (id=null) is supported yet");
    }
    if (!this.context) {
      throw new Error("WebGPUDevice.beginPass: configureCanvas() must be called first");
    }

    const colorView = this.context.getCurrentTexture().createView({ format: this.canvasViewFormat });
    const colorAttachment: GPURenderPassColorAttachment = {
      view: colorView,
      loadOp: clear.color ? "clear" : "load",
      storeOp: "store",
      clearValue: clear.color
        ? { r: clear.color[0], g: clear.color[1], b: clear.color[2], a: clear.color[3] }
        : undefined,
    };

    const depthAttachment: GPURenderPassDepthStencilAttachment | undefined = this.depthView
      ? {
          view: this.depthView,
          // The frame's FIRST pass must provide clear.depth; later passes may "load".
          depthLoadOp: clear.depth !== undefined ? "clear" : "load",
          depthStoreOp: "store",
          depthClearValue: clear.depth ?? 1.0,
        }
      : undefined;

    const enc = this.gpu.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [colorAttachment],
      depthStencilAttachment: depthAttachment,
    });
    this.passCount++;
    return new WebGPUPassEncoder(this.gpu, enc, pass, this);
  }

  /**
   * Read back the current swapchain texture as raw bytes — a screenshot. Used by the smoke
   * harness self-check and the §22 pixel-diff suite. Requires the swapchain to carry `COPY_SRC`
   * usage (the default from `configureCanvas`).
   *
   * INVARIANT: call this in the SAME synchronous task as the frame's draws, with NO `await`
   * between `beginPass` and this call. `getCurrentTexture()` returns the same texture only until
   * the canvas's (optionally-async) expiry task runs at a yield point; awaiting in between could
   * let the texture expire/destroy before the copy is encoded (WEBGPU_FINDINGS readback note).
   *
   * The returned data is row-padded to a 256-byte `bytesPerRow` (a `copyTextureToBuffer`
   * requirement); index a pixel with `(y * bytesPerRow + x * 4)`. Channel order/encoding is the
   * raw swapchain format (`bgra8unorm` or `rgba8unorm` — see `colorFormat`), sRGB-encoded.
   */
  async readCanvasPixels(): Promise<{ width: number; height: number; bytesPerRow: number; data: Uint8Array }> {
    if (!this.context) throw new Error("WebGPUDevice.readCanvasPixels: configureCanvas() must be called first");
    const tex = this.context.getCurrentTexture();
    const width = tex.width;
    const height = tex.height;
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    const readback = this.gpu.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: "canvas-readback",
    });
    const enc = this.gpu.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: readback, bytesPerRow, rowsPerImage: height }, { width, height });
    this.gpu.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(readback.getMappedRange()).slice(); // copy out before unmap detaches it
    readback.unmap();
    readback.destroy();
    return { width, height, bytesPerRow, data };
  }

  destroy(): void {
    // MEM-2: release every retained pool buffer for real before tearing the device down (gpu.destroy() would
    // free them anyway, but keep liveBuffers honest if the gauge is sampled post-destroy).
    for (const bucket of this.bufferPool.values()) {
      for (const rec of bucket) {
        rec.buffer.destroy();
        this.liveBuffers--;
      }
    }
    this.bufferPool.clear();
    this.depthTexture?.destroy();
    this.gpu.destroy();
  }
}
