// Backend-neutral graphics abstraction. RENDERER_PLAN.md §2.
//
// EVERY WebGLBuffer/WebGLTexture/WebGLProgram (and the WebGPU equivalents) lives behind this
// interface. mesh/, world/, and workers/ MUST NOT import a `WebGL*`/`GPU*` type — they deal in
// plain typed arrays, the opaque handles from ./handles, and the neutral enums from ./gpu-enums.
//
// The interface is modeled on COMMAND ENCODING (pass -> setPipeline -> setBindings -> draw), not
// WebGL's global bind-then-draw state machine. WebGL2 can emulate this trivially; WebGPU needs it
// natively. Do NOT add `bindBuffer`-style global-state methods here (TRAP 2.A).
//
// Reusable primitives are split out: opaque handles in ./handles, the GPU vocabulary enums in
// ./gpu-enums. This file is just the DESCRIPTORS + the device/encoder interfaces.

import type { ByteRange } from "../types";
import type {
  BindingsHandle,
  GpuBufferHandle,
  GpuTextureHandle,
  PipelineHandle,
  RenderBundleHandle,
  SamplerHandle,
  ShaderModuleHandle,
} from "./handles";
import type {
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
} from "./gpu-enums";

// Re-export the primitives so `import { GpuBufferHandle, TextureFormat } from "../core/GraphicsDevice"`
// keeps working — callers see one cohesive surface even though the definitions are split out.
export type {
  BindingsHandle,
  GpuBufferHandle,
  GpuTextureHandle,
  PipelineHandle,
  RenderBundleHandle,
  SamplerHandle,
  ShaderModuleHandle,
} from "./handles";
export {
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
} from "./gpu-enums";

// --- Descriptors ------------------------------------------------------------------

export interface BufferDesc {
  sizeBytes: number;
  usage: BufferUsage;
  /** Hint for backends that distinguish static/dynamic storage. */
  dynamic?: boolean;
  label?: string;
}

export interface TextureDesc {
  width: number;
  height: number;
  /** Block atlas is typically a single 2D RGBA8 texture; layers reserved for later. */
  layers?: number;
  format: TextureFormat;
  mips?: number;
  /** Add RENDER_ATTACHMENT usage (needed for copyExternalImageToTexture + mip blits). */
  renderable?: boolean;
  label?: string;
}

export interface SamplerDesc {
  mag?: FilterMode;
  min?: FilterMode;
  mip?: FilterMode;
  address?: AddressMode;
  label?: string;
}

/** One bind-group-layout entry. Omit the whole layout to reflect an 'auto' layout. */
export interface BindingLayoutEntry {
  binding: number;
  visibility: ShaderStage[];
  // `kind` here is a DISCRIMINANT (object-shape tag), not a closed option set, so it stays a
  // string literal rather than an enum — see STYLE_GUIDE.md "Enums vs string literals".
  type:
    | { kind: "uniform-buffer"; dynamicOffset?: boolean }
    // Read-only storage buffer (SSBO). The vertex shader indexes a per-draw array by `instance_index`
    // (F2 completion — the per-section origin lives here, 16B/draw, instead of a 256B dynamic-offset slot).
    | { kind: "storage-buffer" }
    | { kind: "sampler"; sampler?: SamplerBindingType }
    | { kind: "texture"; sampleType?: TextureSampleType };
}

export interface VertexAttribute {
  location: number;
  /**
   * Scalar kind. On WebGPU this maps to a `GPUVertexFormat` (`${kind}` or `${kind}x${components}`);
   * the `asInt` flag is a WebGL-only concern. `unorm*`/`snorm*` decode to floats; `uint*` stay ints.
   */
  kind: VertexScalarKind;
  components: number;
  offsetBytes: number;
  /** WebGL2 only: use `vertexAttribIPointer` (TRAP 10.B). Ignored by WebGPU. */
  asInt: boolean;
}
export interface VertexLayout {
  strideBytes: number;
  attributes: VertexAttribute[];
}

/** Fixed-function state per terrain pass. RENDERER_PLAN §12 (TRAP 12.C). */
export interface PassStateDesc {
  depthTest: boolean;
  depthWrite: boolean;
  blend: boolean;
  /**
   * Face culling. Default `Back` — MC cube faces are wound so back-face culling is correct and
   * halves opaque fragment work. Debug/overlay passes (and the smoke harness, which has no
   * meshing winding to rely on) use `None`.
   */
  cull?: CullMode;
  /** Primitive topology. Default triangle-list; `line-list` draws vertex pairs as real GPU lines
   *  (vanilla `RenderPipelines.LINES` — for the structure-block wireframe box). */
  topology?: PrimitiveTopology;
}

/**
 * A render pipeline: one shader module (vertex+fragment entry points) + vertex layout + the
 * fixed-state knobs (PassStateDesc) + color/depth target formats. Pass-specific behavior is
 * encoded here, not at draw time (pipelines are immutable). RENDERER_PLAN §12, §10.
 */
export interface PipelineDesc {
  label: string;
  shader: ShaderModuleHandle;
  vertexEntry?: string;
  fragmentEntry?: string;
  vertexLayout: VertexLayout;
  pass: PassStateDesc;
  /** Color target format — must match the render target rendered to. */
  colorFormat: TextureFormat;
  /** Depth target format, or null/undefined for no depth attachment. */
  depthFormat?: TextureFormat | null;
  /** Depth comparison; default `Less`. Use `Greater` for reverse-Z (WEBGPU_FINDINGS §17.3). */
  depthCompare?: CompareFn;
  /** Bind group 0 layout. If omitted, an 'auto' layout is reflected from the shader. */
  bindingLayout?: BindingLayoutEntry[];
  /** Pipeline-overridable constants, e.g. { ALPHA_TEST: 1 } for the cutout pass (§10). */
  constants?: Record<string, number>;
}

export type BindingResource =
  | { buffer: GpuBufferHandle; offset?: number; size?: number }
  | { sampler: SamplerHandle }
  | { texture: GpuTextureHandle };

export interface BindingEntry {
  binding: number;
  resource: BindingResource;
}

export interface BindingsDesc {
  /** Pipeline whose bind-group-layout(s) this group is built against. */
  pipeline: PipelineHandle;
  /** Bind group index (default 0). */
  group?: number;
  entries: BindingEntry[];
}

/** A texel sub-rect for partial texture uploads (AUDIT #4 — per-animated-sprite atlas cell updates). */
export interface TextureRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** One src→dst range for a batched GPU-side copy (INFRA-1 — arena relocation in a single submit). */
export interface BufferCopy {
  srcOffset: number;
  size: number;
  dstOffset: number;
}

export interface ClearDesc {
  color?: readonly [number, number, number, number];
  depth?: number;
}

/** Attachment formats a render bundle bakes in (F1). They MUST match the render pass that replays it —
 *  same color/depth target formats the bundle's pipelines were built against (single-sampled today). */
export interface RenderBundleDesc {
  colorFormat: TextureFormat;
  depthFormat?: TextureFormat | null;
  label?: string;
}

export interface RenderTarget {
  /** null = default framebuffer / swapchain. */
  readonly id: string | null;
  width: number;
  height: number;
}

// --- Command encoding -------------------------------------------------------------
/**
 * The draw-recording command subset shared by a live render pass AND a recorded render bundle (F1).
 * A `GPURenderBundle` pre-records exactly these commands so a static-camera frame can replay them with
 * `PassEncoder.executeBundles` instead of re-encoding the per-draw loop (RENDERER_PLAN §15, BACKEND 4.A).
 */
export interface BundleEncoder {
  setPipeline(p: PipelineHandle): void;
  /**
   * Bind group 0. `dynamicOffsets` (bytes, one per dynamic-offset binding in the group) selects a
   * window into a dynamic uniform buffer — used to give each batched draw its own per-section data
   * within a single render pass (RENDERER_PLAN §15). Must be multiples of the device's
   * minUniformBufferOffsetAlignment (≤256, guaranteed).
   */
  setBindings(b: BindingsHandle, dynamicOffsets?: number[]): void;
  /**
   * Bind a vertex buffer at `slot`. `offsetBytes` (default 0) selects a window into the buffer — used
   * to give each section its own slice of a shared per-pass vertex `BufferArena` (RENDERER_PLAN §11):
   * the section's per-section indices stay 0-based and `baseVertex` stays 0, the byte offset locates the
   * slice. Must be a multiple of 4 (WebGPU vertex-buffer offset alignment; the arena aligns to 4).
   */
  setVertexBuffer(slot: number, buf: GpuBufferHandle, offsetBytes?: number): void;
  /** Bind an index buffer. `offsetBytes` (default 0) windows into a shared index `BufferArena` — the same
   *  per-section-slice trick as `setVertexBuffer` (RENDERER_PLAN §11). Must be a multiple of the index size
   *  (4 for Uint32; the arena aligns to 4). */
  setIndexBuffer(buf: GpuBufferHandle, indexType: IndexFormat, offsetBytes?: number): void;
  /**
   * Single indexed draw. `firstInstance` (default 0) seeds the shader's `instance_index` builtin for a
   * one-instance draw — the F2-completion trick that gives each batched draw its own index into a shared
   * per-draw storage buffer (the section origin) with NO per-draw bind-group set. Direct (non-indirect)
   * draws support a non-zero firstInstance on WebGPU without any feature flag.
   */
  drawIndexed(indexCount: number, firstIndex: number, baseVertex: number, firstInstance?: number): void;
}

export interface PassEncoder extends BundleEncoder {
  /**
   * Batched multi-draw. Backends fall back to a drawIndexed loop when neither WEBGL_multi_draw
   * nor native indirect is available (RENDERER_PLAN §15, BACKEND 15.A).
   */
  multiDrawIndexed(counts: Int32Array, firstIndices: Int32Array, baseVertices: Int32Array): void;
  /** Replay pre-recorded render bundles (F1). Their baked color/depth formats + sample count must match
   *  this pass's attachments. The pass's pipeline/bind-group/buffer state is reset afterward (WebGPU spec). */
  executeBundles(bundles: RenderBundleHandle[]): void;
  end(): void;
}

export interface DeviceFeatures {
  multiDraw: boolean;
  timerQuery: boolean;
  /** GPU-to-GPU copy. True on all WebGL2 (copyBufferSubData) and WebGPU (TRAP 2.B). */
  gpuCopy: boolean;
}

export interface GraphicsDevice {
  readonly backend: BackendKind;
  readonly features: DeviceFeatures;

  // Buffers
  createBuffer(desc: BufferDesc): GpuBufferHandle;
  writeBuffer(h: GpuBufferHandle, offsetBytes: number, data: ArrayBufferView): void;
  /** GPU-side copy. Used by arena compaction — NOT a CPU round-trip (TRAP 2.B, 11.A). */
  copyBuffer(src: GpuBufferHandle, srcRange: ByteRange, dst: GpuBufferHandle, dstOffset: number): void;
  /**
   * Batched GPU-side copy: every range in `copies` is copied from `src` into `dst` in ONE command
   * encoder + ONE `queue.submit` (INFRA-1). Arena grow/compact relocate K live blocks per call; this
   * collapses K submits into 1. `src !== dst` is required (same WebGPU constraint as `copyBuffer`);
   * an empty `copies` is a no-op (no submit).
   */
  copyBufferBatch(src: GpuBufferHandle, copies: readonly BufferCopy[], dst: GpuBufferHandle): void;
  /**
   * INFRA-1 (cross-arena): open a copy batch so EVERY `copyBuffer`/`copyBufferBatch` until `endCopyBatch`
   * encodes into ONE shared command encoder + ONE `queue.submit`, instead of one submit per arena
   * relocation (the uploader compacts several arenas in a single commit). While a batch is open,
   * `destroyBuffer` is DEFERRED until `endCopyBatch` runs (after the submit), so a copy SOURCE buffer
   * outlives the GPU copy that reads it (WebGPU frees a destroyed buffer only once pending ops complete).
   * Optional — backends without an encoder model (WebGL2's synchronous `copyBufferSubData`) omit it and
   * callers guard with `?.()`. Must be paired with `endCopyBatch`; not re-entrant (one commit at a time).
   */
  beginCopyBatch?(): void;
  /** Close the {@link beginCopyBatch} batch: submit the single encoder (if any copy was recorded), then run
   *  the deferred buffer destroys. No-op if no batch is open / nothing was copied. */
  endCopyBatch?(): void;
  destroyBuffer(h: GpuBufferHandle): void;

  // Textures / samplers
  createTexture(desc: TextureDesc): GpuTextureHandle;
  /**
   * Upload pixels to a texture. With `region` (AUDIT #4), only that texel sub-rect is written and `data`
   * is the tightly-packed sub-rect (region.width × region.height × bytesPerTexel) — used to refresh a
   * single animated atlas cell (~1 KB) instead of re-uploading the whole multi-MB atlas.
   */
  writeTexture(h: GpuTextureHandle, data: ArrayBufferView | ImageBitmap, mip?: number, region?: TextureRegion): void;
  destroyTexture(h: GpuTextureHandle): void;
  createSampler(desc?: SamplerDesc): SamplerHandle;

  // Shaders / pipelines / bindings
  createShaderModule(code: string, label?: string): ShaderModuleHandle;
  createPipeline(desc: PipelineDesc): PipelineHandle;
  createBindings(desc: BindingsDesc): BindingsHandle;

  /**
   * Record a reusable render bundle (F1). `record` issues the same draw commands as a pass would; the
   * returned handle is replayed via `PassEncoder.executeBundles`. Bundles are GC-managed (no destroy).
   * The recorded buffer/pipeline handles must outlive every replay — the caller invalidates the bundle
   * whenever any are freed/relocated (TRAP 4.A: key the bundle on the draw-list + arena layout revision).
   */
  createRenderBundle(desc: RenderBundleDesc, record: (enc: BundleEncoder) => void): RenderBundleHandle;

  // Frame
  beginPass(target: RenderTarget, clear: ClearDesc): PassEncoder;

  // Lifecycle
  destroy(): void;
}
