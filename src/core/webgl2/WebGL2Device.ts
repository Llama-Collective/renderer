// WebGL2 backend — the optional fallback GraphicsDevice. RENDERER_PLAN.md §2; WEBGL_FINDINGS.md;
// WEBGL_PLAN.md (built phase-by-phase).
//
// Maps the backend-neutral GraphicsDevice onto WebGL2's bind-then-draw state machine: the neutral
// command-encoding interface (beginPass -> setPipeline -> setBindings -> draw) is emulated with
// immediate GL state calls (TRAP 2.A — WebGL2 emulates the command model trivially). Opaque handles
// wrap the real GL objects; the ONLY casts that bridge handle<->record are `handleOf`/`recOf`.
//
// PHASE STATUS (WEBGL_PLAN.md):
//  - W0 ✅ device skeleton: `create` (context + feature detect), canvas clear via `beginPass`, the
//    swapchain surface (colorFormat/depthFormat/resize/readback) and HUD counters, `destroy`.
//  - W1 ✅ buffers (type-correct allocation, TRAP W1.A), GPU copy (gl.copyBufferSubData — no CPU mirror),
//    textures (immutable texStorage2D + full/sub-rect texSubImage2D), samplers.
//  - W2 ✅ shaders (GLSL ES 3.00 twins resolved by label — core/webgl2/shaders.ts), pipelines (program
//    link + per-`constants` #define injection + fixed-function state), bindings (UBO bindBufferRange +
//    texture units), and the draw path (VAO + integer-attribute setup, TRAP 10.B).
//  - W3 ✅ the terrain vertical slice: the terrain twin (integer attrs, atlas + light-LUT samplers) and
//    divergence (A) — the per-draw section origin. WGSL's `var<storage, read> origins[]` indexed by
//    instance_index has no WebGL2 analogue, so a storage buffer binds as a std140 UBO `origins[N]` (N =
//    MAX_UNIFORM_BLOCK_SIZE/16, injected as ORIGIN_COUNT) indexed by the per-draw `u_BaseInstance` int the
//    device sets from `firstInstance` (TRAP W3.A); the z∈[0,1]→[-1,1] depth fixup rides in every twin.
//  - W4 ✅ divergence (C) the linear-sRGB color chain: every pass renders into an offscreen SRGB8_ALPHA8
//    FBO (+ its own DEPTH_COMPONENT24 depth, for multi-pass LOAD) so blending is LINEAR; each pass `end()`
//    presents it to the default canvas via a full-screen shader that re-encodes linear→sRGB (a raw blit
//    would double-darken — TRAP W4.C). Cutout/translucent passes ride the existing setPipeline state.
//  - W5 ✅ the rest of the shader roster — the entity twin (dynamic-offset UBO, gl_FrontFacing PER_FACE
//    lighting; frontFace(CCW) set unconditionally so it's deterministic) and the end-portal twin (drops
//    the WGSL Y-flip — gl_FragCoord is already Y-up; one WGSL sampler feeds two units via createBindings).
//    readCanvasPixels reads the presented default FB + Y-flips (top-first, bytesPerRow=w*4).
//  - W6 ✅ render-bundle replay (D — F1 degrades to a captured-and-replayed thunk list; the scene.ts guard
//    is now removed) + multidraw (E — WEBGL_multi_draw batch when all baseVertices 0, else the loop) + the
//    feature-detected fast-paths: EXT_clip_control (omit the z*2-w fixup via CLIP_ZERO_TO_ONE),
//    drawingBufferStorage (sRGB default buffer → plain-blit present, no re-encode shader). Each fast-path
//    has a pixel-identical core fallback (BACKEND W6.A); `fastPaths` reports which are active.
//  - W7 ⏳ cross-backend pixel-diff across all scenes + the §12.1 torture scene; program lifecycle; no-GPU UI.

import { notImplemented } from "../../notImplemented";
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
  PassEncoder,
  PassStateDesc,
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
  VertexLayout,
  VertexScalarKind,
} from "../GraphicsDevice";
import { AddressMode, BackendKind, BufferUsage, CompareFn, CullMode, FilterMode, IndexFormat, PrimitiveTopology, TextureFormat } from "../GraphicsDevice";
import type { CanvasDevice } from "../createDevice";
import { GLSL_TWINS, type GlslTwin, PRESENT_VERTEX, PRESENT_FRAGMENT } from "./shaders";
import { buildGlScalarTable, isNormalizedKind } from "./glAttrib";

// --- Handle backing records (opaque handles ARE these objects at runtime) — STYLE_GUIDE §3 ----------
interface BufferRec {
  buffer: WebGLBuffer;
  sizeBytes: number;
  /** Drives the type-assigning bind target (TRAP W1.A): Index → ELEMENT_ARRAY_BUFFER, else "other data". */
  usage: BufferUsage;
}
interface TextureRec {
  texture: WebGLTexture;
  width: number;
  height: number;
  glFormat: number;
  glType: number;
  bytesPerTexel: number;
}
interface SamplerRec {
  sampler: WebGLSampler;
}
interface ShaderRec {
  twin: GlslTwin;
  label: string;
}
/** Fixed-function GL state baked into a pipeline (mirrors WebGPU's immutable PassStateDesc). */
interface GlState {
  blend: boolean;
  depthTest: boolean;
  depthWrite: boolean;
  depthFunc: number;
  /** GL_BACK / GL_FRONT, or null for no culling. */
  cull: number | null;
}
interface PipelineRec {
  program: WebGLProgram;
  state: GlState;
  /** GL_TRIANGLES or GL_LINES. */
  drawMode: number;
  layout: VertexLayout;
  twin: GlslTwin;
  /** Attribute locations the linked program actually consumes (skip the rest — entity loc-3 quirk, TRAP 5.1). */
  activeLocations: Set<number>;
  /** Neutral bindings declared with a dynamic offset (drives setBindings' dynamicOffsets consumption). */
  dynamicOffsetBindings: Set<number>;
  /** The reserved per-draw section-origin selector (W3 terrain); null when the program has no such uniform. */
  uBaseInstanceLoc: WebGLUniformLocation | null;
}
interface UboBinding {
  binding: number;
  buffer: WebGLBuffer;
  offset: number;
  size: number;
  hasDynamicOffset: boolean;
}
interface TexBinding {
  unit: number;
  texture: WebGLTexture;
  sampler: WebGLSampler | null;
}
interface BindingsRec {
  ubos: UboBinding[];
  textures: TexBinding[];
}

/** The two — and only two — places the handle brand is bridged (mirrors WebGPUDevice; STYLE_GUIDE §3). */
function handleOf<H>(rec: object): H {
  return rec as unknown as H;
}
function recOf<R>(handle: object): R {
  return handle as unknown as R;
}

/** Inject `#define K V` for each override constant right after the mandatory `#version` first line. */
function injectDefines(source: string, constants?: Record<string, number>): string {
  if (!constants) return source;
  const entries = Object.entries(constants);
  if (entries.length === 0) return source;
  const defs = entries.map(([k, v]) => `#define ${k} ${v}`).join("\n");
  const nl = source.indexOf("\n");
  return source.slice(0, nl + 1) + defs + "\n" + source.slice(nl + 1);
}

/**
 * Context attributes for the fallback canvas. Single-sampled (`antialias: false`) to match the WebGPU
 * path's `count: 1` for cross-backend pixel parity; `alpha: true` keeps the W6 `drawingBufferStorage`
 * sRGB fast-path available. `preserveDrawingBuffer: true` makes the WebGL2 canvas keep its last frame
 * the way a WebGPU canvas does (it displays the last presented texture) — the parity that lets
 * `readCanvasPixels` (W5) and the Playwright visual harness capture the canvas after the frame settles.
 * W4 owns its depth: every pass renders into an offscreen SRGB8_ALPHA8 FBO with its own DEPTH_COMPONENT24
 * renderbuffer, so the context's default-framebuffer depth is pure waste — `depth: false`. The default
 * framebuffer is only ever the *present* target (color-only, written by the present shader).
 */
const CONTEXT_ATTRIBUTES: WebGLContextAttributes = {
  alpha: true,
  antialias: false,
  depth: false,
  stencil: false,
  premultipliedAlpha: false,
  preserveDrawingBuffer: true,
  powerPreference: "high-performance",
};

/** One texture format's WebGL2 triple: the immutable-storage internalFormat + the writable format/type. */
interface GlTexFormat {
  internalFormat: number;
  format: number;
  type: number;
  bytesPerTexel: number;
}

// --- W6 fast-path typings (these postdate / are absent from the WebGL2 lib.dom typings) --------------
/** EXT_clip_control: reconfigure the clip volume so z∈[0,1] (WebGPU's range) is GL's clip range. */
interface ExtClipControl {
  readonly LOWER_LEFT_EXT: number;
  readonly ZERO_TO_ONE_EXT: number;
  clipControlEXT(origin: number, depth: number): void;
}
/** WEBGL_multi_draw: one batched indexed multi-draw (no baseVertex array — usable only when all are 0). */
interface WebglMultiDraw {
  multiDrawElementsWEBGL(mode: number, counts: Int32Array, countsOffset: number, type: number, offsets: Int32Array, offsetsOffset: number, drawcount: number): void;
}
/** drawingBufferStorage: respecify the DEFAULT drawing buffer's sized format (e.g. SRGB8_ALPHA8). */
interface DrawingBufferStorageGL {
  drawingBufferStorage(sizedFormat: number, width: number, height: number): void;
}

export class WebGL2Device implements CanvasDevice {
  readonly backend = BackendKind.WebGL2;
  readonly features: DeviceFeatures;

  /** Render passes opened since construction (one per `beginPass`). HUD metric (Phase-P). */
  passCount = 0;
  /** Draw calls issued since construction. */
  drawCalls = 0;
  /** Live WebGLBuffers (created − destroyed). */
  liveBuffers = 0;
  /** Cumulative real gl.createBuffer calls (no buffer pool yet → equals logical creates). */
  createBufferCalls = 0;
  /** Buffer-pool hits. The fallback has no pool yet (W7, optional) → stays 0. */
  poolHits = 0;

  private readonly gl: WebGL2RenderingContext;
  /** Enable the (future, W7) freed-buffer pool. Captured now to mirror WebGPUDevice's create signature. */
  private readonly bufferPool: boolean;
  /** GL_MAX_UNIFORM_BLOCK_SIZE — the hard cap on any UBO bind range (origins is the only one near it). */
  private readonly maxUniformBlockSize: number;
  /** W3 (A): how many vec4 section origins fit in one UBO = floor(MAX_UNIFORM_BLOCK_SIZE/16). The terrain
   *  twin declares `origins[ORIGIN_COUNT]` at this size; the per-draw `u_BaseInstance` index is clamped to
   *  [0,N) so a pass with more sections is bounded + logged, never silent garbage (TRAP W3.B). */
  readonly maxOrigins: number;
  /** The terrain `origins[ORIGIN_COUNT]` UBO's byte size (maxOrigins·16). A Storage buffer is physically
   *  allocated to at least this so a bound range fully backs the std140 block — ANGLE raises INVALID_OPERATION
   *  ("uniform buffer too small") otherwise, even though the shader only ever reads the written prefix. */
  private readonly originBlockBytes: number;
  /** Device-level `#define`s injected into every program (alongside per-pipeline `constants`): ORIGIN_COUNT. */
  private readonly deviceDefines: Record<string, number>;
  /** One-shot guard so the >N-sections overflow warning (TRAP W3.B) is logged once, not per draw. */
  private originOverflowWarned = false;
  /** A dedicated VAO bound only for buffer uploads, so typing an index buffer via ELEMENT_ARRAY_BUFFER
   *  (TRAP W1.A) never disturbs a render VAO's element binding. */
  private readonly uploadVao: WebGLVertexArrayObject;
  /** The single VAO every draw binds; attributes are re-pointed per draw (the per-section VAO cache is a
   *  W7 optimization). `drawEnabledLocs` tracks which attribute locations are currently enabled on it so a
   *  draw can disable any a prior draw left on. */
  private readonly drawVao: WebGLVertexArrayObject;
  private drawEnabledLocs = new Set<number>();
  /** Linked programs, deleted in destroy() (PipelineCache has no device-side destroy hook — WEBGL_FINDINGS §8). */
  private readonly programs: WebGLProgram[] = [];
  /** Created GL samplers, deleted in destroy() (W7). Samplers have no caller-side destroy hook — the WebGPU
   *  twin frees its sampler state implicitly in `gpu.destroy()`, so the device must track + free these. */
  private readonly samplers: WebGLSampler[] = [];
  /** Set by destroy() so a second call is a no-op (idempotent teardown — the non-nullable VAO fields can't be
   *  nulled, so a flag is the clean guard; double-destroy is unreachable through the app but cheap to harden). */
  private destroyed = false;
  // --- W4: the offscreen SRGB8_ALPHA8 color chain --------------------------------------------------
  /** The offscreen render target every pass draws into (so blending is LINEAR). Color is an SRGB8_ALPHA8
   *  TEXTURE (sampled by the present shader); depth is its own DEPTH_COMPONENT24 renderbuffer so depth
   *  LOADs across the multi-pass frame. Recreated by `resize`; null until the first resize. */
  private offscreenFbo: WebGLFramebuffer | null = null;
  private offscreenColor: WebGLTexture | null = null;
  private offscreenDepth: WebGLRenderbuffer | null = null;
  private offscreenW = 0;
  private offscreenH = 0;
  /** The present pass: a full-screen triangle that samples the offscreen color (sRGB→linear) and re-encodes
   *  linear→sRGB into the linear default framebuffer (a raw blit would double-darken — TRAP W4.C). */
  private presentProgram: WebGLProgram | null = null;
  /** An attribute-less VAO for the present draw (positions come from gl_VertexID, so no buffers). */
  private presentVao: WebGLVertexArrayObject | null = null;
  // --- W6: feature-detected fast-paths (BACKEND W6.A — each has a correct, pixel-identical core fallback) ---
  /** EXT_clip_control: when present, the clip volume is z∈[0,1] (WebGPU's) so the per-twin z*2-w depth fixup
   *  is compiled out (CLIP_ZERO_TO_ONE define). null → the core fixup path (W3). */
  private readonly clipControlExt: ExtClipControl | null;
  /** WEBGL_multi_draw: batches multiDrawIndexed when every baseVertex is 0; null → the drawIndexed loop. */
  private readonly multiDrawExt: WebglMultiDraw | null;
  /** drawingBufferStorage available → the DEFAULT buffer is SRGB8_ALPHA8, so present is a plain
   *  encoding-preserving blit instead of the re-encoding shader. false → the W4 present-shader path. */
  private readonly hasDrawingBufferStorage: boolean;
  // Neutral-enum → GLenum tables (STYLE_GUIDE §4), built from `gl` so values stay readable and a missing
  // mapping fails at the Record literal.
  private readonly glFilter: Record<FilterMode, number>;
  private readonly glAddress: Record<AddressMode, number>;
  private readonly glTexFormat: Record<TextureFormat, GlTexFormat>;
  private readonly glScalar: Record<VertexScalarKind, number>;
  private readonly glCompare: Record<CompareFn, number>;
  private readonly glCull: Record<CullMode, number | null>;
  // colorFormat reports the sRGB format the W4 offscreen FBO stores into, so pipeline/bundle color-format
  // keys stay consistent with the WebGPU path even while W0–W3 draw to a plain default framebuffer.
  private _colorFormat: TextureFormat = TextureFormat.Rgba8Srgb;
  private _depthFormat: TextureFormat = TextureFormat.Depth24Plus;

  constructor(gl: WebGL2RenderingContext, opts: { bufferPool?: boolean; preferCoreFallbacks?: boolean } = {}) {
    this.gl = gl;
    this.bufferPool = opts.bufferPool ?? false;
    // W6 fast-path detection (BACKEND W6.A): probe each capability ONCE, store the ext object where the
    // call/enums live on it. Real extensions → getExtension; methods that postdate the 2.0 baseline → typeof.
    // `preferCoreFallbacks` (test hook, GATE W6.1) forces every fast-path off so the core paths (z*2-w fixup,
    // present shader, drawIndexed loop) can be A/B'd against the fast paths for pixel-identity.
    const core = opts.preferCoreFallbacks ?? false;
    this.clipControlExt = core ? null : (gl.getExtension("EXT_clip_control") as ExtClipControl | null);
    this.multiDrawExt = core ? null : (gl.getExtension("WEBGL_multi_draw") as WebglMultiDraw | null);
    this.hasDrawingBufferStorage = core ? false : typeof (gl as unknown as Partial<DrawingBufferStorageGL>).drawingBufferStorage === "function";
    this.features = {
      // WEBGL_multi_draw backs multiDrawIndexed batching; feature-detected, drawElements-loop fallback (W6).
      multiDraw: this.multiDrawExt !== null,
      timerQuery: gl.getExtension("EXT_disjoint_timer_query_webgl2") !== null,
      // WebGL2 ALWAYS has copyBufferSubData (GPU-to-GPU) — arena compaction needs no CPU mirror (TRAP 2.B).
      gpuCopy: true,
    };

    // W3 (A): size the origins UBO to the real device limit so a large scene fits in one bind (TRAP W3.B).
    // Guaranteed minimum is 16384 B ⇒ ≥1024 sections; desktop is typically 65536 ⇒ 4096.
    this.maxUniformBlockSize = gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE) as number;
    this.maxOrigins = Math.max(1, Math.floor(this.maxUniformBlockSize / 16));
    this.originBlockBytes = this.maxOrigins * 16;
    this.deviceDefines = { ORIGIN_COUNT: this.maxOrigins };
    if (this.clipControlExt) {
      // W6 (clip_control fast-path): set GL's clip volume to z∈[0,1] (WebGPU's range) so the per-twin
      // `gl_Position.z*2-w` fixup is a no-op double-transform → compile it OUT via the twins' `#ifndef
      // CLIP_ZERO_TO_ONE` guard (injected device-wide like ORIGIN_COUNT). LOWER_LEFT keeps GL's bottom-left
      // origin (the winding/gl_FragCoord reasoning — W5 — depends on it). Absent ⇒ the core z*2-w fixup.
      this.clipControlExt.clipControlEXT(this.clipControlExt.LOWER_LEFT_EXT, this.clipControlExt.ZERO_TO_ONE_EXT);
      this.deviceDefines.CLIP_ZERO_TO_ONE = 1;
    }

    const uvao = gl.createVertexArray();
    const dvao = gl.createVertexArray();
    const pvao = gl.createVertexArray();
    if (!uvao || !dvao || !pvao) throw new Error("WebGL2Device: gl.createVertexArray returned null");
    this.uploadVao = uvao;
    this.drawVao = dvao;
    this.presentVao = pvao;
    this.presentProgram = this.buildPresentProgram();
    // Tightly-packed uploads: animated atlas sub-rects can be odd-width, which the default UNPACK_ALIGNMENT
    // of 4 would mis-stride. We never need any other alignment, so set it once (WEBGL_FINDINGS §8).
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    this.glFilter = {
      [FilterMode.Nearest]: gl.NEAREST,
      [FilterMode.Linear]: gl.LINEAR,
    };
    this.glAddress = {
      [AddressMode.Clamp]: gl.CLAMP_TO_EDGE,
      [AddressMode.Repeat]: gl.REPEAT,
      [AddressMode.Mirror]: gl.MIRRORED_REPEAT,
    };
    // SRGB8_ALPHA8 is the sample-decoding analogue of WebGPU's rgba8unorm-srgb (TRAP W1.B). Bgra8* have no
    // sized BGRA internal format in WebGL2 and are swapchain-only (never reach createTexture) — mapped to
    // RGBA for table exhaustiveness. Depth formats become renderbuffers in W4; listed here for completeness.
    this.glTexFormat = {
      [TextureFormat.Rgba8]: { internalFormat: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, bytesPerTexel: 4 },
      [TextureFormat.Rgba8Srgb]: { internalFormat: gl.SRGB8_ALPHA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, bytesPerTexel: 4 },
      [TextureFormat.R8]: { internalFormat: gl.R8, format: gl.RED, type: gl.UNSIGNED_BYTE, bytesPerTexel: 1 },
      [TextureFormat.Bgra8]: { internalFormat: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, bytesPerTexel: 4 },
      [TextureFormat.Bgra8Srgb]: { internalFormat: gl.SRGB8_ALPHA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, bytesPerTexel: 4 },
      [TextureFormat.Depth24Plus]: { internalFormat: gl.DEPTH_COMPONENT24, format: gl.DEPTH_COMPONENT, type: gl.UNSIGNED_INT, bytesPerTexel: 4 },
      [TextureFormat.Depth32Float]: { internalFormat: gl.DEPTH_COMPONENT32F, format: gl.DEPTH_COMPONENT, type: gl.FLOAT, bytesPerTexel: 4 },
    };
    this.glScalar = buildGlScalarTable(gl);
    this.glCompare = {
      [CompareFn.Less]: gl.LESS,
      [CompareFn.LessEqual]: gl.LEQUAL,
      [CompareFn.Greater]: gl.GREATER,
      [CompareFn.GreaterEqual]: gl.GEQUAL,
      [CompareFn.Always]: gl.ALWAYS,
    };
    this.glCull = {
      [CullMode.None]: null,
      [CullMode.Back]: gl.BACK,
      [CullMode.Front]: gl.FRONT,
    };
  }

  /**
   * Acquire a 'webgl2' context and return a ready device. Synchronous (no adapter request) — the
   * createDevice factory awaits it uniformly alongside the async WebGPU path.
   */
  static create(canvas: HTMLCanvasElement | OffscreenCanvas, opts: { bufferPool?: boolean; preferCoreFallbacks?: boolean } = {}): WebGL2Device {
    const gl = canvas.getContext("webgl2", CONTEXT_ATTRIBUTES) as WebGL2RenderingContext | null;
    if (!gl) throw new Error("WebGL2 is not available (getContext('webgl2') returned null)");
    const device = new WebGL2Device(gl, opts);
    device.resize(canvas.width, canvas.height);
    return device;
  }

  /** Color-target format pipelines/bundles must be built against. */
  get colorFormat(): TextureFormat {
    return this._colorFormat;
  }
  /** Depth-target format pipelines must be built against. */
  get depthFormat(): TextureFormat {
    return this._depthFormat;
  }
  /** The underlying context, for advanced paths. */
  get raw(): WebGL2RenderingContext {
    return this.gl;
  }

  /** Diagnostic: which W6 fast-paths are active on this device/context (HUD + harness). Each has a
   *  pixel-identical core fallback, so these only affect performance/precision, never correctness. */
  get fastPaths(): { clipControl: boolean; drawingBufferStorage: boolean; multiDraw: boolean } {
    return { clipControl: this.clipControlExt !== null, drawingBufferStorage: this.hasDrawingBufferStorage, multiDraw: this.multiDrawExt !== null };
  }

  /** Internal (the pass encoder): is the WEBGL_multi_draw batch path available? */
  canMultiDraw(): boolean {
    return this.multiDrawExt !== null;
  }

  /**
   * Match the viewport to the canvas size. The default framebuffer's color/depth buffers auto-resize
   * with the canvas drawing buffer; W4's offscreen SRGB8_ALPHA8 FBO + depth renderbuffer get recreated
   * here too once they exist.
   */
  resize(width: number, height: number): void {
    const w = Math.max(1, width);
    const h = Math.max(1, height);
    this.createOffscreen(w, h); // W4: the offscreen sRGB color chain is canvas-sized
    this.gl.viewport(0, 0, w, h);
  }

  /** (Re)create the offscreen render target at `w×h`: an SRGB8_ALPHA8 color **texture** (the present shader
   *  samples it) + a DEPTH_COMPONENT24 depth **renderbuffer** (its own, so depth LOADs across the multi-pass
   *  frame). Idempotent on unchanged size. Every pass renders here so "over" blending runs in linear space. */
  private createOffscreen(w: number, h: number): void {
    const gl = this.gl;
    if (this.offscreenFbo && this.offscreenW === w && this.offscreenH === h) return;
    // W6 (drawingBufferStorage fast-path): make the DEFAULT drawing buffer SRGB8_ALPHA8 too, so present() is
    // a plain encoding-preserving blit (no re-encode shader). The offscreen FBO is still needed for its own
    // depth (you can't attach depth to FBO 0). Re-applied on every size change (it pins the buffer size).
    if (this.hasDrawingBufferStorage) (gl as unknown as DrawingBufferStorageGL).drawingBufferStorage(gl.SRGB8_ALPHA8, w, h);
    if (this.offscreenColor) gl.deleteTexture(this.offscreenColor);
    if (this.offscreenDepth) gl.deleteRenderbuffer(this.offscreenDepth);
    if (this.offscreenFbo) gl.deleteFramebuffer(this.offscreenFbo);

    const color = gl.createTexture();
    if (!color) throw new Error("WebGL2Device.createOffscreen: gl.createTexture returned null");
    gl.bindTexture(gl.TEXTURE_2D, color);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.SRGB8_ALPHA8, w, h);
    // texStorage2D defaults MIN_FILTER to NEAREST_MIPMAP_LINEAR — incomplete for a 1-mip texture (samples
    // black). The present is a 1:1 copy, so NEAREST + clamp everywhere.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    const depth = gl.createRenderbuffer();
    if (!depth) throw new Error("WebGL2Device.createOffscreen: gl.createRenderbuffer returned null");
    gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h); // matches depthFormat Depth24Plus
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);

    const fbo = gl.createFramebuffer();
    if (!fbo) throw new Error("WebGL2Device.createOffscreen: gl.createFramebuffer returned null");
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, color, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`WebGL2Device.createOffscreen: offscreen FBO incomplete (0x${status.toString(16)})`);
    }
    this.offscreenColor = color;
    this.offscreenDepth = depth;
    this.offscreenFbo = fbo;
    this.offscreenW = w;
    this.offscreenH = h;
  }

  /** Link the device-internal present program (full-screen triangle; samples the offscreen sRGB texture on
   *  unit 0 and re-encodes linear→sRGB to the linear default framebuffer — TRAP W4.C). Built once in the ctor. */
  private buildPresentProgram(): WebGLProgram {
    const gl = this.gl;
    const vs = this.compile(gl.VERTEX_SHADER, PRESENT_VERTEX, "present");
    const fs = this.compile(gl.FRAGMENT_SHADER, PRESENT_FRAGMENT, "present");
    const program = gl.createProgram();
    if (!program) throw new Error("WebGL2Device.buildPresentProgram: gl.createProgram returned null");
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`WebGL2Device.buildPresentProgram: link failed: ${log}`);
    }
    gl.useProgram(program);
    const loc = gl.getUniformLocation(program, "uSrc");
    if (loc) gl.uniform1i(loc, 0); // offscreen color sampled from texture unit 0
    return program;
  }

  /**
   * Present the offscreen sRGB color onto the default canvas framebuffer. Called from EVERY pass `end()` —
   * the only contract-compatible seam (a frame is many beginPass/end pairs all LOADing into the same offscreen
   * attachment, and the GraphicsDevice interface has no frame-end hook), so each end() leaves the canvas
   * showing the cumulative result. A raw blitFramebuffer would sRGB-decode-without-re-encode → too dark
   * (TRAP W4.C); the present shader samples (decode) + re-encodes instead.
   */
  present(): void {
    const gl = this.gl;
    if (!this.offscreenFbo) return;
    if (this.hasDrawingBufferStorage) {
      // W6: the default buffer is SRGB8_ALPHA8 too, so blit offscreen(sRGB)→default(sRGB) is encoding-
      // preserving — no present shader / re-encode round-trip (the W6 win; TRAP W4.C only bit a LINEAR default
      // buffer). Leave the default FB bound + depth writable for readback / the next frame's clear.
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.offscreenFbo);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
      gl.blitFramebuffer(0, 0, this.offscreenW, this.offscreenH, 0, 0, this.offscreenW, this.offscreenH, gl.COLOR_BUFFER_BIT, gl.NEAREST);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.depthMask(true); // TRAP W4.B
      return;
    }
    if (!this.offscreenColor || !this.presentProgram) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.offscreenW, this.offscreenH);
    gl.useProgram(this.presentProgram);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);
    gl.depthMask(true); // restore for the next frame's offscreen depth clear (TRAP W4.B)
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.offscreenColor);
    // Sample through the texture's OWN NEAREST/CLAMP params, not a sampler object a prior pass left on unit 0
    // (a sampler object overrides texParameteri). Today's only unit-0 sampler is the atlas's NEAREST/CLAMP
    // (a no-op for this 1:1 texel-center copy), but a future LINEAR/REPEAT unit-0 sampler (W5 entity) would
    // otherwise distort the present — so make it independent of leftover state.
    gl.bindSampler(0, null);
    gl.bindVertexArray(this.presentVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    // Leave the default framebuffer bound so a readback/screenshot reads the presented canvas; the next
    // beginPass rebinds the offscreen FBO and the next setPipeline re-applies all GL state wholesale.
  }

  // --- Frame ----------------------------------------------------------------------
  beginPass(target: RenderTarget, clear: ClearDesc): PassEncoder {
    if (target.id !== null) {
      throw new Error("WebGL2Device.beginPass: only the canvas target (id=null) is supported yet");
    }
    const gl = this.gl;
    // W4: bind the offscreen SRGB8_ALPHA8 FBO (not the default framebuffer) so "over" blending runs in linear
    // space. A frame is many passes (clear, then loadOp 'load') all rendering here; the depth attachment is the
    // offscreen's own renderbuffer, so depth LOADs across passes. The default FB is only the present target.
    if (!this.offscreenFbo || this.offscreenW !== target.width || this.offscreenH !== target.height) {
      this.createOffscreen(Math.max(1, target.width), Math.max(1, target.height));
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.offscreenFbo);
    gl.viewport(0, 0, target.width, target.height);

    let mask = 0;
    if (clear.color) {
      gl.clearColor(clear.color[0], clear.color[1], clear.color[2], clear.color[3]);
      mask |= gl.COLOR_BUFFER_BIT;
    }
    if (clear.depth !== undefined) {
      // TRAP W0.A: a prior translucent pass may have left depthMask off → the depth clear would no-op.
      gl.depthMask(true);
      gl.clearDepth(clear.depth);
      mask |= gl.DEPTH_BUFFER_BIT;
    }
    if (mask !== 0) gl.clear(mask);

    this.passCount++;
    return new WebGL2PassEncoder(this, gl);
  }

  /**
   * Screenshot the canvas (test/harness hook). Reads the DEFAULT (presented) framebuffer — `present()`
   * leaves it bound and `preserveDrawingBuffer` keeps it — so the bytes are the sRGB-ENCODED presented
   * result, matching WebGPU's swapchain readback. `gl.readPixels` has a bottom-left origin, so the rows
   * are Y-FLIPPED into top-first order to match the WebGPU contract; `bytesPerRow = width*4` (tight, no
   * 256-byte copyTextureToBuffer padding). Synchronous read wrapped in a resolved Promise.
   */
  readCanvasPixels(): Promise<{ width: number; height: number; bytesPerRow: number; data: Uint8Array }> {
    const gl = this.gl;
    const width = this.offscreenW;
    const height = this.offscreenH;
    const bytesPerRow = width * 4;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); // the presented canvas (RGBA8, sRGB-encoded bytes)
    gl.pixelStorei(gl.PACK_ALIGNMENT, 4);
    const raw = new Uint8Array(bytesPerRow * height);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, raw);
    // Y-flip: GL row 0 is the BOTTOM; the contract wants row 0 = TOP (like WebGPU copyTextureToBuffer).
    const data = new Uint8Array(bytesPerRow * height);
    for (let y = 0; y < height; y++) {
      data.set(raw.subarray(y * bytesPerRow, (y + 1) * bytesPerRow), (height - 1 - y) * bytesPerRow);
    }
    return Promise.resolve({ width, height, bytesPerRow, data });
  }

  // --- Buffers --------------------------------------------------------------------
  createBuffer(desc: BufferDesc): GpuBufferHandle {
    const gl = this.gl;
    const buffer = gl.createBuffer();
    if (!buffer) throw new Error("WebGL2Device.createBuffer: gl.createBuffer returned null");
    const hint = desc.dynamic ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW;
    // W3 (A): a Storage buffer becomes the origins UBO, so it must be at least one full `origins[ORIGIN_COUNT]`
    // block — ANGLE rejects a draw whose bound UBO range is smaller than the declared block, even though the
    // shader only reads the renderer-written prefix. Pad here; the extra is zero (bufferData) and never read.
    const allocBytes = desc.usage === BufferUsage.Storage ? Math.max(desc.sizeBytes, this.originBlockBytes) : desc.sizeBytes;
    // TRAP W1.A: the FIRST bind permanently sets a buffer's element-vs-other-data type, and a buffer can
    // only ever be one. Index buffers MUST be typed via ELEMENT_ARRAY_BUFFER (on the scratch VAO so a
    // render VAO is untouched); everything else is "other data" via ARRAY_BUFFER. Allocating an index
    // buffer through COPY_WRITE_BUFFER would wrongly type it "other data" and break its draw-time bind.
    if (desc.usage === BufferUsage.Index) {
      gl.bindVertexArray(this.uploadVao);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, allocBytes, hint);
    } else {
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, allocBytes, hint);
    }
    this.liveBuffers++;
    this.createBufferCalls++;
    return handleOf<GpuBufferHandle>({ buffer, sizeBytes: allocBytes, usage: desc.usage } satisfies BufferRec);
  }

  writeBuffer(h: GpuBufferHandle, offsetBytes: number, data: ArrayBufferView): void {
    const gl = this.gl;
    // Once typed by createBuffer, a buffer can be (re)written through the type-neutral COPY_WRITE_BUFFER
    // without re-typing it or disturbing ARRAY_BUFFER/ELEMENT_ARRAY_BUFFER/VAO state. bufferSubData honors
    // the view's byteOffset/byteLength — callers pass subarrays.
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, recOf<BufferRec>(h).buffer);
    gl.bufferSubData(gl.COPY_WRITE_BUFFER, offsetBytes, data);
  }

  copyBuffer(src: GpuBufferHandle, srcRange: ByteRange, dst: GpuBufferHandle, dstOffset: number): void {
    const gl = this.gl;
    const s = recOf<BufferRec>(src).buffer;
    const d = recOf<BufferRec>(dst).buffer;
    // Keep the WebGPU contract's src !== dst (the arena always compacts into a fresh buffer); a WebGL2
    // buffer also cannot bind to COPY_READ and COPY_WRITE at once.
    if (s === d) throw new Error("WebGL2Device.copyBuffer: source and destination must differ (src !== dst)");
    gl.bindBuffer(gl.COPY_READ_BUFFER, s);
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, d);
    gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.COPY_WRITE_BUFFER, srcRange.offsetBytes, dstOffset, srcRange.sizeBytes);
  }

  copyBufferBatch(src: GpuBufferHandle, copies: readonly BufferCopy[], dst: GpuBufferHandle): void {
    if (copies.length === 0) return; // no work ⇒ no GL calls (matches WebGPU/Fake)
    const gl = this.gl;
    const s = recOf<BufferRec>(src).buffer;
    const d = recOf<BufferRec>(dst).buffer;
    if (s === d) throw new Error("WebGL2Device.copyBufferBatch: source and destination must differ (src !== dst)");
    // One bind pair, N synchronous copies (TRAP F.1: never crosses the index/non-index boundary — the arena
    // copies index→index and vertex→vertex only). No submit to coalesce, so begin/endCopyBatch are omitted.
    gl.bindBuffer(gl.COPY_READ_BUFFER, s);
    gl.bindBuffer(gl.COPY_WRITE_BUFFER, d);
    for (const c of copies) {
      gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.COPY_WRITE_BUFFER, c.srcOffset, c.dstOffset, c.size);
    }
  }
  // beginCopyBatch/endCopyBatch are intentionally OMITTED — they are optional (GraphicsDevice.ts:298-304)
  // and callers guard with `?.()`. WebGL2's copyBufferSubData is synchronous, so there is nothing to
  // coalesce and no source-buffer-lifetime hazard (WEBGL_FINDINGS §3-F), so destroy can run eagerly.

  destroyBuffer(h: GpuBufferHandle): void {
    this.gl.deleteBuffer(recOf<BufferRec>(h).buffer);
    this.liveBuffers--;
  }

  // --- Textures / samplers --------------------------------------------------------
  createTexture(desc: TextureDesc): GpuTextureHandle {
    const gl = this.gl;
    if ((desc.layers ?? 1) !== 1) {
      // The block/entity atlas and the light LUT are all single-layer 2D textures (WEBGL_FINDINGS §8).
      return notImplemented("WebGL2Device.createTexture: array textures (layers>1)");
    }
    const f = this.glTexFormat[desc.format];
    const texture = gl.createTexture();
    if (!texture) throw new Error("WebGL2Device.createTexture: gl.createTexture returned null");
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Immutable single-level storage — the renderer never mips (atlases sample Nearest; WEBGL_FINDINGS §8).
    gl.texStorage2D(gl.TEXTURE_2D, desc.mips ?? 1, f.internalFormat, desc.width, desc.height);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return handleOf<GpuTextureHandle>({
      texture,
      width: desc.width,
      height: desc.height,
      glFormat: f.format,
      glType: f.type,
      bytesPerTexel: f.bytesPerTexel,
    } satisfies TextureRec);
  }

  writeTexture(h: GpuTextureHandle, data: ArrayBufferView | ImageBitmap, mip = 0, region?: TextureRegion): void {
    const gl = this.gl;
    const t = recOf<TextureRec>(h);
    if (!ArrayBuffer.isView(data)) {
      // The atlas/LUT paths only ever pass typed arrays (WEBGL_FINDINGS §8); the ImageBitmap branch lands later.
      return notImplemented("WebGL2Device.writeTexture: ImageBitmap source");
    }
    // With `region` only that texel sub-rect is written and `data` is the tightly-packed sub-rect; otherwise
    // the whole mip level. (UNPACK_ALIGNMENT=1 set once in the ctor handles odd-width sub-rects.)
    const x = region ? region.x : 0;
    const y = region ? region.y : 0;
    const w = region ? region.width : Math.max(1, t.width >> mip);
    const hgt = region ? region.height : Math.max(1, t.height >> mip);
    gl.bindTexture(gl.TEXTURE_2D, t.texture);
    gl.texSubImage2D(gl.TEXTURE_2D, mip, x, y, w, hgt, t.glFormat, t.glType, data);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  destroyTexture(h: GpuTextureHandle): void {
    this.gl.deleteTexture(recOf<TextureRec>(h).texture);
  }

  createSampler(desc: SamplerDesc = {}): SamplerHandle {
    const gl = this.gl;
    const sampler = gl.createSampler();
    if (!sampler) throw new Error("WebGL2Device.createSampler: gl.createSampler returned null");
    // Single-mip textures ⇒ no mipmap min-filter variant. MC terrain is Nearest everywhere; the WebGPU-only
    // `NonFiltering` sampler-binding concept has no WebGL2 analogue and is ignored (any sampler may filter).
    gl.samplerParameteri(sampler, gl.TEXTURE_MAG_FILTER, this.glFilter[desc.mag ?? FilterMode.Nearest]);
    gl.samplerParameteri(sampler, gl.TEXTURE_MIN_FILTER, this.glFilter[desc.min ?? FilterMode.Nearest]);
    const wrap = this.glAddress[desc.address ?? AddressMode.Clamp];
    gl.samplerParameteri(sampler, gl.TEXTURE_WRAP_S, wrap);
    gl.samplerParameteri(sampler, gl.TEXTURE_WRAP_T, wrap);
    gl.samplerParameteri(sampler, gl.TEXTURE_WRAP_R, wrap);
    this.samplers.push(sampler); // W7: tracked so destroy() frees them (no per-sampler destroy hook)
    return handleOf<SamplerHandle>({ sampler } satisfies SamplerRec);
  }

  // --- Shaders / pipelines / bindings ---------------------------------------------
  createShaderModule(_code: string, label?: string): ShaderModuleHandle {
    // Resolve the hand-written GLSL ES 3.00 twin by LABEL (the WGSL `code` can't be imported here without a
    // core→render dependency). Multiple labels can share one twin (e.g. overlay/explosion).
    const twin = label !== undefined ? GLSL_TWINS[label] : undefined;
    if (label === undefined || !twin) {
      return notImplemented(
        `WebGL2Device.createShaderModule: no GLSL twin for shader label "${label ?? "(none)"}" — add it to core/webgl2/shaders.ts (terrain/entity/portal are W3/W5)`,
      );
    }
    return handleOf<ShaderModuleHandle>({ twin, label } satisfies ShaderRec);
  }

  createPipeline(desc: PipelineDesc): PipelineHandle {
    const gl = this.gl;
    const twin = recOf<ShaderRec>(desc.shader).twin;
    // Device defines (ORIGIN_COUNT) + the pipeline's override constants. Per-pipeline constants win on a
    // name clash (none today). Injected into every program; twins that don't use ORIGIN_COUNT ignore it.
    const defines = desc.constants ? { ...this.deviceDefines, ...desc.constants } : this.deviceDefines;
    const vs = this.compile(gl.VERTEX_SHADER, injectDefines(twin.vertex, defines), desc.label);
    const fs = this.compile(gl.FRAGMENT_SHADER, injectDefines(twin.fragment, defines), desc.label);
    const program = gl.createProgram();
    if (!program) throw new Error("WebGL2Device.createPipeline: gl.createProgram returned null");
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`WebGL2Device.createPipeline: link failed for "${desc.label}": ${log}`);
    }

    // GLSL ES 3.00 has no `layout(binding=)` for uniform blocks or samplers — assign them after link.
    for (const b of twin.uniformBlocks) {
      const idx = gl.getUniformBlockIndex(program, b.name);
      if (idx !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, idx, b.binding);
    }
    gl.useProgram(program);
    for (const s of twin.samplers ?? []) {
      const loc = gl.getUniformLocation(program, s.name);
      if (loc) gl.uniform1i(loc, s.unit);
    }

    // Attribute locations the program actually consumes (so the draw path skips the rest — TRAP 5.1).
    const activeLocations = new Set<number>();
    const nAttr = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES) as number;
    for (let i = 0; i < nAttr; i++) {
      const info = gl.getActiveAttrib(program, i);
      if (!info) continue;
      const loc = gl.getAttribLocation(program, info.name);
      if (loc >= 0) activeLocations.add(loc);
    }

    const dynamicOffsetBindings = new Set<number>();
    for (const b of desc.bindingLayout ?? []) {
      if (b.type.kind === "uniform-buffer" && b.type.dynamicOffset) dynamicOffsetBindings.add(b.binding);
    }

    this.programs.push(program);
    return handleOf<PipelineHandle>({
      program,
      state: this.glState(desc.pass, desc.depthCompare),
      drawMode: (desc.pass.topology ?? PrimitiveTopology.TriangleList) === PrimitiveTopology.LineList ? gl.LINES : gl.TRIANGLES,
      layout: desc.vertexLayout,
      twin,
      activeLocations,
      dynamicOffsetBindings,
      uBaseInstanceLoc: gl.getUniformLocation(program, "u_BaseInstance"),
    } satisfies PipelineRec);
  }

  private compile(type: number, source: string, label: string): WebGLShader {
    const gl = this.gl;
    const sh = gl.createShader(type);
    if (!sh) throw new Error("WebGL2Device: gl.createShader returned null");
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
      const stage = type === gl.VERTEX_SHADER ? "vertex" : "fragment";
      throw new Error(`WebGL2Device: ${stage} shader compile failed for "${label}": ${log}`);
    }
    return sh;
  }

  private glState(pass: PassStateDesc, depthCompare?: CompareFn): GlState {
    // Match WebGPU createPipeline's default: depthCompare ?? (depthTest ? Less : Always) — BoxEffect xray
    // relies on the `depthTest:false → Always` resolution.
    const cmp = depthCompare ?? (pass.depthTest ? CompareFn.Less : CompareFn.Always);
    return {
      blend: pass.blend,
      depthTest: pass.depthTest,
      depthWrite: pass.depthWrite,
      depthFunc: this.glCompare[cmp],
      cull: this.glCull[pass.cull ?? CullMode.Back],
    };
  }

  createBindings(desc: BindingsDesc): BindingsHandle {
    const pipe = recOf<PipelineRec>(desc.pipeline);
    const samplers = pipe.twin.samplers ?? [];
    const ubos: UboBinding[] = [];
    const texByUnit = new Map<number, { texture: WebGLTexture | null; sampler: WebGLSampler | null }>();
    for (const e of desc.entries) {
      const r = e.resource;
      if ("buffer" in r) {
        const buf = recOf<BufferRec>(r.buffer);
        ubos.push({
          binding: e.binding,
          buffer: buf.buffer,
          offset: r.offset ?? 0,
          // Bind the whole buffer by default; the origins buffer is padded (createBuffer) to ≥ one full
          // `origins[ORIGIN_COUNT]` block so this fully backs it. Clamp to MAX_UNIFORM_BLOCK_SIZE for the
          // >ORIGIN_COUNT-sections overflow case (the per-draw u_BaseInstance clamp backstops indexing).
          size: Math.min(r.size ?? buf.sizeBytes, this.maxUniformBlockSize),
          hasDynamicOffset: pipe.dynamicOffsetBindings.has(e.binding),
        });
      } else if ("texture" in r) {
        const meta = samplers.find((s) => s.textureBinding === e.binding);
        if (!meta) return notImplemented(`WebGL2Device.createBindings: no sampler2D maps texture @binding(${e.binding}) — declare it in the twin (W3)`);
        const slot = texByUnit.get(meta.unit) ?? { texture: null, sampler: null };
        slot.texture = recOf<TextureRec>(r.texture).texture;
        texByUnit.set(meta.unit, slot);
      } else {
        // One WGSL sampler can serve MULTIPLE sampler2D units (the portal's single `samp` feeds both
        // skyTex@unit0 and portalTex@unit1) — bind it to every unit whose twin metadata references it.
        const metas = samplers.filter((s) => s.samplerBinding === e.binding);
        if (metas.length === 0) return notImplemented(`WebGL2Device.createBindings: no sampler2D maps sampler @binding(${e.binding}) — declare it in the twin`);
        const sampler = recOf<SamplerRec>(r.sampler).sampler;
        for (const meta of metas) {
          const slot = texByUnit.get(meta.unit) ?? { texture: null, sampler: null };
          slot.sampler = sampler;
          texByUnit.set(meta.unit, slot);
        }
      }
    }
    const textures: TexBinding[] = [];
    for (const [unit, v] of texByUnit) {
      if (v.texture) textures.push({ unit, texture: v.texture, sampler: v.sampler });
    }
    return handleOf<BindingsHandle>({ ubos, textures } satisfies BindingsRec);
  }

  createRenderBundle(_desc: RenderBundleDesc, record: (enc: BundleEncoder) => void): RenderBundleHandle {
    // W6 (D): WebGL2 has no native render bundles (BACKEND 4.A) — F1 degrades to a CAPTURED-AND-REPLAYED
    // command list. The recorder captures the neutral command stream as thunks; executeBundles re-issues them
    // against the live pass, reproducing the exact GL call sequence a direct encode would. The caller's
    // invalidation key (draw-list identity + layoutRev + viewProj) guarantees a replay never binds a
    // freed/relocated buffer (TRAP 4.A), so the captured handles stay valid for the bundle's life.
    const enc = new WebGL2BundleEncoder();
    record(enc);
    return handleOf<RenderBundleHandle>({ thunks: enc.thunks, drawCount: enc.drawCount } satisfies RenderBundleRec);
  }

  /**
   * Internal: the pass encoder delegates each indexed draw here so the VAO + integer-attribute setup
   * (which needs the device's private draw VAO, enable-tracking, and scalar-type table) lives in one place.
   */
  issueDrawIndexed(
    p: PipelineRec,
    vbuf: WebGLBuffer,
    vOffset: number,
    ibuf: WebGLBuffer,
    iOffset: number,
    indexType: number,
    indexCount: number,
    firstIndex: number,
    baseVertex: number,
    firstInstance: number,
  ): void {
    const gl = this.gl;
    // W3 (A): the per-draw section-origin selector — `firstInstance` becomes `u_BaseInstance`, indexing the
    // origins UBO (TRAP W3.A; the WebGL2 stand-in for WGSL's instance_index into a storage array). Programs
    // without `u_BaseInstance` (the W2 shaders) skip it. Clamp to [0, maxOrigins) so a pass with more sections
    // than the UBO holds is bounded + logged once, never silent out-of-bounds garbage (TRAP W3.B / 16.A).
    if (p.uBaseInstanceLoc) {
      let slot = firstInstance;
      if (slot >= this.maxOrigins) {
        if (!this.originOverflowWarned) {
          console.warn(
            `[WebGL2] origins UBO holds ${this.maxOrigins} sections (MAX_UNIFORM_BLOCK_SIZE=${this.maxUniformBlockSize}); a pass exceeded it — clamping origin index (TRAP W3.B). Per-chunk re-bind / a data-texture path is the fix for very large scenes.`,
          );
          this.originOverflowWarned = true;
        }
        slot = this.maxOrigins - 1;
      }
      gl.uniform1i(p.uBaseInstanceLoc, slot);
    }
    this.pointAttributes(p, vbuf, vOffset, baseVertex);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibuf);
    const idxSize = indexType === gl.UNSIGNED_INT ? 4 : 2;
    gl.drawElements(p.drawMode, indexCount, indexType, iOffset + firstIndex * idxSize);
    this.drawCalls++;
  }

  /** Bind the draw VAO and point/enable exactly the attributes the program consumes (skipping the rest —
   *  entity loc 3, TRAP 5.1; integer attrs via IPointer, TRAP W2.A), disabling any a prior draw left on.
   *  Shared by the single draw and the W6 multi-draw batch (which calls it once with baseVertex 0). */
  private pointAttributes(p: PipelineRec, vbuf: WebGLBuffer, vOffset: number, baseVertex: number): void {
    const gl = this.gl;
    gl.bindVertexArray(this.drawVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbuf);
    const stride = p.layout.strideBytes;
    const used = new Set<number>();
    for (const a of p.layout.attributes) {
      if (!p.activeLocations.has(a.location)) continue;
      used.add(a.location);
      const base = vOffset + a.offsetBytes + baseVertex * stride;
      const type = this.glScalar[a.kind];
      gl.enableVertexAttribArray(a.location);
      if (a.asInt) gl.vertexAttribIPointer(a.location, a.components, type, stride, base);
      else gl.vertexAttribPointer(a.location, a.components, type, isNormalizedKind(a.kind), stride, base);
    }
    for (const loc of this.drawEnabledLocs) if (!used.has(loc)) gl.disableVertexAttribArray(loc);
    this.drawEnabledLocs = used;
  }

  /**
   * W6 (E): the WEBGL_multi_draw batch — one call for the whole run. The caller guarantees the extension is
   * present AND every baseVertex is 0 (the extension has NO baseVertex array), so the attribute pointers are
   * identical for every sub-draw → set once. No per-draw `u_BaseInstance` (multidraw is the future region
   * tier, not the per-section-origin terrain path).
   */
  issueMultiDrawIndexed(
    p: PipelineRec,
    vbuf: WebGLBuffer,
    vOffset: number,
    ibuf: WebGLBuffer,
    iOffset: number,
    indexType: number,
    counts: Int32Array,
    firstIndices: Int32Array,
  ): void {
    const gl = this.gl;
    this.pointAttributes(p, vbuf, vOffset, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibuf);
    const idxSize = indexType === gl.UNSIGNED_INT ? 4 : 2;
    const byteOffsets = new Int32Array(counts.length); // multiDraw takes BYTE offsets, not firstIndex counts
    for (let i = 0; i < counts.length; i++) byteOffsets[i] = iOffset + firstIndices[i] * idxSize;
    this.multiDrawExt!.multiDrawElementsWEBGL(p.drawMode, counts, 0, indexType, byteOffsets, 0, counts.length);
    this.drawCalls += counts.length;
  }

  destroy(): void {
    if (this.destroyed) return; // idempotent: a second destroy() must not re-delete stale handles
    this.destroyed = true;
    const gl = this.gl;
    // W7: free the device-owned GL objects PipelineCache won't (no per-pipeline destroy hook, WEBGL_FINDINGS §8).
    for (const p of this.programs) gl.deleteProgram(p);
    this.programs.length = 0;
    for (const s of this.samplers) gl.deleteSampler(s);
    this.samplers.length = 0;
    if (this.presentProgram) gl.deleteProgram(this.presentProgram);
    if (this.offscreenColor) gl.deleteTexture(this.offscreenColor);
    if (this.offscreenDepth) gl.deleteRenderbuffer(this.offscreenDepth);
    if (this.offscreenFbo) gl.deleteFramebuffer(this.offscreenFbo);
    if (this.presentVao) gl.deleteVertexArray(this.presentVao);
    gl.deleteVertexArray(this.uploadVao);
    gl.deleteVertexArray(this.drawVao);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}

/**
 * The pass-recording command surface — emulating WebGPU's command encoder over WebGL's bind-then-draw
 * state machine (TRAP 2.A). setPipeline/setBindings apply GL state immediately; setVertexBuffer/
 * setIndexBuffer record the bound buffers; drawIndexed delegates to the device (which owns the draw VAO).
 */
class WebGL2PassEncoder implements PassEncoder {
  private pipeline: PipelineRec | null = null;
  private vbuf: WebGLBuffer | null = null;
  private vOffset = 0;
  private ibuf: WebGLBuffer | null = null;
  private iOffset = 0;
  private indexType: number;

  constructor(
    private readonly device: WebGL2Device,
    private readonly gl: WebGL2RenderingContext,
  ) {
    this.indexType = gl.UNSIGNED_INT;
  }

  setPipeline(p: PipelineHandle): void {
    const gl = this.gl;
    const rec = recOf<PipelineRec>(p);
    this.pipeline = rec;
    gl.useProgram(rec.program);
    const s = rec.state;
    if (s.depthTest) gl.enable(gl.DEPTH_TEST);
    else gl.disable(gl.DEPTH_TEST);
    gl.depthMask(s.depthWrite);
    gl.depthFunc(s.depthFunc);
    if (s.blend) {
      gl.enable(gl.BLEND);
      // Straight-alpha "over", matching WebGPU's blend state (WebGPUDevice §17.2).
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    } else {
      gl.disable(gl.BLEND);
    }
    // WINDING. GL_CCW matches WebGPU's frontFace:"ccw" here — NOT inverted, despite WebGPU's +Y-down
    // framebuffer vs WebGL2's +Y-up (the classic "canonical gotcha" does NOT apply with this renderer's
    // Y-up projection). Settled EMPIRICALLY by the cross-backend winding probe (harness/windingProbe.ts):
    // a CCW-in-NDC quad survives cull:Back as FRONT on BOTH real-Metal WebGPU and WebGL2 under GL_CCW.
    // (GL_CW would invert culling → inside-out terrain — guarded by the harness `winding-cullback` check.)
    // Set UNCONDITIONALLY (even when culling is off) so `gl_FrontFacing` is deterministic for the entity
    // PER_FACE box-lighting select (cull:None pipelines would otherwise never establish the winding).
    gl.frontFace(gl.CCW);
    if (s.cull !== null) {
      gl.enable(gl.CULL_FACE);
      gl.cullFace(s.cull);
    } else {
      gl.disable(gl.CULL_FACE);
    }
  }

  setBindings(b: BindingsHandle, dynamicOffsets?: number[]): void {
    const gl = this.gl;
    const rec = recOf<BindingsRec>(b);
    let d = 0;
    for (const u of rec.ubos) {
      const off = u.offset + (u.hasDynamicOffset && dynamicOffsets ? dynamicOffsets[d++] : 0);
      gl.bindBufferRange(gl.UNIFORM_BUFFER, u.binding, u.buffer, off, u.size);
    }
    for (const t of rec.textures) {
      gl.activeTexture(gl.TEXTURE0 + t.unit);
      gl.bindTexture(gl.TEXTURE_2D, t.texture);
      if (t.sampler) gl.bindSampler(t.unit, t.sampler);
    }
  }

  setVertexBuffer(_slot: number, buf: GpuBufferHandle, offsetBytes = 0): void {
    this.vbuf = recOf<BufferRec>(buf).buffer;
    this.vOffset = offsetBytes;
  }

  setIndexBuffer(buf: GpuBufferHandle, indexType: IndexFormat, offsetBytes = 0): void {
    this.ibuf = recOf<BufferRec>(buf).buffer;
    this.iOffset = offsetBytes;
    this.indexType = indexType === IndexFormat.Uint32 ? this.gl.UNSIGNED_INT : this.gl.UNSIGNED_SHORT;
  }

  drawIndexed(indexCount: number, firstIndex: number, baseVertex: number, firstInstance = 0): void {
    if (!this.pipeline || !this.vbuf || !this.ibuf) {
      throw new Error("WebGL2PassEncoder.drawIndexed: setPipeline + setVertexBuffer + setIndexBuffer must precede the draw");
    }
    this.device.issueDrawIndexed(this.pipeline, this.vbuf, this.vOffset, this.ibuf, this.iOffset, this.indexType, indexCount, firstIndex, baseVertex, firstInstance);
  }

  multiDrawIndexed(counts: Int32Array, firstIndices: Int32Array, baseVertices: Int32Array): void {
    if (!this.pipeline || !this.vbuf || !this.ibuf) {
      throw new Error("WebGL2PassEncoder.multiDrawIndexed: setPipeline + setVertexBuffer + setIndexBuffer must precede the draw");
    }
    // W6 (E): WEBGL_multi_draw has NO baseVertex array, so batch ONLY when every baseVertex is 0 (the core
    // path bakes baseVertex into the attribute pointer). Otherwise the drawIndexed loop is the correct
    // fallback (BACKEND W6.A). multiDrawIndexed is not on the render path yet (the future RenderRegion tier).
    let allZeroBase = true;
    for (let i = 0; i < baseVertices.length; i++) {
      if (baseVertices[i] !== 0) {
        allZeroBase = false;
        break;
      }
    }
    if (allZeroBase && this.device.canMultiDraw()) {
      this.device.issueMultiDrawIndexed(this.pipeline, this.vbuf, this.vOffset, this.ibuf, this.iOffset, this.indexType, counts, firstIndices);
    } else {
      for (let i = 0; i < counts.length; i++) this.drawIndexed(counts[i], firstIndices[i], baseVertices[i], 0);
    }
  }

  executeBundles(bundles: RenderBundleHandle[]): void {
    // W6 (D): replay each recorded command against THIS live pass. The replayed drawIndexed thunks already
    // bump drawCalls (issueDrawIndexed), so do NOT add drawCount — that would double-count (WEBGL_FINDINGS:
    // WebGPU counts at replay because its bundle draws are invisible to drawCalls; WebGL2 counts at replay
    // because the thunks ARE real pass.drawIndexed calls). The replayed setPipeline/setVertexBuffer/
    // setIndexBuffer thunks set this encoder's state so the drawIndexed precondition passes.
    for (const b of bundles) {
      for (const thunk of recOf<RenderBundleRec>(b).thunks) thunk(this);
    }
  }

  end(): void {
    // W4: present the offscreen sRGB color onto the default framebuffer (re-encoding linear→sRGB). This runs
    // on every pass end (the only contract seam — a frame is many passes), so the canvas always shows the
    // cumulative frame; the browser auto-presents the default FB at the end of the rAF task.
    this.device.present();
  }
}

/** A recorded F1 bundle: the captured command thunks + the draw count (for parity; WebGL2 doesn't use it —
 *  see executeBundles). GC-managed (no GL objects), so destroy() frees nothing. */
interface RenderBundleRec {
  thunks: ((pass: WebGL2PassEncoder) => void)[];
  drawCount: number;
}

/**
 * Records the neutral command stream (F1) as replayable thunks — WebGL2's stand-in for a GPURenderBundle
 * (BACKEND 4.A). `WebGL2PassEncoder.executeBundles` re-issues them against a live pass, reproducing exactly
 * the GL calls a direct encode would. Handles (pipeline/bindings/buffers) are captured BY REFERENCE (the
 * caller's invalidation key keeps them valid); scalars BY VALUE.
 */
class WebGL2BundleEncoder implements BundleEncoder {
  readonly thunks: ((pass: WebGL2PassEncoder) => void)[] = [];
  drawCount = 0;

  setPipeline(p: PipelineHandle): void {
    this.thunks.push((pass) => pass.setPipeline(p));
  }

  setBindings(b: BindingsHandle, dynamicOffsets?: number[]): void {
    // COPY the dynamic-offset array: a caller may reuse + mutate it per draw (EntityRenderer.dynOff), so a
    // captured reference would replay every draw with the LAST offset. Terrain — the only bundling renderer
    // today — passes none (undefined), but the copy keeps the recorder correct for any future caller.
    const off = dynamicOffsets ? dynamicOffsets.slice() : undefined;
    this.thunks.push((pass) => pass.setBindings(b, off));
  }

  setVertexBuffer(slot: number, buf: GpuBufferHandle, offsetBytes = 0): void {
    this.thunks.push((pass) => pass.setVertexBuffer(slot, buf, offsetBytes));
  }

  setIndexBuffer(buf: GpuBufferHandle, indexType: IndexFormat, offsetBytes = 0): void {
    this.thunks.push((pass) => pass.setIndexBuffer(buf, indexType, offsetBytes));
  }

  drawIndexed(indexCount: number, firstIndex: number, baseVertex: number, firstInstance = 0): void {
    this.drawCount++;
    this.thunks.push((pass) => pass.drawIndexed(indexCount, firstIndex, baseVertex, firstInstance));
  }
}
