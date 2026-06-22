// Submits terrain passes. RENDERER_PLAN.md §3 (frame loop), §12, §15, §24.7.
//
// Draw order: SOLID → CUTOUT (both depth-write; alpha-test discards holes in cutout) → MOVING-TRANSLUCENT
// (the piston transient: blend on, depth-test on, depth-WRITE ON — vanilla `translucentMovingBlock`) →
// TRANSLUCENT terrain (blend on, depth-test on, depth-write OFF — TRAP 12.C). The moving-translucent draw
// writes depth BEFORE terrain translucent, so a moving glass block orders against static terrain glass by
// depth — no geometry merge, no terrain suppression (BUG_REPORT.md "distant glass flicker").
//
// Transparency matches (§12, translucent_sorting/):
//  - Translucent sections are drawn BACK-TO-FRONT by section→camera distance (TRAP 12.B: section
//    order AND per-quad order; neither alone suffices).
//  - Per-quad order comes from the section's index buffer, chosen by its sort type:
//      AnyOrder      → shared quad EBO.
//      StaticNormal  → baked normal-relative order.
//      StaticTopo    → baked topological order.
//      Dynamic       → re-sorted HERE on a GFNI plane-crossing trigger. The re-sort is the
//                      topological sort (the order a BSP would produce); INDICES ONLY (TRAP 12.A).
//  - Back-face culling is ALWAYS ON for translucent terrain — exactly (ShaderChunkRenderer
//    builds every terrain pipeline with `.withCull(true)`, unconditionally) and vanilla
//    (TRANSLUCENT_TERRAIN inherits the default cull = on). `needsDirectionMixing` does NOT
//    touch cull state; it only forces a section's translucent quads into one vertex bucket so the
//    GLOBAL sort can interleave facings. GPU back-face cull is per-triangle by winding — independent
//    of index order — so it culls the same away-facing quads regardless of how the sort interleaves
//    them. (Earlier this path wrongly disabled cull for StaticTopo/Dynamic, leaking glass back faces.)

import type {
  BindingLayoutEntry,
  BindingsHandle,
  BundleEncoder,
  GpuBufferHandle,
  GpuTextureHandle,
  GraphicsDevice,
  PipelineHandle,
  RenderBundleHandle,
  SamplerHandle,
  TextureFormat,
} from "../core/GraphicsDevice";
import { AddressMode, BufferUsage, CompareFn, CullMode, FilterMode, IndexFormat, SamplerBindingType, ShaderStage, TextureFormat as TexFormat, TextureSampleType } from "../core/GraphicsDevice";
import { defaultLightLut, LIGHT_LUT_SIZE } from "./LightLut";
import { PipelineCache } from "../core/PipelineCache";
import { PASS_STATE } from "../core/RenderPassState";
import type { AtlasManager } from "../core/AtlasManager";
import { quadIndices, terrainVertexLayout } from "../mesh/VertexFormat";
import { getVisibleFaces } from "./FaceCull";
import type { TQuad } from "../mesh/TranslucentCollector";
import { computeSortIndex, crossedAnyPlane, farAngleResortAllowed, uniquePlanes, newTopoSortState, GATE_QUAD_THRESHOLD, GATE_DIST_EPS, GATE_ANGLE_COS, GATE_CENTER_LOCAL, type TopoSortState } from "../mesh/SortTrigger";
import { SortType } from "../mesh/SortTypes";
import { clamp, TerrainPass } from "../types";
import { TERRAIN_WGSL } from "./terrainShader";
import type { CameraView, Mat4 } from "../camera/Camera";
import { multiplyInto } from "../camera/math";
import { Frustum } from "../camera/Frustum";
import type { Vec3 } from "../types";
import { instrument } from "../core/Instrument";

const SECTION_SIZE = 16;

/** One section's geometry for one render pass. */
export interface SectionDraw {
  originBlocks: Vec3;
  vertex: GpuBufferHandle;
  /** Byte offset of this section's vertices within `vertex` (a shared per-pass arena, §11). Default 0
   *  for standalone buffers (transient piston/merged draws). Resolved per-frame in collectDraws so an
   *  arena grow/compaction stays invisible to `presented`. */
  vertexOffset?: number;
  /** baseVertex (VERTEX index, = `vertexOffset / 20`) for the bind-once draw path: the arena
   *  vertex buffer is bound ONCE per arena (offset 0) and each section locates its slice via baseVertex
   *  instead of a per-draw `setVertexBuffer(offset)` rebind (DefaultChunkRenderer.RegionChunkRenderer). Vertex
   *  allocations are always `4 verts × 20B × quadCount = 80·q` bytes, so `vertexOffset` is always a multiple of
   *  the 20B stride ⇒ this is exact. Default 0 (standalone buffers / start-of-arena). */
  baseVertex?: number;
  quadCount: number;
  pass: TerrainPass;
  /** Translucent only: the shared index `BufferArena` buffer holding this section's sorted indices (§11).
   *  Absent ⇒ use the shared quad EBO. The DYNAMIC re-sort identity is `cpuVertex`, NOT this (shared) handle. */
  index?: GpuBufferHandle;
  /** Translucent only: byte offset of this section's indices within `index` (the shared arena). Default 0. */
  indexOffset?: number;
  /** Translucent only: sort classification (§12). */
  sortType?: SortType;
  /** Translucent only: per-quad geometry for DYNAMIC topological re-sort. */
  sortQuads?: readonly TQuad[];
  /** Translucent only: draw with depth-WRITE ON via the depth-writing translucent pipeline, BEFORE terrain
   *  translucent (vanilla's `translucentMovingBlock` = LESS_EQUAL + writeDepth). Set ONLY by the moving-piston
   *  transient draw (TransientTranslucent): a self-sorted moving glass block orders against static terrain glass
   *  by DEPTH — no geometry merge, no terrain suppression. Terrain translucent keeps depth-write OFF (TRAP 12.C
   *  — this is unset/false there). */
  depthWrite?: boolean;
  /** Translucent only: retained CPU vertices (packed, 20B stride). The stable per-section identity for the
   *  DYNAMIC re-sort trigger — the shared `index`/`vertex` arena handles can't identify a section (§11), and it
   *  changes on rebuild so the cached trigger state resets when the geometry changes. */
  cpuVertex?: ArrayBuffer;
  /** FS-2/FS-3 (opaque/cutout only): per-facing VERTEX-count runs (canonical order 0..6) when the section
   *  was meshed with `partitionFacing`. The vertices are concatenated in this order, so the draw layer can
   *  emit one drawIndexed per VISIBLE facing run (FS-4) and skip the camera-away slices the GPU back-face-
   *  culls anyway. Vertex-relative ⇒ relocation-safe. Absent ⇒ one legacy single draw. */
  facingVertexCounts?: Uint32Array;
  /** FS-3: bitmask of facings (1<<f) that have geometry — the OR of non-empty `facingVertexCounts` slots.
   *  Lets FS-4 skip the empty slices without re-scanning the counts. Absent ⇒ no partition. */
  sliceMask?: number;
}

// AUDIT F2: viewProj is identical for every draw in a frame, so it lives in ONE shared (non-dynamic)
// frame uniform (binding 0, written once). F2 COMPLETION (P4): the per-section origin lives in a read-only
// STORAGE array (binding 3) indexed by the draw's instance_index (each draw is one instance, drawn with
// firstInstance = its slot). vs the old 256B-aligned dynamic-offset slot per draw, this is a tight 16B/draw
// packed upload (16× less) AND needs NO per-draw bind-group set — one bind serves the whole pass.
// PREC-1: the frame uniform is { mat4 viewProj (offset 0), vec4 camOrigin (offset 64) }. The vec4 rides as a
// SIBLING of the mat4 (std140: vec4 is 16-aligned, a mat4 ends at 64 → camOrigin lands cleanly at offset 64,
// the same on WebGPU and the WebGL2 std140 transpile — BACKEND PREC-1). DEFAULT-OFF the camOrigin field is
// uploaded as (0,0,0,0) and viewProj is the WORLD-space matrix, so the render is byte-identical to pre-PREC-1.
const FRAME_BYTES = 20 * 4; // mat4 viewProj (64) + vec4 camOrigin (16) = 80
const FRAME_CAM_ORIGIN_OFFSET = 16 * 4; // 64 — byte offset of the camOrigin vec4 within the frame uniform
const ORIGIN_FLOATS = 4; // vec4 per draw (origin xyz + pad) — std430 array<vec3> still strides 16B
const ORIGIN_BYTES = ORIGIN_FLOATS * 4; // 16
const DEFAULT_BG: readonly [number, number, number, number] = [0.0049, 0.0058, 0.0067, 1]; // #101316 (app bg), linear
const SECTION_CENTER = 8; // section is 16³, center offset for distance sorting
// AUDIT TR-3: per-section render-thread budget for the DYNAMIC topological re-sort; over budget → the
// distance-sort fallback (keeps a pathological glass section from hitching the frame).
const TOPO_BUDGET_MS = 2;
// TR-1: the distance/angle re-sort GATE for HUGE translucent sections is DEFAULT-OFF (`hugeSortGate`). The
// threshold/eps/center constants are shared with the off-thread site — see SortTrigger.ts (GATE_QUAD_THRESHOLD
// mirrors MAX_TOPO_SORT_QUADS; GATE_DIST_EPS/GATE_ANGLE_COS/GATE_CENTER_LOCAL drive `farAngleResortAllowed`).
// `WORLD_FWD` is the fallback view-forward for a `CameraView` lacking `viewBasis()` — the angle term is then a
// constant (never trips), so the gate is driven by translation only (conservative: just less damping).
const WORLD_FWD: Vec3 = [0, 0, -1];

const BIND_LAYOUT: BindingLayoutEntry[] = [
  { binding: 0, visibility: [ShaderStage.Vertex], type: { kind: "uniform-buffer", dynamicOffset: false } }, // F2: shared viewProj
  { binding: 1, visibility: [ShaderStage.Fragment], type: { kind: "texture", sampleType: TextureSampleType.Float } },
  { binding: 2, visibility: [ShaderStage.Fragment], type: { kind: "sampler", sampler: SamplerBindingType.Filtering } },
  { binding: 3, visibility: [ShaderStage.Vertex], type: { kind: "storage-buffer" } }, // F2 completion: per-draw origin array (instance_index)
  { binding: 4, visibility: [ShaderStage.Fragment], type: { kind: "texture", sampleType: TextureSampleType.Float } }, // P6/LM-1: light LUT
  { binding: 5, visibility: [ShaderStage.Fragment], type: { kind: "sampler", sampler: SamplerBindingType.Filtering } }, // P6/LM-1: light sampler
];

/** Per-DYNAMIC-section trigger state, keyed by its index buffer handle. */
interface SortState {
  cam: [number, number, number]; // section-local camera position the index was last sorted for (mutated in place)
  /** TR-1: the camera FORWARD unit vector the index was last sorted for (the angle reference for the huge-
   *  section gate). Written only when the index is ACTUALLY re-sorted (TRAP TR-1.C — never advance past a
   *  suppressed move). */
  fwd: [number, number, number];
  /** AUDIT TR-2: the section's UNIQUE (normal,dot) planes — packed [nx,ny,nz,dot]×k, computed once. */
  planes: Float32Array;
  /** DynamicTopoData persistent topo-sort disable (P3): a section that repeatedly fails/times-out the
   *  topo sort stops re-attempting it on the render thread and distance-sorts. Reset by a remesh (fresh state
   *  with the fresh `cpuVertex` key). */
  topo: TopoSortState;
}

export class TerrainRenderer {
  private readonly pipelines: PipelineCache;
  private solid: PipelineHandle | null = null;
  private cutout: PipelineHandle | null = null;
  private translucent: PipelineHandle | null = null; // all sort types — back-face cull always on
  private translucentDepthWrite: PipelineHandle | null = null; // moving-piston transient: vanilla `translucentMovingBlock` (depth-WRITE + LESS_EQUAL)
  private originBuf: GpuBufferHandle | null = null; // per-draw section origins, indexed by instance_index (binding 3, storage)
  private frameUniform: GpuBufferHandle | null = null; // shared viewProj, written once/frame (binding 0, F2)
  private bindings: BindingsHandle | null = null;
  private sampler: SamplerHandle | null = null;
  // P6/LM-1: the light LUT (16×16 RGBA8) + its linear/clamp sampler (binding 4/5). A global brightness or
  // day-night change is `setLightLut` → one writeTexture, NO remesh (the only free win — TRAP 6.A). The
  // bind group references the texture by handle, so a LUT rewrite even shows through a replayed F1 bundle.
  private lightLut: GpuTextureHandle | null = null;
  private lightSampler: SamplerHandle | null = null;
  private indexBuf: GpuBufferHandle | null = null;
  private indexCapacityQuads = 0;
  private originCapacity = 0; // capacity of originBuf in draws
  private originData = new Float32Array(0); // CPU staging: ORIGIN_FLOATS per batched draw
  /** GFNI trigger state per DYNAMIC index buffer — only re-sort when the camera crosses a quad plane. */
  private readonly sortState = new WeakMap<object, SortState>();
  // Per-frame cull/draw-list scratch (AUDIT OCC-2 layer 1 + F4): reused across frames so a static
  // camera over a large world allocates nothing per frame. The frustum is a reused zero-alloc instance.
  private readonly frustum = new Frustum();
  private readonly scratchOpaque: SectionDraw[] = [];
  private readonly scratchTranslucent: SectionDraw[] = [];
  /** Depth-writing moving-translucent draws (the piston transient) — drawn BEFORE terrain translucent. */
  private readonly scratchTransient: SectionDraw[] = [];
  private readonly scratchTIndex: GpuBufferHandle[] = [];
  /** Reused one-slot array for `executeBundles` so the static-camera fast path allocates nothing (A5). */
  private readonly bundleScratch: RenderBundleHandle[] = [];
  /** Reused section-local camera scratch for the DYNAMIC re-sort trigger test (A7 — filled in place). */
  private readonly scratchCamLocal: [number, number, number] = [0, 0, 0];
  // ── F1 adaptive render-bundle replay ───────────────────────────────────────────────────────────────
  // A static camera over an unchanged draw list replays a pre-recorded bundle (skipping cull + sort +
  // upload + per-draw encode — the "nearly free" frame). ADAPTIVE: a bundle is only RECORDED after a frame
  // whose (draws, viewProj) matched the previous frame (the camera SETTLED), so a moving/rotating camera —
  // the regression case — never pays the record cost; it stays on the direct-encode path. The validity key
  // is the draw-list array IDENTITY (a commit / arena relocation / occlusion-set change all yield a FRESH
  // array — F3's DrawList + the viewer's occlusion filter) plus the viewProj bits, so a stale bundle can
  // NEVER be replayed against a freed/relocated buffer (TRAP 4.A — over-invalidation is always safe).
  /** Toggle the F1 bundle replay (on by default; off keeps the pure direct-encode path). */
  bundleReplay = true;
  private bundle: RenderBundleHandle | null = null;
  /** F1 split bundles (renderInterleaved): one per pass — opaque (SOLID+CUTOUT) and translucent (moving-glass +
   *  terrain translucent), with the entity pass replayed between them. Either is null if its phase had no draws.
   *  They share the bundleDraws/VP/anchor/layoutRev key + lastDraws/lastVP settle-detection with the single-pass
   *  bundle (a renderer is used in only one mode), gated by `splitBundleValid` (the split analogue of `bundle !== null`). */
  private bundleOpaque: RenderBundleHandle | null = null;
  private bundleTranslucent: RenderBundleHandle | null = null;
  private splitBundleValid = false;
  private bundleDraws: SectionDraw[] | null = null;
  private readonly bundleVP = new Float32Array(16);
  /** INFRA-6: the arena layout revision the bundle was recorded against (defense-in-depth in the replay
   *  key). The draw-list array identity already invalidates on relocation (a fresh array), but folding the
   *  uploader's layoutRev in makes "never replay a bundle against relocated buffers" explicit rather than
   *  reliant on every relocation coincidentally minting a new draws array (TRAP 4.A). Lives/dies with the
   *  bundle (stamped at record; only ever read when `bundle !== null`). */
  private bundleLayoutRev = 0;
  private lastDraws: SectionDraw[] | null = null;
  private readonly lastVP = new Float32Array(16);
  /** F1: render-bundle frames this renderer replayed / recorded (instrumentation; production reads deltas). */
  bundleReplays = 0;
  bundleRecords = 0;
  /** TR-1: when true, DYNAMIC translucent re-sorts run OFF-THREAD (the viewer dispatches them to the worker
   *  pool, generation-matched); the renderer does NO inline re-sort and just draws the committed index. Set
   *  by the viewer when a worker pool is present. Off ⇒ the cheap main-thread re-sort below (no pool). */
  offThreadSort = false;
  /** TR-1: DEFAULT-OFF distance/angle re-sort GATE for HUGE translucent sections (quadCount >= GATE_QUAD_
   *  THRESHOLD). When false (default) every trigger site takes today's exact `crossedAnyPlane`-only path
   *  (the gate expression short-circuits before any new computation), so the shipped render is byte-identical
   *  and the FNV quadHash golden gate is untouched (index-only — no vertex bytes change on either branch).
   *  When true, a huge section additionally requires the camera to have moved/rotated enough RELATIVE TO THE
   *  SECTION CENTER (`farAngleResortAllowed`) before its plane-crossing re-sort fires — an ADDITIONAL damper
   *  layered ON TOP of `crossedAnyPlane` (NEVER a replacement — TRAP TR-1.A), so a barely-moved orbiting camera
   *  does not re-sort a far/huge glass/water volume whose painter's order barely changed. Set by the viewer. */
  hugeSortGate = false;
  /** Instrumentation: index-only re-sorts performed (re-sorts must never remesh vertices — TRAP 12.A). */
  resortCount = 0;
  /** FS-4: when on (and a section was meshed with `partitionFacing`), drop the camera-away opaque/cutout
   *  facing slices before recording draws — pixel-identical (the GPU back-face-culls them) but ~half the
   *  submitted opaque triangles. Default OFF ⇒ the single per-section opaque draw (byte-for-byte the old path). */
  faceSliceCull = false;
  /** FS-5 instrumentation: facing slices culled away (non-empty, camera-away) since construction — proves
   *  the cull actually fired in the pixel-identity harness. */
  sliceDrawsCulled = 0;
  /** PREC-1: DEFAULT-OFF camera-relative origin path for float32 vertex precision at large world coordinates.
   *  When false (default) the frame uniform is uploaded as { world viewProj, camOrigin=(0,0,0,0) } and the
   *  shader's `(origin - camOrigin) + local` with a world matrix is the SAME f32 op sequence as today's
   *  `origin + local` → byte-identical clip coords (the FNV quadHash golden gate is untouched; it covers mesh
   *  bytes, which this change does NOT touch). When true AND the camera exposes `viewMatrix()/projectionMatrix()`,
   *  the renderer builds a TRANSLATION-FREE `viewProjRel = proj · viewRel` (the shared world `viewProjection()`
   *  stays world-space for the frustum/BFS — TRAP PREC-1.B) and uploads camOrigin = the camera block; the large
   *  `origin - camOrigin` term is then cancelled in view space, so the f32 transform runs near zero (no jitter).
   *  `origins[]` stays WORLD-static (the subtraction is per-vertex against the uniform — TRAP PREC-1.C). The
   *  viewer auto-gates this on world-coordinate magnitude (CAMERA_RELATIVE_THRESHOLD). */
  cameraRelative = false;
  /** PREC-1 reused scratch (zero-alloc): the translation-free view + the relative viewProj uploaded when on. */
  private readonly viewRelScratch = new Float32Array(16);
  private readonly viewProjRelScratch = new Float32Array(16);
  /** PREC-1: the camOrigin vec4 (xyz = camPos − anchor sub-section residual, w pad) at frame-uniform offset 64. */
  private readonly camOriginScratch = new Float32Array(4);
  /** PREC-1: the camera-SECTION anchor (integer world-block double) the per-draw origins are uploaded relative
   *  to when cameraRelative is on. [0,0,0] when off ⇒ origins stay world-absolute (byte-identical). Doubles, so
   *  `origin − anchor` is exact even at 30M; only the small relative result is truncated to f32 on upload. */
  private readonly originAnchor = [0, 0, 0];
  /** PREC-1: the anchor the active F1 bundle's origins were baked relative to. A translation-free viewProjRel
   *  does NOT change on a pure-translation camera move, so the anchor must gate bundle replay too (a section
   *  crossing while replaying would otherwise pair stale relative origins with a new-anchor camOrigin). */
  private readonly bundleAnchor = [0, 0, 0];
  /** Per-frame stats (large-world instrumentation): how many section-passes survived culling. */
  stats = { total: 0, drawn: 0, culled: 0 };
  /** Phase-P frame-time split (ms), populated only while `instrument.enabled`. `cullMs` covers
   *  frustum+cull+draw-list build (incl. `resortMs` as a sub-part); `encodeMs` is the pass encode. */
  timing = { cullMs: 0, encodeMs: 0, resortMs: 0 };

  constructor(
    private readonly device: GraphicsDevice,
    private readonly atlas: AtlasManager,
    private readonly colorFormat: TextureFormat,
    private readonly depthFormat: TextureFormat,
    private readonly background: readonly [number, number, number, number] = DEFAULT_BG,
    /** Distance-cull radius in blocks (section center → camera). Infinity = no distance cull. */
    private readonly renderDistanceBlocks: number = Infinity,
  ) {
    this.pipelines = new PipelineCache(device);
  }

  private ensureResources(): void {
    if (this.solid) return;
    const shader = this.device.createShaderModule(TERRAIN_WGSL, "terrain");
    const base = {
      shader,
      vertexEntry: "vs",
      fragmentEntry: "fs",
      vertexLayout: terrainVertexLayout(),
      colorFormat: this.colorFormat,
      depthFormat: this.depthFormat,
      depthCompare: CompareFn.Less,
      bindingLayout: BIND_LAYOUT,
    } as const;
    // Depth/blend per pass come from the canonical PASS_STATE table (the load-bearing transparency rules,
    // §12 / TRAP 12.C) — single source of truth; `cull` (renderer-specific) is layered on per pipeline.
    //
    // ALL terrain passes back-face cull, matching the gold standard — (`ShaderChunkRenderer` builds
    // every terrain pipeline `.withCull(true)`) and vanilla (chunk render layers cull by default). Culling
    // is what makes vanilla's INWARD faces work: the redstone-torch glow box and the spawner/vault
    // `cube_all_inner_faces` cages have faces wound toward the block centre — their near (back-facing) side
    // is culled so you see THROUGH to the far faces + the geometry underneath (the "translucency illusion").
    // The mesher therefore KEEPS every face the model defines (no inner-face/flat-pair dropping); culling,
    // not deletion, hides the back side — so a flat cross still shows from both sides (each side's front face).
    this.solid = this.pipelines.get({ ...base, label: "terrain-solid", pass: { ...PASS_STATE[TerrainPass.Solid], cull: CullMode.Back } });
    this.cutout = this.pipelines.get({ ...base, label: "terrain-cutout", pass: { ...PASS_STATE[TerrainPass.Cutout], cull: CullMode.Back }, constants: { ALPHA_TEST: 1 } });
    // Translucent (depth WRITE off + blend on, from PASS_STATE) ALWAYS back-face culls — matching
    // (`ShaderChunkRenderer` builds every terrain pipeline `.withCull(true)`) and vanilla
    // (TRANSLUCENT_TERRAIN cull default on). The per-section sort only reorders INDICES; back-face
    // culling is per-triangle by winding and independent of that order, so one pipeline serves every
    // sort type (AnyOrder/StaticNormal/StaticTopo/Dynamic). With depth-write off, GPU cull is the only
    // thing that removes a translucent block's away-facing back face — so it must stay on.
    this.translucent = this.pipelines.get({
      ...base, label: "terrain-translucent",
      pass: { ...PASS_STATE[TerrainPass.Translucent], cull: CullMode.Back }, constants: { EMIT_ALPHA: 1 },
    });
    // The moving-piston transient draw — vanilla's `translucentMovingBlock` (RenderTypes): same blend +
    // back-face cull as terrain translucent, but depth-WRITE ON and LESS_EQUAL (DepthStencilState.DEFAULT),
    // drawn BEFORE terrain translucent. The depth write is what orders a moving glass block against static
    // terrain glass WITHOUT merging geometry or suppressing terrain (BUG_REPORT.md). Same shader + EMIT_ALPHA
    // constant as terrain translucent — only depth-write + the compare op differ.
    this.translucentDepthWrite = this.pipelines.get({
      ...base, label: "terrain-translucent-depthwrite",
      pass: { ...PASS_STATE[TerrainPass.Translucent], depthWrite: true, cull: CullMode.Back },
      depthCompare: CompareFn.LessEqual, constants: { EMIT_ALPHA: 1 },
    });

    this.sampler = this.device.createSampler({ mag: FilterMode.Nearest, min: FilterMode.Nearest });
    // P6/LM-1: the light LUT (Rgba8, NOT -srgb — its texels are linear multipliers used directly) + a
    // linear/clamp sampler. The stored lightmap coords hit exact texel centers, so filtering is a no-op at
    // those points; clamp-to-edge keeps an out-of-range coord from wrapping. Default table = full daylight,
    // so a full-bright scene is byte-identical to before lighting (the golden-image safety, TRAP 6.B).
    this.lightLut = this.device.createTexture({ width: LIGHT_LUT_SIZE, height: LIGHT_LUT_SIZE, format: TexFormat.Rgba8, label: "light-lut" });
    this.device.writeTexture(this.lightLut, defaultLightLut());
    this.lightSampler = this.device.createSampler({ mag: FilterMode.Linear, min: FilterMode.Linear, address: AddressMode.Clamp });
    // F2: one shared viewProj uniform for the whole frame (binding 0), written once in render().
    this.frameUniform = this.device.createBuffer({ sizeBytes: FRAME_BYTES, usage: BufferUsage.Uniform, label: "terrain-frame" });
    // The per-draw origin storage buffer + bind group are sized to the frame's draw count (one vec4 per
    // draw, indexed by instance_index) and (re)created in ensureOrigins().
  }

  /** Grow the per-draw origin STORAGE buffer (and rebuild the single shared bind group) to fit `count` draws. */
  private ensureOrigins(count: number): void {
    if (this.bindings && count <= this.originCapacity) return;
    const cap = Math.max(count, 64, this.originCapacity * 2);
    if (this.originBuf) this.device.destroyBuffer(this.originBuf);
    this.originBuf = this.device.createBuffer({ sizeBytes: cap * ORIGIN_BYTES, usage: BufferUsage.Storage, label: "terrain-origins" });
    this.bindings = this.device.createBindings({
      pipeline: this.solid!, // identical bind layout → valid for every terrain pipeline
      group: 0,
      entries: [
        { binding: 0, resource: { buffer: this.frameUniform!, offset: 0, size: FRAME_BYTES } }, // F2: shared viewProj
        { binding: 1, resource: { texture: this.atlas.texture } },
        { binding: 2, resource: { sampler: this.sampler! } },
        { binding: 3, resource: { buffer: this.originBuf } }, // F2 completion: per-draw origin array (whole buffer)
        { binding: 4, resource: { texture: this.lightLut! } }, // P6/LM-1: light LUT (16×16)
        { binding: 5, resource: { sampler: this.lightSampler! } }, // P6/LM-1: light sampler
      ],
    });
    this.originCapacity = cap;
    this.originData = new Float32Array(cap * ORIGIN_FLOATS);
  }

  /**
   * P6/LM-1 — rewrite the light LUT (global brightness / day-night / night-vision). A 16×16 RGBA8 table
   * (`render/LightLut.buildLightLut`); this is ONE `writeTexture` with ZERO remesh jobs — the only
   * genuinely free light win (TRAP 6.A). Per-block light changes are NOT free (they go through the
   * light-only invalidation path — see SectionStore/uploadLight). Safe to call before the first frame
   * (it lazily ensures resources) and during a static-camera F1 bundle replay (the bundle samples the
   * texture by reference, so the new table shows next frame without re-recording).
   */
  setLightLut(rgba: Uint8Array): void {
    this.ensureResources();
    if (rgba.length !== LIGHT_LUT_SIZE * LIGHT_LUT_SIZE * 4) throw new Error(`setLightLut: expected ${LIGHT_LUT_SIZE}×${LIGHT_LUT_SIZE} RGBA8 (${LIGHT_LUT_SIZE * LIGHT_LUT_SIZE * 4} bytes), got ${rgba.length}`);
    this.device.writeTexture(this.lightLut!, rgba);
  }

  private ensureIndex(maxQuads: number): void {
    if (this.indexBuf && maxQuads <= this.indexCapacityQuads) return;
    if (this.indexBuf) this.device.destroyBuffer(this.indexBuf);
    const idx = quadIndices(maxQuads);
    this.indexBuf = this.device.createBuffer({ sizeBytes: idx.byteLength, usage: BufferUsage.Index, label: "shared-quad-index" });
    this.device.writeBuffer(this.indexBuf, 0, idx);
    this.indexCapacityQuads = maxQuads;
  }

  /** Render this frame's SOLID + CUTOUT + TRANSLUCENT sections from `camera`'s view in ONE pass. (Entities
   *  drawn after this paint OVER translucent terrain — for depth-correct entities vs glass, use
   *  `renderInterleaved`, which splits opaque/translucent and runs entities between them.) */
  render(draws: SectionDraw[], camera: CameraView, width: number, height: number, layoutRev = 0): void {
    this.ensureResources();
    const target = { id: null, width, height } as const;
    // `vp` is the WORLD-space view-projection — shared by the frustum cull below AND (via the caller) the
    // occlusion BFS / overlay / explosion renderers. It MUST stay world-space (TRAP PREC-1.B): the precision
    // path NEVER rebuilds it.
    const vp = camera.viewProjection();
    const camPos = camera.position;
    // PREC-1: `vpUpload` is the matrix actually written to the frame uniform AND keyed by the F1 bundle (eq16)
    // — they must agree (TRAP PREC-1.E). OFF (or a camera without view/proj) ⇒ vpUpload = the world vp and
    // camOrigin stays (0,0,0,0) → byte-identical to today. ON ⇒ vpUpload = a translation-free viewProjRel and
    // camOrigin = the camera block. `vpUpload` is bit-static iff the camera is bit-static (a static camera ⇒
    // static camOrigin ⇒ static viewProjRel), so the F1 replay key holds under exactly the same condition.
    const vpUpload = this.computeFrameUpload(camera, vp, camPos);
    // Phase-P frame-time split (gated — production pays nothing). cullMs = cull+draw-list build (incl.
    // resortMs sub-part); encodeMs = pass encode. resortMs = the index-only re-sort.
    const prof = instrument.enabled;
    const tCull0 = prof ? performance.now() : 0;

    // F1 fast path: the draw list is the SAME array AND the camera is bit-static since the bundle was
    // recorded ⇒ every recorded command is still valid (no commit/relocation can have happened without a
    // FRESH draws array). Replay it and skip cull + sort + upload + encode — the "nearly free" static frame.
    // (Animated atlas cells still update: the texture is re-uploaded by the viewer outside the bundle, and
    // the bundle samples it by reference at replay.)
    // PREC-1: the bundle baked the UPLOADED matrix (viewProjRel when on, world vp when off) into its frame
    // uniform, so the replay key compares `vpUpload` to `bundleVP` (TRAP PREC-1.E — key the matrix that was
    // baked, not the always-world `vp`). The replay returns early below WITHOUT rewriting camOrigin — that is
    // correct, not a bug: the replay key requires a bit-static camera (eq16 vpUpload + matching anchor), so the
    // camOrigin already in the frame uniform is unchanged. (P3: the prior comment wrongly implied a per-frame
    // camOrigin write runs on the replay path; it does not — the early return skips it.)
    if (this.bundleReplay && this.bundle !== null && draws === this.bundleDraws && layoutRev === this.bundleLayoutRev && eq16(vpUpload, this.bundleVP)
        && this.originAnchor[0] === this.bundleAnchor[0] && this.originAnchor[1] === this.bundleAnchor[1] && this.originAnchor[2] === this.bundleAnchor[2]) {
      const pass = this.device.beginPass(target, { color: this.background, depth: 1 });
      this.bundleScratch[0] = this.bundle!;
      pass.executeBundles(this.bundleScratch);
      pass.end();
      this.bundleReplays++;
      if (prof) { this.timing.cullMs = 0; this.timing.encodeMs = performance.now() - tCull0; this.timing.resortMs = 0; }
      return;
    }

    const prepped = this.prepareFrame(draws, vp, vpUpload, camPos, camera, prof);
    if (!prepped) {
      this.device.beginPass(target, { color: this.background, depth: 1 }).end();
      this.bundle = null; // nothing to replay this frame — drop any prior bundle (F1 state stays consistent)
      this.bundleDraws = null;
      this.lastDraws = draws;
      this.lastVP.set(vpUpload); // PREC-1: track the UPLOADED matrix (world vp off / viewProjRel on)
      if (prof) { this.timing.cullMs = performance.now() - tCull0; this.timing.encodeMs = 0; this.timing.resortMs = 0; }
      return;
    }
    const { count, resortMs } = prepped;

    // ONE render pass, ONE submit for the whole frame. F1: if this frame's (draws, viewProj) match the
    // PREVIOUS frame's, the camera has settled on a stable draw list → RECORD a reusable bundle now (and
    // execute it) so subsequent identical frames hit the fast path above for ~0 cost. A moving/rotating
    // camera never matches the previous frame → it stays on the direct-encode path, paying no record cost.
    const tEnc0 = prof ? performance.now() : 0;
    const pass = this.device.beginPass(target, { color: this.background, depth: 1 });
    if (this.bundleReplay && this.lastDraws === draws && eq16(vpUpload, this.lastVP)) {
      this.bundle = this.device.createRenderBundle(
        { colorFormat: this.colorFormat, depthFormat: this.depthFormat, label: "terrain-bundle" },
        // The bundle bakes the slice cull for the SETTLED camera; it only ever replays while the camera is
        // bit-static (eq16 vpUpload), so the recorded visible-face mask stays correct (FS-4 / FS-5 0px on replay).
        (enc) => this.encodeDraws(enc, this.scratchOpaque, this.scratchTransient, this.scratchTranslucent, this.scratchTIndex, 0, count, camPos),
      );
      this.bundleDraws = draws;
      this.bundleVP.set(vpUpload); // PREC-1: key the matrix actually baked into the bundle's frame uniform
      this.bundleAnchor[0] = this.originAnchor[0]; this.bundleAnchor[1] = this.originAnchor[1]; this.bundleAnchor[2] = this.originAnchor[2]; // PREC-1: anchor the relative origins were baked at
      this.bundleLayoutRev = layoutRev; // INFRA-6: stamp the layout the bundle's (buffer, offset) snapshots assume
      this.bundleRecords++;
      this.bundleScratch[0] = this.bundle!;
      pass.executeBundles(this.bundleScratch);
    } else {
      this.encodeDraws(pass, this.scratchOpaque, this.scratchTransient, this.scratchTranslucent, this.scratchTIndex, 0, count, camPos); // direct (camera moving / draw list changed)
      this.bundle = null; // any prior bundle was recorded for a different state — never replay it stale
      this.bundleDraws = null;
    }
    pass.end();
    this.lastDraws = draws;
    this.lastVP.set(vpUpload); // PREC-1: settle on the UPLOADED matrix (so next-frame record keys on it)
    if (prof) { this.timing.cullMs = tEnc0 - tCull0; this.timing.encodeMs = performance.now() - tEnc0; this.timing.resortMs = resortMs; }
  }

  /**
   * Vanilla-order render that interleaves an external pass (entities) BETWEEN opaque terrain and translucent
   * terrain — the fix for "entities paint over glass". SOLID+CUTOUT draw first (pass 1, CLEARS color+depth),
   * then `betweenPasses()` runs (the entity geometry, which LOADS + depth-tests against the opaque depth and
   * WRITES depth), then the depth-writing moving-translucent + terrain translucent draw last (pass 2, LOADS).
   * Glass in pass 2 therefore depth-orders against entities (an entity behind glass is tinted by it; an entity
   * in front occludes it) instead of being overpainted. Specials/overlays still run AFTER (the caller draws
   * them past pass 2). F1 is preserved with a SEPARATE bundle per pass — when the camera is bit-static the two
   * are replayed with the entity pass between, skipping cull/sort/upload/encode just like the single pass.
   */
  renderInterleaved(draws: SectionDraw[], camera: CameraView, width: number, height: number, layoutRev: number, betweenPasses: () => void): void {
    this.ensureResources();
    const target = { id: null, width, height } as const;
    const bg = this.background;
    const vp = camera.viewProjection();
    const camPos = camera.position;
    const vpUpload = this.computeFrameUpload(camera, vp, camPos);
    const prof = instrument.enabled;
    const tCull0 = prof ? performance.now() : 0;

    // F1 fast path (split): both phase bundles are valid for this (draws, vp, layout, anchor) ⇒ replay each in
    // its own pass with the entity pass between, skipping cull+sort+upload+encode. Same bit-static-camera key as
    // the single-pass path (TRAP 4.A — a fresh draws array / relocation / camera move drops it). camOrigin/origins
    // are unchanged under a bit-static camera, so (like the single pass) the replay does not rewrite them.
    if (this.bundleReplay && this.splitBundleValid && draws === this.bundleDraws && layoutRev === this.bundleLayoutRev && eq16(vpUpload, this.bundleVP)
        && this.originAnchor[0] === this.bundleAnchor[0] && this.originAnchor[1] === this.bundleAnchor[1] && this.originAnchor[2] === this.bundleAnchor[2]) {
      const p1 = this.device.beginPass(target, { color: bg, depth: 1 });
      if (this.bundleOpaque) { this.bundleScratch[0] = this.bundleOpaque; p1.executeBundles(this.bundleScratch); }
      p1.end();
      betweenPasses();
      const p2 = this.device.beginPass(target, {}); // LOAD terrain's color + depth (entities wrote between)
      if (this.bundleTranslucent) { this.bundleScratch[0] = this.bundleTranslucent; p2.executeBundles(this.bundleScratch); }
      p2.end();
      this.bundleReplays++;
      if (prof) { this.timing.cullMs = 0; this.timing.encodeMs = performance.now() - tCull0; this.timing.resortMs = 0; }
      return;
    }

    const prepped = this.prepareFrame(draws, vp, vpUpload, camPos, camera, prof);
    if (!prepped) {
      // No terrain this frame — still CLEAR, then let entities draw over the cleared frame (no translucent pass).
      this.device.beginPass(target, { color: bg, depth: 1 }).end();
      betweenPasses();
      this.splitBundleValid = false; this.bundleOpaque = null; this.bundleTranslucent = null; this.bundleDraws = null;
      this.lastDraws = draws;
      this.lastVP.set(vpUpload);
      if (prof) { this.timing.cullMs = performance.now() - tCull0; this.timing.encodeMs = 0; this.timing.resortMs = 0; }
      return;
    }
    const { opaqueN, count, resortMs } = prepped;
    const settled = this.bundleReplay && this.lastDraws === draws && eq16(vpUpload, this.lastVP);
    const tEnc0 = prof ? performance.now() : 0;

    // Pass 1 — opaque terrain (SOLID + CUTOUT). CLEARS color + depth (the frame's first pass).
    const p1 = this.device.beginPass(target, { color: bg, depth: 1 });
    if (settled) {
      this.bundleOpaque = opaqueN > 0
        ? this.device.createRenderBundle({ colorFormat: this.colorFormat, depthFormat: this.depthFormat, label: "terrain-bundle-opaque" },
            (enc) => this.encodeDraws(enc, this.scratchOpaque, this.scratchTransient, this.scratchTranslucent, this.scratchTIndex, 0, opaqueN, camPos))
        : null;
      if (this.bundleOpaque) { this.bundleScratch[0] = this.bundleOpaque; p1.executeBundles(this.bundleScratch); }
    } else if (opaqueN > 0) {
      this.encodeDraws(p1, this.scratchOpaque, this.scratchTransient, this.scratchTranslucent, this.scratchTIndex, 0, opaqueN, camPos);
    }
    p1.end();

    // Entities (and anything else the caller injects) — they LOAD + depth-test against the opaque depth and
    // write depth, so the glass drawn in pass 2 orders correctly against them.
    betweenPasses();

    // Pass 2 — moving-translucent (depth-write) + terrain translucent. LOADS (depth from passes 1 + entities).
    const p2 = this.device.beginPass(target, {});
    if (settled) {
      this.bundleTranslucent = count > opaqueN
        ? this.device.createRenderBundle({ colorFormat: this.colorFormat, depthFormat: this.depthFormat, label: "terrain-bundle-translucent" },
            (enc) => this.encodeDraws(enc, this.scratchOpaque, this.scratchTransient, this.scratchTranslucent, this.scratchTIndex, opaqueN, count, camPos))
        : null;
      if (this.bundleTranslucent) { this.bundleScratch[0] = this.bundleTranslucent; p2.executeBundles(this.bundleScratch); }
    } else if (count > opaqueN) {
      this.encodeDraws(p2, this.scratchOpaque, this.scratchTransient, this.scratchTranslucent, this.scratchTIndex, opaqueN, count, camPos);
    }
    p2.end();

    if (settled) {
      this.bundleDraws = draws;
      this.bundleVP.set(vpUpload);
      this.bundleAnchor[0] = this.originAnchor[0]; this.bundleAnchor[1] = this.originAnchor[1]; this.bundleAnchor[2] = this.originAnchor[2];
      this.bundleLayoutRev = layoutRev;
      this.splitBundleValid = true;
      this.bundleRecords++;
    } else {
      this.splitBundleValid = false; this.bundleOpaque = null; this.bundleTranslucent = null; this.bundleDraws = null;
    }
    this.lastDraws = draws;
    this.lastVP.set(vpUpload);
    if (prof) { this.timing.cullMs = tEnc0 - tCull0; this.timing.encodeMs = performance.now() - tEnc0; this.timing.resortMs = resortMs; }
  }

  /** Frustum-cull + bucket the draws (opaque / depth-writing transient / terrain translucent), resolve each
   *  translucent section's index, and upload the per-draw origins + the frame uniform. Shared by the single-pass
   *  `render` and `renderInterleaved`; called only on the NON-replay path (a replay skips all of this). Returns
   *  the slot counts (`opaqueN` = the opaque/translucent boundary, also the pass-1/pass-2 split; `count` = total),
   *  or null when nothing is visible (the caller clears the frame). Fills `scratchOpaque/Transient/Translucent`
   *  + `scratchTIndex` (transient indices first, then terrain translucent) that encodeDraws reads. */
  private prepareFrame(draws: SectionDraw[], vp: Mat4, vpUpload: Mat4, camPos: Vec3, camera: CameraView, prof: boolean): { opaqueN: number; count: number; resortMs: number } | null {
    // §14 v1 culling into REUSED scratch (AUDIT OCC-2 layer 1 + F4): skip section-passes outside the view
    // frustum or beyond the render distance. The frustum is a reused zero-alloc instance and the section box
    // test takes scalars; the draw buckets are reused arrays, so a static camera allocates nothing here.
    this.frustum.setFromViewProjection(vp);
    const R = this.renderDistanceBlocks;
    const opaque = this.scratchOpaque;
    const translucent = this.scratchTranslucent;
    const transient = this.scratchTransient; // depth-writing moving translucent — drawn BEFORE terrain translucent
    opaque.length = 0;
    translucent.length = 0;
    transient.length = 0;
    let total = 0;
    for (let i = 0; i < draws.length; i++) {
      const d = draws[i];
      if (d.quadCount === 0) continue;
      total++;
      const o = d.originBlocks;
      if (!this.frustum.testSection(o[0], o[1], o[2])) continue;
      if (R !== Infinity && !withinRenderDistance(o, camPos, R)) continue;
      if (d.pass === TerrainPass.Translucent) (d.depthWrite ? transient : translucent).push(d);
      else opaque.push(d);
    }
    opaque.sort((a, b) => a.pass - b.pass); // SOLID before CUTOUT (≤2 pipeline switches)
    this.stats.total = total; this.stats.drawn = opaque.length + transient.length + translucent.length; this.stats.culled = total - this.stats.drawn;
    if (opaque.length === 0 && transient.length === 0 && translucent.length === 0) return null;

    let maxQuads = 0;
    for (let i = 0; i < opaque.length; i++) if (opaque[i].quadCount > maxQuads) maxQuads = opaque[i].quadCount;
    for (let i = 0; i < transient.length; i++) if (transient[i].quadCount > maxQuads) maxQuads = transient[i].quadCount;
    for (let i = 0; i < translucent.length; i++) if (translucent[i].quadCount > maxQuads) maxQuads = translucent[i].quadCount;
    this.ensureIndex(maxQuads);
    transient.sort((a, b) => sectionDist2(b, camPos) - sectionDist2(a, camPos)); // farthest-first (≤1 in practice)
    translucent.sort((a, b) => sectionDist2(b, camPos) - sectionDist2(a, camPos)); // farthest-first

    // Resolve each translucent section's index (DYNAMIC may re-sort + writeBuffer) BEFORE the pass opens.
    // The transient (moving-translucent) indices come FIRST in `tIndex`, matching the encode order (moving
    // translucent is drawn before terrain translucent); its draws are StaticTopo (pre-sorted) ⇒ no re-sort.
    // TR-1: the camera FORWARD is the huge-section re-sort gate's angle reference — read it ONCE per frame
    // (zero-alloc memoized on `Camera`; `worldFwd` fallback when a camera doesn't expose `viewBasis`). Only the
    // gate (default-off, huge sections) reads it, so off-path this is a single cheap basis read.
    const camFwd = camera.viewBasis ? camera.viewBasis().forward : WORLD_FWD;
    const tIndex = this.scratchTIndex;
    tIndex.length = 0;
    const tRes0 = prof ? performance.now() : 0;
    for (let i = 0; i < transient.length; i++) tIndex.push(this.resolveTranslucentIndex(transient[i], camPos, camFwd));
    for (let i = 0; i < translucent.length; i++) tIndex.push(this.resolveTranslucentIndex(translucent[i], camPos, camFwd));
    const resortMs = prof ? performance.now() - tRes0 : 0;

    // F2: the view-projection is shared across the whole frame — write it ONCE to the frame uniform (binding 0).
    // PREC-1: write the UPLOADED matrix (world vp when off / viewProjRel when on) at offset 0 and the camOrigin
    // vec4 at offset 64. The camOrigin write rides OUTSIDE the F1 bundle (a frame-uniform writeBuffer), so it
    // updates even on a replayed frame and never invalidates the bundle.
    this.device.writeBuffer(this.frameUniform!, 0, vpUpload);
    this.device.writeBuffer(this.frameUniform!, FRAME_CAM_ORIGIN_OFFSET, this.camOriginScratch);
    // F2 completion: one tightly-packed vec4 per draw in the origin storage array (binding 3), indexed by
    // instance_index. Order = opaque, then moving-translucent (transient), then terrain translucent — the
    // SAME order encodeDraws emits, so firstInstance = slot picks the right origin. One 16B/draw upload.
    const opaqueN = opaque.length;
    const transientN = transient.length;
    const count = opaqueN + transientN + translucent.length;
    this.ensureOrigins(count);
    const origins = this.originData;
    for (let i = 0; i < count; i++) {
      const d = i < opaqueN ? opaque[i] : i < opaqueN + transientN ? transient[i - opaqueN] : translucent[i - opaqueN - transientN];
      const base = i * ORIGIN_FLOATS;
      const o = d.originBlocks;
      // PREC-1: upload RELATIVE to the camera-section anchor (computed in double, so `origin − anchor` is exact).
      // Off-path the anchor is [0,0,0] ⇒ world-absolute origins ⇒ byte-identical to before.
      origins[base + 0] = o[0] - this.originAnchor[0];
      origins[base + 1] = o[1] - this.originAnchor[1];
      origins[base + 2] = o[2] - this.originAnchor[2];
      origins[base + 3] = 0;
    }
    this.device.writeBuffer(this.originBuf!, 0, origins.subarray(0, count * ORIGIN_FLOATS));
    return { opaqueN, count, resortMs };
  }

  /**
   * PREC-1 — produce the matrix to upload to the frame uniform (offset 0) and fill `camOriginScratch` (the
   * vec4 written at offset 64). Returns a matrix that is bit-static iff the camera is bit-static.
   *
   *  - OFF (or a camera lacking `viewMatrix()/projectionMatrix()`): returns the WORLD-space `vp` unchanged and
   *    sets camOrigin = (0,0,0,0). The shader's `(origin - 0) + local` with a world matrix is then the SAME
   *    f32 op sequence as the pre-PREC-1 `origin + local` → byte-identical clip coords.
   *  - ON: builds `viewRel` = the view matrix with its translation column zeroed (rows 12/13/14 of the
   *    column-major view), `viewProjRel = proj · viewRel` into reused scratch (the shared world `viewProjection()`
   *    is NEVER touched — TRAP PREC-1.B), and camOrigin = the camera position rounded to an integer block (so
   *    `origin - camOrigin` is an exact small integer when origins are block-granular — no new rounding source).
   */
  private computeFrameUpload(camera: CameraView, vp: Mat4, camPos: Vec3): Mat4 {
    const co = this.camOriginScratch;
    const anc = this.originAnchor;
    // OFF path (and the safe fallback when the camera can't supply the two matrices): world vp + camOrigin 0 +
    // anchor 0 ⇒ the per-draw origins stay world-absolute and the shader's `(origin − 0) + local` is the exact
    // f32 op sequence as before (byte-identical).
    if (!this.cameraRelative || !camera.viewMatrix || !camera.projectionMatrix) {
      co[0] = 0; co[1] = 0; co[2] = 0; co[3] = 0;
      anc[0] = 0; anc[1] = 0; anc[2] = 0;
      return vp;
    }
    const view = camera.viewMatrix();
    const proj = camera.projectionMatrix();
    // CPU-side DOUBLE-precision camera-relative origins. The anchor is the camera's SECTION origin (an integer
    // world-block double). The per-draw origins are uploaded RELATIVE to it (`origin − anchor`, an EXACT small
    // integer in double — f32-exact on upload even at 30M), and camOrigin is the sub-section residual
    // `camPos − anchor` (∈[0,16) per axis). The shader then forms (origin − anchor) − (camPos − anchor) + local =
    // origin − camPos + local entirely from SMALL f32 values ⇒ NO large-coordinate precision loss. (A naive
    // camera-relative that subtracts world-absolute f32 origins in the shader inherits the ~ULP-2 error of the
    // 30M operands and fixes nothing — the subtraction MUST happen in double, before the f32 truncation.)
    anc[0] = Math.floor(camPos[0] / 16) * 16;
    anc[1] = Math.floor(camPos[1] / 16) * 16;
    anc[2] = Math.floor(camPos[2] / 16) * 16;
    co[0] = camPos[0] - anc[0]; co[1] = camPos[1] - anc[1]; co[2] = camPos[2] - anc[2]; co[3] = 0;
    // viewRel = view with the translation column zeroed (pure rotation R = the upper-left of view = R·T(−eye)).
    const vr = this.viewRelScratch;
    vr.set(view);
    vr[12] = 0; vr[13] = 0; vr[14] = 0;
    multiplyInto(this.viewProjRelScratch, proj, vr);
    return this.viewProjRelScratch;
  }

  /** Record this frame's SOLID → CUTOUT → MOVING-TRANSLUCENT → TRANSLUCENT draws into `enc` (a live pass OR a
   *  render bundle). `transient` is the depth-WRITING moving-translucent group (vanilla `translucentMovingBlock`),
   *  emitted between opaque and terrain translucent; `tIndex` holds the transient indices first, then the terrain
   *  translucent ones (same order). Each draw selects its section origin via firstInstance = its slot
   *  (instance_index → origins[slot]), so the bind group is set ONCE (no per-draw dynamic-offset bind — F2
   *  completion). setPipeline/setIndexBuffer are GUARDED (AUDIT F7): opaque is pass-sorted (≤2 pipelines, all
   *  share the EBO) so redundant sets skip. */
  private encodeDraws(enc: BundleEncoder, opaque: SectionDraw[], transient: SectionDraw[], translucent: SectionDraw[], tIndex: GpuBufferHandle[], start: number, end: number, camPos: Vec3): void {
    enc.setBindings(this.bindings!); // one bind for the whole pass/bundle (origins indexed per draw)
    let lastPipeline: PipelineHandle | null = null;
    let lastVertex: GpuBufferHandle | null = null;
    let lastVertexBindOffset = -1;
    let lastIndex: GpuBufferHandle | null = null;
    let lastIndexOffset = -1;
    const opaqueN = opaque.length;
    const transientN = transient.length;
    // `[start, end)` is a slot RANGE into the combined [opaque…, transient…, translucent…] ordering, so a
    // split caller can encode just opaque (pass 1) or just the translucent groups (pass 2); `i` stays the
    // GLOBAL slot, so firstInstance = i and the tIndex slot (i − opaqueN) are correct for either range.
    for (let i = start; i < end; i++) {
      const isOpaque = i < opaqueN;
      const ti = i - opaqueN; // slot within the translucent groups (transient first, then terrain) == tIndex slot
      const isTransient = !isOpaque && ti < transientN;
      const d = isOpaque ? opaque[i] : isTransient ? transient[ti] : translucent[ti - transientN];
      const pipeline = isOpaque ? (d.pass === TerrainPass.Cutout ? this.cutout! : this.solid!) : isTransient ? this.translucentDepthWrite! : this.translucent!;
      const index = isOpaque ? this.indexBuf! : tIndex[ti];
      // Opaque + AnyOrder use the shared EBO at offset 0; sorted translucent indexes into the shared index
      // arena at its per-section byte offset. Guard on BOTH buffer and offset (the arena buffer is the same
      // across translucent draws — only the offset differs), so the opaque run still skips redundant sets (F7).
      const indexOffset = isOpaque || index === this.indexBuf ? 0 : d.indexOffset ?? 0;
      // bind-once (DefaultChunkRenderer): when `baseVertex` is present (real geometry — its arena offset
      // is stride-aligned) the shared VERTEX ARENA is bound ONCE at offset 0 and each section locates its slice
      // via baseVertex, NOT a per-draw `setVertexBuffer(offset)` rebind. Consecutive sections share the arena
      // handle ⇒ one bind serves the whole run. A draw WITHOUT baseVertex (transient/standalone or a synthetic
      // non-stride-aligned fixture) falls back to binding at its byte offset (baseVertex 0) — the legacy path.
      // The byte the GPU fetches is identical: base·20 + index·20 == `setVertexBuffer(offset)` + baseVertex 0.
      const base = d.baseVertex ?? 0;
      const bindOffset = d.baseVertex !== undefined ? 0 : d.vertexOffset ?? 0;
      if (pipeline !== lastPipeline) {
        enc.setPipeline(pipeline);
        lastPipeline = pipeline;
      }
      if (d.vertex !== lastVertex || bindOffset !== lastVertexBindOffset) {
        enc.setVertexBuffer(0, d.vertex, bindOffset);
        lastVertex = d.vertex;
        lastVertexBindOffset = bindOffset;
      }
      if (index !== lastIndex || indexOffset !== lastIndexOffset) {
        enc.setIndexBuffer(index, IndexFormat.Uint32, indexOffset);
        lastIndex = index;
        lastIndexOffset = indexOffset;
      }
      // FS-4: an opaque/cutout section meshed with partitionFacing draws one drawIndexed per VISIBLE facing
      // run (camera-away slices skipped — the GPU back-face-culls them, so this is pixel-identical). The
      // section's vertices start at `base` (baseVertex) within the bound arena; the run's local vertex offset
      // adds to it, indexCount is in INDICES (runQuads*6), firstInstance stays `i` (same origin slot).
      if (this.faceSliceCull && isOpaque && d.facingVertexCounts) {
        const counts = d.facingVertexCounts;
        const vis = getVisibleFaces(camPos[0], camPos[1], camPos[2], d.originBlocks[0], d.originBlocks[1], d.originBlocks[2]);
        let cursor = 0; // running VERTEX offset within the section's concatenated facing runs
        let runStart = -1; // open run's start vertex (-1 = none)
        let runVerts = 0;
        for (let f = 0; f < counts.length; f++) {
          const c = counts[f];
          if (c > 0) {
            if ((vis & (1 << f)) !== 0) {
              if (runStart < 0) { runStart = cursor; runVerts = 0; }
              runVerts += c; // coalesce adjacent visible (or empty) facings into one draw
            } else {
              if (runStart >= 0) { enc.drawIndexed((runVerts >> 2) * 6, 0, base + runStart, i); runStart = -1; runVerts = 0; }
              this.sliceDrawsCulled++; // a non-empty camera-away slice — dropped
            }
          }
          cursor += c;
        }
        if (runStart >= 0) enc.drawIndexed((runVerts >> 2) * 6, 0, base + runStart, i);
      } else {
        enc.drawIndexed(d.quadCount * 6, 0, base, i); // firstInstance = i → shader reads origins[i]
      }
    }
  }

  /** Pick (and, for DYNAMIC, re-sort) the index buffer a translucent section draws with. `camFwd` is the
   *  camera forward (TR-1 huge-section gate angle reference; ignored unless `hugeSortGate` is on). */
  private resolveTranslucentIndex(d: SectionDraw, camPos: Vec3, camFwd: Vec3): GpuBufferHandle {
    // DYNAMIC sort state is keyed on `cpuVertex` (the per-section ArrayBuffer) — `index` is now a SHARED
    // arena buffer (§11) and can't identify a section. cpuVertex is present for every translucent section
    // and changes on rebuild, so the cached trigger state resets correctly when geometry changes.
    // TR-1: when `offThreadSort` is on, the re-sort runs on the worker layer (the viewer dispatches it,
    // generation-matched) — the renderer just draws the committed index (old until the new one commits, no
    // pop) and does NO inline re-sort (it would double-work + race the worker's in-place index update).
    if (!this.offThreadSort && d.sortType === SortType.Dynamic && d.index && d.cpuVertex && d.sortQuads && d.sortQuads.length > 0) {
      // A7: fill the reused scratch in place for the trigger TEST (no per-frame Vec3); only persist an OWNED
      // copy into `state.cam` when actually re-sorting (state.cam is retained across frames for the next test).
      const camLocal = this.scratchCamLocal;
      camLocal[0] = camPos[0] - d.originBlocks[0];
      camLocal[1] = camPos[1] - d.originBlocks[1];
      camLocal[2] = camPos[2] - d.originBlocks[2];
      let state = this.sortState.get(d.cpuVertex);
      // GFNI trigger: re-sort only when the camera crossed one of this section's quad planes. The unique
      // (normal,dot) plane set is camera-independent, so it's computed ONCE here and cached (TR-2) — the
      // per-frame test is then O(unique planes), not O(quads).
      if (!state) {
        // First sight: ALWAYS sort (no prior index to fall back on — never gated, TRAP TR-1: leaving it gated
        // would draw the possibly-bad mesh-time index until the camera happens to move past eps).
        state = { cam: [camLocal[0], camLocal[1], camLocal[2]], fwd: [camFwd[0], camFwd[1], camFwd[2]], planes: uniquePlanes(d.sortQuads), topo: newTopoSortState() };
        this.sortState.set(d.cpuVertex, state);
        this.writeDynamicIndex(d, camLocal, state.topo);
        this.resortCount++;
      } else if (
        crossedAnyPlane(state.planes, state.cam, camLocal) &&
        // TR-1: huge-section damper layered ON TOP of crossedAnyPlane (NEVER a replacement — TRAP TR-1.A).
        // Below the threshold OR with the gate off ⇒ today's exact behavior (TRAP TR-1.B). Only a HUGE section
        // with the gate on additionally requires moved/rotated-enough (center-relative). Short-circuits on
        // !hugeSortGate before any new computation, so the default render is byte-identical.
        (!this.hugeSortGate || d.quadCount < GATE_QUAD_THRESHOLD ||
          farAngleResortAllowed(GATE_CENTER_LOCAL, state.cam, camLocal, state.fwd, camFwd, GATE_DIST_EPS, GATE_ANGLE_COS))
      ) {
        this.writeDynamicIndex(d, camLocal, state.topo);
        // TRAP TR-1.C: advance state.cam/state.fwd ONLY inside the branch that actually re-sorts — the deadband
        // must be measured from the last ACTUAL sort, or accumulated drift silently pushes the order stale.
        state.cam[0] = camLocal[0]; state.cam[1] = camLocal[1]; state.cam[2] = camLocal[2];
        state.fwd[0] = camFwd[0]; state.fwd[1] = camFwd[1]; state.fwd[2] = camFwd[2];
        this.resortCount++;
      }
      return d.index;
    }
    // StaticNormal / StaticTopo → baked static index; AnyOrder → the shared quad EBO.
    return d.index ?? this.indexBuf!;
  }

  /** Re-sort a DYNAMIC section's index for `camLocal` and upload it — INDICES ONLY (TRAP 12.A), into this
   *  section's slice of the shared index arena at its byte offset. `topo` carries the persistent-disable
   *  state so a chronically failing/slow section stops re-attempting topo on the render thread. */
  private writeDynamicIndex(d: SectionDraw, camLocal: Vec3, topo: TopoSortState): void {
    const idx = computeSortIndex(d.sortQuads as TQuad[], camLocal, TOPO_BUDGET_MS, topo);
    this.device.writeBuffer(d.index!, d.indexOffset ?? 0, idx);
  }

  dispose(): void {
    if (this.indexBuf) this.device.destroyBuffer(this.indexBuf);
    if (this.originBuf) this.device.destroyBuffer(this.originBuf);
    if (this.frameUniform) this.device.destroyBuffer(this.frameUniform);
    if (this.lightLut) this.device.destroyTexture(this.lightLut);
    this.pipelines.dispose();
    this.solid = null;
    this.cutout = null;
    this.translucent = null;
    this.translucentDepthWrite = null;
    this.bindings = null;
    this.indexBuf = null;
    this.originBuf = null;
    this.frameUniform = null;
    this.lightLut = null;
    this.bundle = null; // GC-managed (no GPU destroy); the buffers it referenced are freed above
    this.bundleOpaque = null;
    this.bundleTranslucent = null;
    this.splitBundleValid = false;
    this.bundleDraws = null;
    this.lastDraws = null;
  }
}

/** Are two 16-float view-projection matrices bit-identical? (F1 bundle validity — a bit-static camera.) */
function eq16(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  for (let i = 0; i < 16; i++) if (a[i] !== b[i]) return false;
  return true;
}

/**
 * Is a 16³ section (world origin `o`) within render distance of `camPos`? This is 
 * `OcclusionCuller.testDistance` — vanilla's **cylindrical fog**, NOT a sphere to the section center:
 *  - distance is measured to the **nearest point** of the section box (camera clamped into
 *    `[o-1, o+17]` — the section + a 1-block apron), so a section is culled only when its WHOLE
 *    volume is out of range (never when its near edge is still visible);
 *  - the horizontal (XZ) and vertical (Y) distances are tested **separately** (`xz² < R²` AND
 *    `|y| < R`).
 * The naive sphere-to-center test over-culls far-horizon sections whose nearest edge is in view —
 * the visible-terrain "holes" behind hills. See / OcclusionCuller.visitNode.
 */
export function withinRenderDistance(o: Vec3, camPos: Vec3, renderDistanceBlocks: number): boolean {
  if (renderDistanceBlocks === Infinity) return true;
  const dx = clamp(camPos[0], o[0] - 1, o[0] + SECTION_SIZE + 1) - camPos[0];
  const dy = clamp(camPos[1], o[1] - 1, o[1] + SECTION_SIZE + 1) - camPos[1];
  const dz = clamp(camPos[2], o[2] - 1, o[2] + SECTION_SIZE + 1) - camPos[2];
  return dx * dx + dz * dz < renderDistanceBlocks * renderDistanceBlocks && Math.abs(dy) < renderDistanceBlocks;
}

/** Squared distance from the camera to a section's center (back-to-front section ordering). */
function sectionDist2(d: SectionDraw, camPos: Vec3): number {
  const cx = d.originBlocks[0] + SECTION_CENTER - camPos[0];
  const cy = d.originBlocks[1] + SECTION_CENTER - camPos[1];
  const cz = d.originBlocks[2] + SECTION_CENTER - camPos[2];
  return cx * cx + cy * cy + cz * cz;
}
