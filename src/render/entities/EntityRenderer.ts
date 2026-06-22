// Entity / block-entity forward pass. RENDERER_PLAN.md §18, Phase 4.5 (GATE 18.1: native box models,
// not glTF). Draws entities in a render pass that LOADS terrain's color + depth (shared depth buffer —
// resolves TRAP 18.B), after terrain. Each entity carries a model matrix + tint/flash uniform and an
// atlas texture (block atlas for block-display entities; entity atlas for box models). One dynamic-offset
// uniform slot per draw, batched like TerrainRenderer.
//
// Depth ordering vs translucent terrain (FIXED): the viewer drives this pass via
// `TerrainRenderer.renderInterleaved`, which splits terrain into an opaque pass and a translucent pass and
// runs THIS entity pass BETWEEN them (vanilla order: opaque terrain → entities → translucent terrain). So an
// entity behind glass is now tinted by it (the glass blends over the entity in the later pass) and an entity
// in front occludes it — entities no longer paint over already-blended translucent terrain. (The legacy
// all-in-one `EntityWorld.render` still draws after a single combined terrain pass — used only by tests/
// harnesses that don't interleave.)

import type {
  BindingLayoutEntry,
  BindingsHandle,
  ClearDesc,
  GpuBufferHandle,
  GpuTextureHandle,
  GraphicsDevice,
  PipelineHandle,
  SamplerHandle,
  TextureFormat,
} from "../../core/GraphicsDevice";
import { BufferUsage, CompareFn, CullMode, FilterMode, IndexFormat, SamplerBindingType, ShaderStage, TextureSampleType } from "../../core/GraphicsDevice";
import { PipelineCache } from "../../core/PipelineCache";
import { quadIndices, terrainVertexLayout } from "../../mesh/VertexFormat";
import { dist2, TerrainPass } from "../../types";
import type { Vec3 } from "../../types";
import { multiplyInto } from "../../camera/math";
import { ENTITY_WGSL } from "./entityShader";

export type Mat4 = Float32Array; // column-major, length 16

/** One entity's geometry for one pass, with its world transform + per-entity colour state. */
export interface EntityDraw {
  vertex: GpuBufferHandle;
  quadCount: number;
  pass: TerrainPass;
  /** Atlas to sample (block atlas for block-display entities; entity atlas for box models). */
  texture: GpuTextureHandle;
  /** Model→world matrix (places + orients + scales the model-local geometry). */
  model: Mat4;
  /** Linear rgba multiply (sheep wool dye etc.); default opaque white. */
  tint?: readonly [number, number, number, number];
  /** Linear flash rgb + mix amount (TNT white blink / hurt flash); default none. */
  flash?: readonly [number, number, number, number];
  /** Diffuse light dir 0 (xyz) + mode flag (w > 0.5 ⇒ vanilla item diffuse lighting); default axis-shade. */
  light0?: readonly [number, number, number, number];
  /** Diffuse light dir 1 (xyz); paired with `light0`. */
  light1?: readonly [number, number, number, number];
  /** Sort key: world position for back-to-front translucent ordering. ALWAYS absolute world coords (even when
   *  `modelRelative` is set — sorting is a double-precision distance, it must not be camera-relative). */
  sortPos?: Vec3;
  /** M6: this `model` was already built camera-relative (its world-placement translation was anchored via
   *  `Mat4Frame.Tw` in DOUBLE), so the renderer's camera-relative path must NOT subtract the anchor again — it
   *  uploads `model` as-is (no f32 precision loss). Absent ⇒ `model` is absolute → the renderer subtracts the
   *  rounded camera origin in f32 (the legacy path, retained for the BE-static / sortOf-derived sites). */
  modelRelative?: boolean;
  /** Back-face cull (solid block items: closed, terrain-wound, positive-determinant model). Default off. */
  cull?: boolean;
}

const UNIFORM_FLOATS = 48; // mat4 viewProj(16) + mat4 model(16) + tint(4) + flash(4) + light0(4) + light1(4)
const UNIFORM_BYTES = UNIFORM_FLOATS * 4; // 192
const SLOT_BYTES = 256; // minUniformBufferOffsetAlignment guaranteed max
const SLOT_FLOATS = SLOT_BYTES / 4; // 64
const WHITE: readonly [number, number, number, number] = [1, 1, 1, 1];
const NO_FLASH: readonly [number, number, number, number] = [1, 1, 1, 0];
const NO_LIGHT: readonly [number, number, number, number] = [0, 0, 0, 0]; // w = 0 ⇒ axis shade (default)

const BIND_LAYOUT: BindingLayoutEntry[] = [
  { binding: 0, visibility: [ShaderStage.Vertex, ShaderStage.Fragment], type: { kind: "uniform-buffer", dynamicOffset: true } },
  { binding: 1, visibility: [ShaderStage.Fragment], type: { kind: "texture", sampleType: TextureSampleType.Float } },
  { binding: 2, visibility: [ShaderStage.Fragment], type: { kind: "sampler", sampler: SamplerBindingType.Filtering } },
];

export class EntityRenderer {
  private readonly pipelines: PipelineCache;
  private solid: PipelineHandle | null = null;
  private cutout: PipelineHandle | null = null;
  private translucent: PipelineHandle | null = null;
  private solidCull: PipelineHandle | null = null;
  private cutoutCull: PipelineHandle | null = null;
  private translucentCull: PipelineHandle | null = null;
  private sampler: SamplerHandle | null = null;
  private uniform: GpuBufferHandle | null = null;
  private uniformCapacitySlots = 0;
  private slotData = new Float32Array(0);
  private indexBuf: GpuBufferHandle | null = null;
  private indexCapacityQuads = 0;
  // F8: reused per-frame partition/sort scratch — no fresh filter/spread arrays each render.
  private readonly opaqueScratch: EntityDraw[] = [];
  private readonly translucentScratch: EntityDraw[] = [];
  private readonly itemsScratch: EntityDraw[] = [];
  /** One bind group per atlas texture (uniform is dynamic-offset, so it's shared across draws). */
  private readonly bindingsByTexture = new Map<GpuTextureHandle, BindingsHandle>();
  /** Reused 1-element dynamic-offset array (AUDIT F7) — no per-draw `[i*SLOT_BYTES]` allocation. */
  private readonly dynOff = [0];
  /** PREC-1 (entity parity): DEFAULT-OFF camera-relative origin path. Entities NEVER defer (TRAP PREC-1.D), so
   *  their precision must move in lockstep with terrain or they jitter against stable terrain. When false
   *  (default) `render` uploads the WORLD viewProj + each model's UNMODIFIED world transform → byte-identical.
   *  When true AND `render` is given the separate view/proj, it uploads a TRANSLATION-FREE viewProjRel and
   *  pre-subtracts the camera origin from every model's translation column on the CPU, so `model · local`
   *  lands near the camera and the f32 transform keeps full precision. The viewer sets this from the same
   *  auto-gate as the terrain renderer. */
  cameraRelative = false;
  /** PREC-1 reused scratch: translation-free view, the relative viewProj, and the per-entity relative model. */
  private readonly viewRelScratch = new Float32Array(16);
  private readonly viewProjRelScratch = new Float32Array(16);
  private readonly modelRelScratch = new Float32Array(16);
  /** Per-frame instrumentation. */
  stats = { drawn: 0 };

  constructor(
    private readonly device: GraphicsDevice,
    private readonly colorFormat: TextureFormat,
    private readonly depthFormat: TextureFormat,
  ) {
    this.pipelines = new PipelineCache(device);
  }

  private ensureResources(): void {
    if (this.solid) return;
    const shader = this.device.createShaderModule(ENTITY_WGSL, "entity");
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
    // Cull NONE for v1 (entity models aren't guaranteed closed/consistent-wound everywhere; the 1px
    // generated-item slab is intentionally double-sided).
    this.solid = this.pipelines.get({ ...base, label: "entity-solid", pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.None }, constants: { ALPHA_TEST: 0 } });
    this.cutout = this.pipelines.get({ ...base, label: "entity-cutout", pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.None }, constants: { ALPHA_TEST: 1 } });
    this.translucent = this.pipelines.get({ ...base, label: "entity-translucent", pass: { depthTest: true, depthWrite: false, blend: true, cull: CullMode.None }, constants: { ALPHA_TEST: 0, EMIT_ALPHA: 1 } });
    // Back-face-culled variants: a SOLID block item (terrain-wound, closed cube) must cull its back faces
    // or they z-fight through the silhouette and flicker a bright sliver as it spins. Only used when a
    // draw sets `cull` (block items in the world); the model's determinant must be positive (no Y-flip).
    this.solidCull = this.pipelines.get({ ...base, label: "entity-solid-cull", pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.Back }, constants: { ALPHA_TEST: 0 } });
    this.cutoutCull = this.pipelines.get({ ...base, label: "entity-cutout-cull", pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.Back }, constants: { ALPHA_TEST: 1 } });
    this.translucentCull = this.pipelines.get({ ...base, label: "entity-translucent-cull", pass: { depthTest: true, depthWrite: false, blend: true, cull: CullMode.Back }, constants: { ALPHA_TEST: 0, EMIT_ALPHA: 1 } });
    this.sampler = this.device.createSampler({ mag: FilterMode.Nearest, min: FilterMode.Nearest });
  }

  private ensureUniform(slots: number): void {
    if (this.uniform && slots <= this.uniformCapacitySlots) return;
    const cap = Math.max(slots, 32, this.uniformCapacitySlots * 2);
    if (this.uniform) this.device.destroyBuffer(this.uniform);
    this.uniform = this.device.createBuffer({ sizeBytes: cap * SLOT_BYTES, usage: BufferUsage.Uniform, label: "entity-uniform" });
    this.uniformCapacitySlots = cap;
    this.slotData = new Float32Array(cap * SLOT_FLOATS);
    this.bindingsByTexture.clear(); // bind groups reference the old buffer — rebuild lazily
  }

  private ensureIndex(maxQuads: number): void {
    if (this.indexBuf && maxQuads <= this.indexCapacityQuads) return;
    if (this.indexBuf) this.device.destroyBuffer(this.indexBuf);
    const idx = quadIndices(maxQuads);
    this.indexBuf = this.device.createBuffer({ sizeBytes: idx.byteLength, usage: BufferUsage.Index, label: "entity-quad-index" });
    this.device.writeBuffer(this.indexBuf, 0, idx);
    this.indexCapacityQuads = maxQuads;
  }

  private bindingsFor(texture: GpuTextureHandle): BindingsHandle {
    let b = this.bindingsByTexture.get(texture);
    if (!b) {
      b = this.device.createBindings({
        pipeline: this.solid!,
        group: 0,
        entries: [
          { binding: 0, resource: { buffer: this.uniform!, offset: 0, size: UNIFORM_BYTES } },
          { binding: 1, resource: { texture } },
          { binding: 2, resource: { sampler: this.sampler! } },
        ],
      });
      this.bindingsByTexture.set(texture, b);
    }
    return b;
  }

  /**
   * Draw this frame's entities. Call AFTER `TerrainRenderer.render` in the same frame: this opens a
   * pass that LOADS the existing color + depth (no clear), so entities depth-interact with terrain.
   */
  render(draws: readonly EntityDraw[], viewProj: Mat4, camPos: Vec3, width: number, height: number, clear: ClearDesc = {}, view?: Mat4, proj?: Mat4): void {
    this.ensureResources();
    const target = { id: null, width, height } as const;
    // PREC-1: decide the UPLOADED view-projection + the camera-origin offset folded into model translations.
    // OFF (or no view/proj supplied) ⇒ the WORLD viewProj + a zero offset ⇒ models upload unchanged (byte-
    // identical). ON ⇒ a translation-free viewProjRel and camOrigin = the camera block; each model's translation
    // column is shifted by −camOrigin below so `model · local` lands near the camera (no jitter at large coords).
    const rel = this.cameraRelative && !!view && !!proj; // the vp choice and the model rewrite share ONE gate
    let vpUpload = viewProj;
    let ox = 0, oy = 0, oz = 0;
    if (rel) {
      ox = Math.round(camPos[0]); oy = Math.round(camPos[1]); oz = Math.round(camPos[2]);
      const vr = this.viewRelScratch;
      vr.set(view!);
      vr[12] = 0; vr[13] = 0; vr[14] = 0; // zero the view translation (TRAP PREC-1.B — never touch the shared world vp)
      multiplyInto(this.viewProjRelScratch, proj!, vr);
      vpUpload = this.viewProjRelScratch;
    }
    // F8: partition + sort into REUSED scratch (no per-frame filter/spread arrays). Opaque (solid → cutout)
    // first; translucent back-to-front by distance. Skips quadCount===0 draws inline.
    const opaque = this.opaqueScratch;
    const translucent = this.translucentScratch;
    opaque.length = 0;
    translucent.length = 0;
    for (const d of draws) {
      if (d.quadCount === 0) continue;
      (d.pass === TerrainPass.Translucent ? translucent : opaque).push(d);
    }
    opaque.sort((a, b) => a.pass - b.pass);
    // sortPos can be undefined (entity with no position yet) → treat as distance 0, preserving the old guard.
    translucent.sort((a, b) => (b.sortPos ? dist2(b.sortPos, camPos) : 0) - (a.sortPos ? dist2(a.sortPos, camPos) : 0));
    const items = this.itemsScratch;
    items.length = 0;
    for (const d of opaque) items.push(d);
    for (const d of translucent) items.push(d);
    this.stats = { drawn: items.length };
    if (items.length === 0) return; // nothing to add to the already-presented terrain frame

    const maxQuads = items.reduce((m, d) => Math.max(m, d.quadCount), 0);
    this.ensureIndex(maxQuads);
    this.ensureUniform(items.length);

    const slots = this.slotData;
    const mr = this.modelRelScratch;
    for (let i = 0; i < items.length; i++) {
      const o = i * SLOT_FLOATS;
      slots.set(vpUpload, o);
      // PREC-1: pre-subtract the camera origin from this model's translation column (col 3 = indices 12..14)
      // so the world-placed model lands near the camera. OFF ⇒ the model uploads UNCHANGED (byte-identical).
      if (rel && items[i].modelRelative) {
        // M6: the model was already built camera-relative in DOUBLE (Mat4Frame.Tw subtracted the anchor before
        // the f32 store), so its translation is small + f32-exact — upload it AS-IS, no second subtraction.
        // This is the precision-correct path (matches terrain); it covers the box-model + e.position entities.
        slots.set(items[i].model, o + 16);
      } else if (rel) {
        const m = items[i].model;
        mr.set(m);
        // LEGACY camera-relative path (M6) for models still built ABSOLUTE upstream (BE-static / sortOf-derived
        // display entities): `m[12]` is `fround(worldX)`, which at extreme coords (step ~2 at 30M) already lost
        // sub-grid precision BEFORE this f32 subtraction — a residual placement error (sub-decimeter at ~1M).
        // Those sites would need the same `Tw` conversion to be precision-exact; bounded to idle/embedded
        // entities and cameraRelative is default-off + far-from-origin only.
        mr[12] = m[12] - ox; mr[13] = m[13] - oy; mr[14] = m[14] - oz;
        slots.set(mr, o + 16);
      } else {
        slots.set(items[i].model, o + 16);
      }
      slots.set(items[i].tint ?? WHITE, o + 32);
      slots.set(items[i].flash ?? NO_FLASH, o + 36);
      slots.set(items[i].light0 ?? NO_LIGHT, o + 40);
      slots.set(items[i].light1 ?? NO_LIGHT, o + 44);
    }
    this.device.writeBuffer(this.uniform!, 0, slots.subarray(0, items.length * SLOT_FLOATS));

    const pipelineFor = (d: EntityDraw): PipelineHandle => {
      if (d.cull) return d.pass === TerrainPass.Solid ? this.solidCull! : d.pass === TerrainPass.Cutout ? this.cutoutCull! : this.translucentCull!;
      return d.pass === TerrainPass.Solid ? this.solid! : d.pass === TerrainPass.Cutout ? this.cutout! : this.translucent!;
    };

    // Default = LOAD (keep terrain's color + depth). The GUI item pass passes `{ depth: 1 }` to draw
    // over the world with a fresh depth (an ortho overlay). setPipeline/setIndexBuffer are GUARDED
    // (AUDIT F7/ENT-3): all entity draws share `this.indexBuf` (set once) and pipelines repeat heavily.
    const pass = this.device.beginPass(target, clear);
    let lastPipeline: PipelineHandle | null = null;
    let indexSet = false;
    for (let i = 0; i < items.length; i++) {
      const d = items[i];
      const pipeline = pipelineFor(d);
      if (pipeline !== lastPipeline) {
        pass.setPipeline(pipeline);
        lastPipeline = pipeline;
      }
      this.dynOff[0] = i * SLOT_BYTES;
      pass.setBindings(this.bindingsFor(d.texture), this.dynOff);
      pass.setVertexBuffer(0, d.vertex);
      if (!indexSet) {
        pass.setIndexBuffer(this.indexBuf!, IndexFormat.Uint32);
        indexSet = true;
      }
      pass.drawIndexed(d.quadCount * 6, 0, 0);
    }
    pass.end();
  }

  dispose(): void {
    if (this.uniform) this.device.destroyBuffer(this.uniform);
    if (this.indexBuf) this.device.destroyBuffer(this.indexBuf);
    this.pipelines.dispose();
    this.solid = this.cutout = this.translucent = null;
    this.solidCull = this.cutoutCull = this.translucentCull = null;
    this.uniform = this.indexBuf = null;
    this.bindingsByTexture.clear();
  }
}

