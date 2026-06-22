// End portal / gateway. RENDERER_PLAN.md §18, Phase 4.5f. Faithful port of vanilla
// `rendertype_end_portal.{vsh,fsh}` (RenderPipelines.END_PORTAL): an OPAQUE (no-blend) quad cube whose
// faces show a screen-projective starfield — the real `environment/end_sky` (Sampler0, base) plus 15
// receding, rotating, scrolling `entity/end_portal` layers (Sampler1), each tinted by the exact COLORS
// palette. Repeat-wrapped textures; GameTime drive matched to vanilla (1200-second cycle).

import type { BindingsHandle, GpuTextureHandle, PassEncoder, PipelineHandle, SamplerHandle } from "../../../core/GraphicsDevice";
import { AddressMode, CompareFn, CullMode, FilterMode, SamplerBindingType, ShaderStage, TextureFormat, TextureSampleType, VertexScalarKind } from "../../../core/GraphicsDevice";
import type { Vec3 } from "../../../types";
import type { BESpecialDraw } from "../blockentities";
import { QuadBatch, UNIFORM_BYTES, type SpecialContext, type SpecialEffect } from "./shared";

/** A decoded RGBA texture (tightly packed, length = width·height·4). */
export interface RawTex {
  width: number;
  height: number;
  rgba: Uint8Array;
}
/** The end-portal shader's two source textures (vanilla Sampler0 = end_sky, Sampler1 = end_portal). */
export interface PortalTextures {
  endSky?: RawTex;
  endPortal?: RawTex;
}

const STRIDE = 16; // pos vec3 (12) + layerCount f32 (4) — the starfield is computed from on-screen position
const FLOATS = 4; // x,y,z, layerCount

const WGSL = /* wgsl */ `
struct U { viewProj : mat4x4<f32>, time : f32, resX : f32, resY : f32 };
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var samp : sampler;
@group(0) @binding(2) var skyTex : texture_2d<f32>;     // Sampler0 = environment/end_sky (base)
@group(0) @binding(3) var portalTex : texture_2d<f32>;  // Sampler1 = entity/end_portal/end_portal (layers)
struct VsOut { @builtin(position) clip : vec4<f32>, @location(0) @interpolate(flat) layers : f32 };
@vertex fn vs(@location(0) pos : vec3<f32>, @location(1) layers : f32) -> VsOut {
  var o : VsOut;
  o.clip = u.viewProj * vec4<f32>(pos, 1.0);
  o.layers = layers; // per-block layer count: 15 (end portal) / 16 (end gateway), vanilla #define RENDERTYPE_END_PORTAL_LAYERS
  return o;
}

// Vanilla GameTime is (gameTime % 24000)/24000 ∈ [0,1) — a 1200-second cycle.
const GAME_TIME_PER_SEC : f32 = 1.0 / 1200.0;

// Vanilla end_portal_layer(layer) (rendertype_end_portal.fsh + matrix.glsl), built column-major like GLSL.
fn portalLayerMat(layer : f32, t : f32) -> mat4x4<f32> {
  let scaleTranslate = mat4x4<f32>(
    vec4<f32>(0.5, 0.0, 0.0, 0.25),
    vec4<f32>(0.0, 0.5, 0.0, 0.25),
    vec4<f32>(0.0, 0.0, 1.0, 0.0),
    vec4<f32>(0.0, 0.0, 0.0, 1.0),
  );
  let translate = mat4x4<f32>(
    vec4<f32>(1.0, 0.0, 0.0, 17.0 / layer),
    vec4<f32>(0.0, 1.0, 0.0, (2.0 + layer / 1.5) * (t * 1.5)),
    vec4<f32>(0.0, 0.0, 1.0, 0.0),
    vec4<f32>(0.0, 0.0, 0.0, 1.0),
  );
  let a = radians((layer * layer * 4321.0 + layer * 9.0) * 2.0);
  let cs = cos(a); let sn = sin(a);
  let sc = (4.5 - layer / 4.0) * 2.0; // mat4(scale·mat2_rotate_z): cols (sc·cos,-sc·sin),(sc·sin,sc·cos)
  let scaleRot = mat4x4<f32>(
    vec4<f32>(sc * cs, -sc * sn, 0.0, 0.0),
    vec4<f32>(sc * sn, sc * cs, 0.0, 0.0),
    vec4<f32>(0.0, 0.0, 1.0, 0.0),
    vec4<f32>(0.0, 0.0, 0.0, 1.0),
  );
  return scaleRot * translate * scaleTranslate;
}

@fragment fn fs(in : VsOut) -> @location(0) vec4<f32> {
  // texProj0 = screen-projective coord; the common clip.w factors out of the perspective divide, so
  // q = (screenU, screenV, 0, 1) reproduces it exactly. (rendertype_end_portal.fsh)
  var COLORS = array<vec3<f32>, 16>(
    vec3<f32>(0.022087, 0.098399, 0.110818), vec3<f32>(0.011892, 0.095924, 0.089485),
    vec3<f32>(0.027636, 0.101689, 0.100326), vec3<f32>(0.046564, 0.109883, 0.114838),
    vec3<f32>(0.064901, 0.117696, 0.097189), vec3<f32>(0.063761, 0.086895, 0.123646),
    vec3<f32>(0.084817, 0.111994, 0.166380), vec3<f32>(0.097489, 0.154120, 0.091064),
    vec3<f32>(0.106152, 0.131144, 0.195191), vec3<f32>(0.097721, 0.110188, 0.187229),
    vec3<f32>(0.133516, 0.138278, 0.148582), vec3<f32>(0.070006, 0.243332, 0.235792),
    vec3<f32>(0.196766, 0.142899, 0.214696), vec3<f32>(0.047281, 0.315338, 0.321970),
    vec3<f32>(0.204675, 0.390010, 0.302066), vec3<f32>(0.080955, 0.314821, 0.661491),
  );
  let t = u.time * GAME_TIME_PER_SEC;
  var screen = in.clip.xy / vec2<f32>(max(u.resX, 1.0), max(u.resY, 1.0));
  screen.y = 1.0 - screen.y; // WebGPU fragCoord is Y-DOWN; vanilla's screen-projective coord is Y-up
  let q = vec4<f32>(screen, 0.0, 1.0);
  var col = textureSampleLevel(skyTex, samp, screen, 0.0).rgb * COLORS[0];
  let layers = i32(in.layers); // 15 (end portal) / 16 (end gateway)
  for (var i : i32 = 0; i < layers; i = i + 1) {
    let p = transpose(portalLayerMat(f32(i + 1), t)) * q; // GLSL row-vector: q * end_portal_layer(i+1)
    let uvL = p.xy / p.w;
    col = col + textureSampleLevel(portalTex, samp, uvL, 0.0).rgb * COLORS[i];
  }
  return vec4<f32>(col, 1.0);
}
`;

export class PortalEffect implements SpecialEffect {
  private pipeline: PipelineHandle | null = null;
  private bindings: BindingsHandle | null = null;
  private sampler: SamplerHandle | null = null;
  private skyTex: GpuTextureHandle | null = null;
  private portalTex: GpuTextureHandle | null = null;
  private pending: PortalTextures | null = null;
  private readonly batch: QuadBatch;

  constructor(private readonly ctx: SpecialContext) {
    this.batch = new QuadBatch(ctx.device, "portal");
  }

  /** Supply the shader's textures (Sampler0 = end_sky, Sampler1 = end_portal). */
  setTextures(tex: PortalTextures): void {
    this.pending = tex;
    if (this.pipeline) this.uploadAndBind(); // already initialized → hot-swap
  }

  build(specials: readonly BESpecialDraw[], _clock: number): number {
    const f: number[] = [];
    for (const s of specials) if (s.kind === "portal") portalCube(f, s.pos, s.gateway ?? false);
    const quads = f.length / (4 * FLOATS);
    this.batch.upload(pack(f), quads);
    return quads;
  }

  encode(pass: PassEncoder): void {
    this.ensure();
    pass.setPipeline(this.pipeline!);
    pass.setBindings(this.bindings!);
    this.batch.draw(pass);
  }

  private ensure(): void {
    if (this.pipeline) return;
    const shader = this.ctx.device.createShaderModule(WGSL, "portal");
    this.pipeline = this.ctx.pipelines.get({
      label: "portal",
      shader,
      vertexEntry: "vs",
      fragmentEntry: "fs",
      vertexLayout: {
        strideBytes: STRIDE,
        attributes: [
          { location: 0, kind: VertexScalarKind.Float32, components: 3, offsetBytes: 0, asInt: false },
          { location: 1, kind: VertexScalarKind.Float32, components: 1, offsetBytes: 12, asInt: false }, // layerCount
        ],
      },
      colorFormat: this.ctx.colorFormat,
      depthFormat: this.ctx.depthFormat,
      depthCompare: CompareFn.LessEqual, // vanilla render types use LEQUAL (RenderPipelines default)
      // Vanilla END_PORTAL: opaque (no blend), depth DEFAULT (write on). Cull none (the cube self-occludes
      // via depth-write, so winding doesn't matter).
      pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.None },
      bindingLayout: [
        { binding: 0, visibility: [ShaderStage.Vertex, ShaderStage.Fragment], type: { kind: "uniform-buffer" } },
        { binding: 1, visibility: [ShaderStage.Fragment], type: { kind: "sampler", sampler: SamplerBindingType.Filtering } },
        { binding: 2, visibility: [ShaderStage.Fragment], type: { kind: "texture", sampleType: TextureSampleType.Float } },
        { binding: 3, visibility: [ShaderStage.Fragment], type: { kind: "texture", sampleType: TextureSampleType.Float } },
      ],
    });
    this.sampler = this.ctx.device.createSampler({ mag: FilterMode.Linear, min: FilterMode.Linear, address: AddressMode.Repeat, label: "portal-sampler" });
    this.uploadAndBind();
  }

  private uploadAndBind(): void {
    if (this.skyTex) this.ctx.device.destroyTexture(this.skyTex);
    if (this.portalTex) this.ctx.device.destroyTexture(this.portalTex);
    this.skyTex = this.uploadTex(this.pending?.endSky, "end-sky");
    this.portalTex = this.uploadTex(this.pending?.endPortal, "end-portal");
    if (!this.pipeline || !this.sampler) return;
    this.bindings = this.ctx.device.createBindings({
      pipeline: this.pipeline,
      group: 0,
      entries: [
        { binding: 0, resource: { buffer: this.ctx.uniform, offset: 0, size: UNIFORM_BYTES } },
        { binding: 1, resource: { sampler: this.sampler } },
        { binding: 2, resource: { texture: this.skyTex } },
        { binding: 3, resource: { texture: this.portalTex } },
      ],
    });
  }

  /** Upload an RGBA texture (or a 1×1 dim fallback when the pack lacks it). */
  private uploadTex(raw: RawTex | undefined, label: string): GpuTextureHandle {
    const w = raw?.width ?? 1, h = raw?.height ?? 1;
    const data = raw?.rgba ?? new Uint8Array([8, 10, 12, 255]);
    const tex = this.ctx.device.createTexture({ width: w, height: h, format: TextureFormat.Rgba8, label });
    this.ctx.device.writeTexture(tex, data);
    return tex;
  }

  dispose(): void {
    this.batch.dispose();
    if (this.skyTex) this.ctx.device.destroyTexture(this.skyTex);
    if (this.portalTex) this.ctx.device.destroyTexture(this.portalTex);
    this.skyTex = this.portalTex = null;
    this.pipeline = null;
    this.bindings = null;
  }
}

/**
 * Vanilla TheEndPortalRenderer cube (BOTTOM 0.375 → TOP 0.75; a gateway fills the whole block). Faces
 * follow vanilla `shouldRenderFace`: a non-gateway end-portal block renders **top + bottom only**; an end
 * gateway renders **all 6** faces. Each vertex carries the per-block layer count (15 portal / 16 gateway).
 * Exported for unit tests (FLOATS-stride layout: x,y,z,layerCount per vertex; 4 verts per quad).
 */
export function portalCube(out: number[], pos: Vec3, gateway: boolean): void {
  const layers = gateway ? 16 : 15;
  const x0 = pos[0], x1 = pos[0] + 1, z0 = pos[2], z1 = pos[2] + 1;
  const y0 = pos[1] + (gateway ? 0 : 0.375), y1 = pos[1] + (gateway ? 1 : 0.75);
  quad(out, layers, [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]); // top (UP)
  quad(out, layers, [x0, y0, z1], [x1, y0, z1], [x1, y0, z0], [x0, y0, z0]); // bottom (DOWN)
  if (gateway) {
    quad(out, layers, [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]); // north
    quad(out, layers, [x1, y0, z1], [x0, y0, z1], [x0, y1, z1], [x1, y1, z1]); // south
    quad(out, layers, [x0, y0, z1], [x0, y0, z0], [x0, y1, z0], [x0, y1, z1]); // west
    quad(out, layers, [x1, y0, z0], [x1, y0, z1], [x1, y1, z1], [x1, y1, z0]); // east
  }
}
/** Export the per-vertex float count so tests can index the packed quad layout. */
export const PORTAL_FLOATS_PER_VERTEX = FLOATS;
function quad(out: number[], layers: number, a: Vec3, b: Vec3, c: Vec3, d: Vec3): void {
  for (const p of [a, b, c, d]) out.push(p[0], p[1], p[2], layers);
}

function pack(flat: number[]): Uint8Array {
  const verts = flat.length / FLOATS;
  const buf = new ArrayBuffer(verts * STRIDE);
  const dv = new DataView(buf);
  for (let v = 0; v < verts; v++) {
    const o = v * STRIDE, f = v * FLOATS;
    dv.setFloat32(o, flat[f], true);
    dv.setFloat32(o + 4, flat[f + 1], true);
    dv.setFloat32(o + 8, flat[f + 2], true);
    dv.setFloat32(o + 12, flat[f + 3], true); // layerCount
  }
  return new Uint8Array(buf);
}
