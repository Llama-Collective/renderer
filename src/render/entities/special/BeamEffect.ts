// Beacon beam. RENDERER_PLAN.md §18, Phase 4.5f. Vanilla BeaconRenderer draws the beam as TWO render
// types (RenderPipelines.java): an OPAQUE inner core (`BEACON_BEAM_OPAQUE`, depth-write on, full colour)
// and a wider TRANSLUCENT glow (`BEACON_BEAM_TRANSLUCENT`, depth-write off, α=32/255). NOT additive. We
// reproduce both with one shader + two pipelines; the inner core spins (vanilla 2.25°/tick ≈ 45°/s) and
// the texture-V scroll is emulated as energy bands moving up the beam's world-Y (we're untextured here).

import type { BindingsHandle, GraphicsDevice, PassEncoder, PipelineHandle } from "../../../core/GraphicsDevice";
import { CompareFn, CullMode, ShaderStage, VertexScalarKind } from "../../../core/GraphicsDevice";
import type { Vec3 } from "../../../types";
import type { BESpecialDraw } from "../blockentities";
import { packRgba, QuadBatch, UNIFORM_BYTES, type SpecialContext, type SpecialEffect } from "./shared";
import { unpackRgb } from "../../color";

const STRIDE = 20; // pos vec3 (12) + rgba unorm8x4 (4) + scrollV f32 (4)
const FLOATS = 8; // x,y,z, r,g,b,a, v
const SPIN_RAD_PER_S = (45 * Math.PI) / 180; // vanilla 2.25°/tick × 20 ticks/s

const WGSL = /* wgsl */ `
struct U { viewProj : mat4x4<f32>, time : f32, resX : f32, resY : f32 };
@group(0) @binding(0) var<uniform> u : U;
struct VsIn { @location(0) pos : vec3<f32>, @location(1) color : vec4<f32>, @location(2) v : f32 };
struct VsOut { @builtin(position) clip : vec4<f32>, @location(0) color : vec4<f32>, @location(1) v : f32 };
@vertex fn vs(in : VsIn) -> VsOut {
  var o : VsOut;
  o.clip = u.viewProj * vec4<f32>(in.pos, 1.0);
  o.color = in.color;
  o.v = in.v;
  return o;
}
@fragment fn fs(in : VsOut) -> @location(0) vec4<f32> {
  // Energy bands flowing UP the beam (emulates vanilla's scrolling beacon_beam texture-V).
  let band = 0.5 + 0.5 * sin((in.v - u.time * 0.75) * 3.1415927);
  return vec4<f32>(in.color.rgb * (0.55 + 0.6 * band), in.color.a * (0.45 + 0.55 * band));
}
`;

type Col = [number, number, number, number];

export class BeamEffect implements SpecialEffect {
  private corePipe: PipelineHandle | null = null;
  private glowPipe: PipelineHandle | null = null;
  private coreBind: BindingsHandle | null = null;
  private glowBind: BindingsHandle | null = null;
  private readonly core: QuadBatch;
  private readonly glow: QuadBatch;

  constructor(private readonly ctx: SpecialContext) {
    this.core = new QuadBatch(ctx.device, "beam-core");
    this.glow = new QuadBatch(ctx.device, "beam-glow");
  }

  build(specials: readonly BESpecialDraw[], clock: number): number {
    const coreF: number[] = [];
    const glowF: number[] = [];
    for (const s of specials) if (s.kind === "beam") beamQuads(coreF, glowF, s.pos, s.color, s.height, clock);
    const coreQuads = coreF.length / (4 * FLOATS);
    const glowQuads = glowF.length / (4 * FLOATS);
    this.core.upload(pack(coreF), coreQuads);
    this.glow.upload(pack(glowF), glowQuads);
    return coreQuads + glowQuads;
  }

  encode(pass: PassEncoder): void {
    this.ensure();
    if (this.core.quadCount > 0) {
      pass.setPipeline(this.corePipe!);
      pass.setBindings(this.coreBind!);
      this.core.draw(pass);
    }
    if (this.glow.quadCount > 0) {
      pass.setPipeline(this.glowPipe!);
      pass.setBindings(this.glowBind!);
      this.glow.draw(pass);
    }
  }

  private ensure(): void {
    if (this.corePipe) return;
    const shader = this.ctx.device.createShaderModule(WGSL, "beam");
    const common = {
      label: "beam",
      shader,
      vertexEntry: "vs",
      fragmentEntry: "fs",
      vertexLayout: {
        strideBytes: STRIDE,
        attributes: [
          { location: 0, kind: VertexScalarKind.Float32, components: 3, offsetBytes: 0, asInt: false },
          { location: 1, kind: VertexScalarKind.Unorm8, components: 4, offsetBytes: 12, asInt: false },
          { location: 2, kind: VertexScalarKind.Float32, components: 1, offsetBytes: 16, asInt: false },
        ],
      },
      colorFormat: this.ctx.colorFormat,
      depthFormat: this.ctx.depthFormat,
      depthCompare: CompareFn.LessEqual, // vanilla render types use LEQUAL (RenderPipelines default)
      bindingLayout: [{ binding: 0, visibility: [ShaderStage.Vertex, ShaderStage.Fragment], type: { kind: "uniform-buffer" as const } }],
    };
    // Opaque core (vanilla BEACON_BEAM_OPAQUE: depth-write on, no blend) + translucent glow.
    this.corePipe = this.ctx.pipelines.get({ ...common, label: "beam-core", pass: { depthTest: true, depthWrite: true, blend: false, cull: CullMode.None } });
    this.glowPipe = this.ctx.pipelines.get({ ...common, label: "beam-glow", pass: { depthTest: true, depthWrite: false, blend: true, cull: CullMode.None } });
    this.coreBind = bind(this.ctx, this.corePipe);
    this.glowBind = bind(this.ctx, this.glowPipe);
  }

  dispose(): void {
    this.core.dispose();
    this.glow.dispose();
    this.corePipe = this.glowPipe = null;
    this.coreBind = this.glowBind = null;
  }
}

function bind(ctx: SpecialContext, pipeline: PipelineHandle): BindingsHandle {
  return ctx.device.createBindings({ pipeline, group: 0, entries: [{ binding: 0, resource: { buffer: ctx.uniform, offset: 0, size: UNIFORM_BYTES } }] });
}

/** Per-vertex float count of the packed beam layout (x,y,z, r,g,b,a, scrollV). Exported for tests. */
export const BEAM_FLOATS_PER_VERTEX = FLOATS;

/** Beacon beam geometry: inner solid tube (spins, → core) + wider glow tube (static, faint → glow). */
export function beamQuads(core: number[], glow: number[], pos: Vec3, color: number, height: number, clock: number): void {
  const cx = pos[0] + 0.5, cz = pos[2] + 0.5, y0 = pos[1], y1 = pos[1] + height;
  const rgb = unpackRgb(color);
  const a = clock * SPIN_RAD_PER_S;
  const ca = Math.cos(a), sa = Math.sin(a);
  const rot = ([x, z]: [number, number]): [number, number] => [x * ca - z * sa, x * sa + z * ca];
  // Inner core (radius 0.2, opaque): a diamond cross-section spun about Y.
  const inner: [number, number][] = ([[0, 0.2], [0.2, 0], [0, -0.2], [-0.2, 0]] as [number, number][]).map(rot);
  tube(core, inner, cx, cz, y0, y1, [...rgb, 1]);
  // Glow (radius 0.25, α=32/255 = vanilla ARGB.color(32, color)): a wider static square.
  const g = 0.25;
  tube(glow, [[-g, -g], [g, -g], [g, g], [-g, g]], cx, cz, y0, y1, [...rgb, 0.125]);
}

/** 4 side quads of a vertical tube whose XZ cross-section is `corners` (relative to cx,cz). */
function tube(out: number[], corners: [number, number][], cx: number, cz: number, y0: number, y1: number, c: Col): void {
  for (let i = 0; i < 4; i++) {
    const a = corners[i], b = corners[(i + 1) % 4];
    push(out, cx + a[0], y0, cz + a[1], c);
    push(out, cx + b[0], y0, cz + b[1], c);
    push(out, cx + b[0], y1, cz + b[1], c);
    push(out, cx + a[0], y1, cz + a[1], c);
  }
}
function push(out: number[], x: number, y: number, z: number, c: Col): void {
  out.push(x, y, z, c[0], c[1], c[2], c[3], y); // v = world Y (scroll coordinate)
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
    packRgba(dv, o + 12, flat[f + 3], flat[f + 4], flat[f + 5], flat[f + 6]);
    dv.setFloat32(o + 16, flat[f + 7], true);
  }
  return new Uint8Array(buf);
}
