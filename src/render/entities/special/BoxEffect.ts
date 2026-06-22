// Wireframe bounding box (structure / jigsaw / test blocks + entity hitboxes). RENDERER_PLAN.md §18, 4.5f.
// Vanilla draws structure-block boxes with RenderPipelines.LINES (TRANSLUCENT blend, real `line-list`
// topology, depth-tested). We draw the 12 edges as REAL GPU lines (24 verts, drawn as pairs via a
// sequential index buffer + line-list pipeline) — crisp 1px edges that don't fatten with perspective.
//
// Two depth modes (a lineBox sets `xray`):
//   - SOLID (default): depth-tested (LEQUAL, writes depth) — BE wireframes occlude behind geometry, vanilla.
//   - XRAY: depthCompare ALWAYS + no depth write — entity bounding boxes render THROUGH blocks/entities so
//     they're always visible (matches the old renderer's depthTest:false hitbox outlines). Drawn last.

import type { BindingsHandle, GpuBufferHandle, PassEncoder, PipelineHandle } from "../../../core/GraphicsDevice";
import { BufferUsage, CompareFn, CullMode, IndexFormat, PrimitiveTopology, ShaderStage, VertexScalarKind } from "../../../core/GraphicsDevice";
import type { Vec3 } from "../../../types";
import type { BESpecialDraw } from "../blockentities";
import { packRgba, UNIFORM_BYTES, type SpecialContext, type SpecialEffect } from "./shared";
import { unpackRgb } from "../../color";

const STRIDE = 16; // pos vec3 (12) + rgba unorm8x4 (4)
const FLOATS = 7; // x,y,z, r,g,b,a

const WGSL = /* wgsl */ `
struct U { viewProj : mat4x4<f32>, time : f32, resX : f32, resY : f32 };
@group(0) @binding(0) var<uniform> u : U;
struct VsIn { @location(0) pos : vec3<f32>, @location(1) color : vec4<f32> };
struct VsOut { @builtin(position) clip : vec4<f32>, @location(0) color : vec4<f32> };
@vertex fn vs(in : VsIn) -> VsOut {
  var o : VsOut;
  o.clip = u.viewProj * vec4<f32>(in.pos, 1.0);
  o.color = in.color;
  return o;
}
@fragment fn fs(in : VsOut) -> @location(0) vec4<f32> { return in.color; }
`;

type Col = [number, number, number, number];

/** One depth mode's GPU resources (line-list verts + sequential index buffer + pipeline + bindings). */
class BoxBatch {
  pipeline: PipelineHandle | null = null;
  bindings: BindingsHandle | null = null;
  vbuf: GpuBufferHandle | null = null;
  vcap = 0;
  ibuf: GpuBufferHandle | null = null;
  icap = 0;
  verts = 0;

  constructor(private readonly ctx: SpecialContext, private readonly xray: boolean) {}

  /** Upload this mode's flat vertex floats; returns the line count (for the stat). */
  upload(f: number[]): number {
    this.verts = f.length / FLOATS;
    if (this.verts === 0) return 0;
    const data = pack(f);
    if (!this.vbuf || data.byteLength > this.vcap) {
      if (this.vbuf) this.ctx.device.destroyBuffer(this.vbuf);
      this.vcap = Math.max(data.byteLength, 4096);
      this.vbuf = this.ctx.device.createBuffer({ sizeBytes: this.vcap, usage: BufferUsage.Vertex, label: "linebox-verts" });
    }
    this.ctx.device.writeBuffer(this.vbuf, 0, data);
    // line-list draws index pairs as lines → a plain sequential index buffer [0,1,2,…].
    if (!this.ibuf || this.verts > this.icap) {
      if (this.ibuf) this.ctx.device.destroyBuffer(this.ibuf);
      this.icap = this.verts;
      const idx = new Uint32Array(this.verts);
      for (let i = 0; i < this.verts; i++) idx[i] = i;
      this.ibuf = this.ctx.device.createBuffer({ sizeBytes: idx.byteLength, usage: BufferUsage.Index, label: "linebox-index" });
      this.ctx.device.writeBuffer(this.ibuf, 0, new Uint8Array(idx.buffer));
    }
    return this.verts / 2;
  }

  encode(pass: PassEncoder): void {
    if (this.verts === 0 || !this.vbuf || !this.ibuf) return;
    this.ensure();
    pass.setPipeline(this.pipeline!);
    pass.setBindings(this.bindings!);
    pass.setVertexBuffer(0, this.vbuf);
    pass.setIndexBuffer(this.ibuf, IndexFormat.Uint32);
    pass.drawIndexed(this.verts, 0, 0);
  }

  private ensure(): void {
    if (this.pipeline) return;
    const shader = this.ctx.device.createShaderModule(WGSL, this.xray ? "linebox-xray" : "linebox");
    this.pipeline = this.ctx.pipelines.get({
      label: this.xray ? "linebox-xray" : "linebox",
      shader,
      vertexEntry: "vs",
      fragmentEntry: "fs",
      vertexLayout: {
        strideBytes: STRIDE,
        attributes: [
          { location: 0, kind: VertexScalarKind.Float32, components: 3, offsetBytes: 0, asInt: false },
          { location: 1, kind: VertexScalarKind.Unorm8, components: 4, offsetBytes: 12, asInt: false },
        ],
      },
      colorFormat: this.ctx.colorFormat,
      depthFormat: this.ctx.depthFormat,
      // SOLID: LEQUAL + depth write (vanilla RenderPipelines.LINES). XRAY: omit depthCompare so the device
      // uses "always" (depthTest:false) and disable depth write → draws through everything, never occluded.
      depthCompare: this.xray ? undefined : CompareFn.LessEqual,
      pass: { depthTest: !this.xray, depthWrite: !this.xray, blend: true, cull: CullMode.None, topology: PrimitiveTopology.LineList },
      bindingLayout: [{ binding: 0, visibility: [ShaderStage.Vertex, ShaderStage.Fragment], type: { kind: "uniform-buffer" } }],
    });
    this.bindings = this.ctx.device.createBindings({ pipeline: this.pipeline, group: 0, entries: [{ binding: 0, resource: { buffer: this.ctx.uniform, offset: 0, size: UNIFORM_BYTES } }] });
  }

  dispose(): void {
    if (this.vbuf) this.ctx.device.destroyBuffer(this.vbuf);
    if (this.ibuf) this.ctx.device.destroyBuffer(this.ibuf);
    this.vbuf = this.ibuf = null;
    this.pipeline = null;
    this.bindings = null;
  }
}

export class BoxEffect implements SpecialEffect {
  private readonly solid: BoxBatch;
  private readonly xray: BoxBatch;

  constructor(ctx: SpecialContext) {
    this.solid = new BoxBatch(ctx, false);
    this.xray = new BoxBatch(ctx, true);
  }

  build(specials: readonly BESpecialDraw[], _clock: number): number {
    const solidF: number[] = [];
    const xrayF: number[] = [];
    for (const s of specials) if (s.kind === "lineBox") boxEdges(s.xray ? xrayF : solidF, s.min, s.max, s.color);
    return this.solid.upload(solidF) + this.xray.upload(xrayF);
  }

  encode(pass: PassEncoder): void {
    this.solid.encode(pass); // depth-tested first…
    this.xray.encode(pass); // …then the always-on-top hitbox outlines
  }

  dispose(): void {
    this.solid.dispose();
    this.xray.dispose();
  }
}

/** Per-vertex float count of the packed line layout (x,y,z, r,g,b,a). Exported for tests. */
export const BOX_FLOATS_PER_VERTEX = FLOATS;

/** 12 box edges → 24 line-list vertices (each edge = 2 endpoints). */
export function boxEdges(out: number[], min: Vec3, max: Vec3, color: number): void {
  const [r, g, b] = unpackRgb(color);
  const col: Col = [r, g, b, 0.9];
  const C: Vec3[] = [
    [min[0], min[1], min[2]], [max[0], min[1], min[2]], [max[0], min[1], max[2]], [min[0], min[1], max[2]],
    [min[0], max[1], min[2]], [max[0], max[1], min[2]], [max[0], max[1], max[2]], [min[0], max[1], max[2]],
  ];
  const E: [number, number][] = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  for (const [i, j] of E) {
    push(out, C[i], col);
    push(out, C[j], col);
  }
}
function push(out: number[], p: Vec3, c: Col): void {
  out.push(p[0], p[1], p[2], c[0], c[1], c[2], c[3]);
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
  }
  return new Uint8Array(buf);
}
