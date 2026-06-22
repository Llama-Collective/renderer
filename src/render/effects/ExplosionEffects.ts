// Explosion / TNT particle effects as a forward pass: a bright expanding flash + drifting smoke/ember
// particles, drawn camera-facing with straight-alpha "over" blend, depth-test ON, depth-write OFF (so
// the world occludes them but they don't write depth). Compact port of the old renderer's
// ExplosionEffects. RENDERER_INTEGRATION_PLAN Step 8.

import type { BindingsHandle, GpuBufferHandle, GraphicsDevice, PipelineHandle, TextureFormat } from "../../core/GraphicsDevice";
import { BufferUsage, CompareFn, CullMode, IndexFormat, PrimitiveTopology, ShaderStage, VertexScalarKind } from "../../core/GraphicsDevice";
import { PipelineCache } from "../../core/PipelineCache";
import type { Mat4 } from "../../camera/Camera";
import type { Vec3 } from "../../types";
import { COLOR_VERTEX_STRIDE, COLOR_VERTEX_WGSL, packColorVerts, sequentialIndexBuffer } from "../colorVertex";

const FLASH_LIFE = 0.45; // seconds
const PART_LIFE = 0.9;
const MAX_BURSTS = 32;

interface Particle {
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  age: number; life: number; size: number;
  r: number; g: number; b: number;
}
interface Burst {
  x: number; y: number; z: number;
  radius: number; age: number;
  particles: Particle[];
}

export class ExplosionEffects {
  private readonly pipelines: PipelineCache;
  private pipeline: PipelineHandle | null = null;
  private bindings: BindingsHandle | null = null;
  private uniform: GpuBufferHandle | null = null;
  private vbuf: GpuBufferHandle | null = null;
  private vcap = 0;
  private ibuf: GpuBufferHandle | null = null;
  private icap = 0;
  private readonly bursts: Burst[] = [];

  constructor(
    private readonly device: GraphicsDevice,
    private readonly colorFormat: TextureFormat,
    private readonly depthFormat: TextureFormat,
  ) {
    this.pipelines = new PipelineCache(device);
  }

  hasActive(): boolean {
    return this.bursts.length > 0;
  }

  spawn(x: number, y: number, z: number, radius: number): void {
    const r = Math.max(0.5, radius);
    const count = Math.min(120, Math.round(r * 18));
    const particles: Particle[] = [];
    for (let i = 0; i < count; i++) {
      const dir = randUnit();
      const speed = (0.6 + Math.random() * 1.4) * r;
      const ember = Math.random() < 0.5;
      particles.push({
        x, y, z,
        vx: dir[0] * speed, vy: dir[1] * speed * 0.8 + r * 0.4, vz: dir[2] * speed,
        age: 0, life: PART_LIFE * (0.5 + Math.random() * 0.8), size: 0.12 + Math.random() * 0.18,
        r: ember ? 1 : 0.25, g: ember ? 0.55 + Math.random() * 0.3 : 0.25, b: ember ? 0.1 : 0.25,
      });
    }
    this.bursts.push({ x, y, z, radius: r, age: 0, particles });
    if (this.bursts.length > MAX_BURSTS) this.bursts.shift();
  }

  /** Advance active effects by `dt` seconds. */
  update(dt: number): void {
    for (const b of this.bursts) {
      b.age += dt;
      for (const p of b.particles) {
        p.age += dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;
        p.vy -= 3.5 * dt; // gravity
        p.vx *= 1 - 1.6 * dt; // drag
        p.vz *= 1 - 1.6 * dt;
      }
      b.particles = b.particles.filter((p) => p.age < p.life);
    }
    for (let i = this.bursts.length - 1; i >= 0; i--) {
      if (this.bursts[i].age >= FLASH_LIFE && this.bursts[i].particles.length === 0) this.bursts.splice(i, 1);
    }
  }

  /** Build + draw this frame's flash + particle billboards (camera-facing). */
  render(viewProj: Mat4, camRight: Vec3, camUp: Vec3, width: number, height: number): void {
    if (this.bursts.length === 0) return;
    const f: number[] = [];
    const quad = (cx: number, cy: number, cz: number, s: number, r: number, g: number, b: number, a: number): void => {
      const rx = camRight[0] * s * 0.5, ry = camRight[1] * s * 0.5, rz = camRight[2] * s * 0.5;
      const ux = camUp[0] * s * 0.5, uy = camUp[1] * s * 0.5, uz = camUp[2] * s * 0.5;
      const c0: Vec3 = [cx - rx - ux, cy - ry - uy, cz - rz - uz];
      const c1: Vec3 = [cx + rx - ux, cy + ry - uy, cz + rz - uz];
      const c2: Vec3 = [cx + rx + ux, cy + ry + uy, cz + rz + uz];
      const c3: Vec3 = [cx - rx + ux, cy - ry + uy, cz - rz + uz];
      pushTri(f, c0, c1, c2, r, g, b, a);
      pushTri(f, c0, c2, c3, r, g, b, a);
    };
    for (const burst of this.bursts) {
      if (burst.age < FLASH_LIFE) {
        const t = burst.age / FLASH_LIFE;
        const size = burst.radius * (0.8 + t * 1.6);
        const a = (1 - t) * 0.8;
        quad(burst.x, burst.y, burst.z, size, 1, 0.85 - t * 0.4, 0.5 - t * 0.4, a); // orange→dim flash
      }
      for (const p of burst.particles) {
        const a = (1 - p.age / p.life) * 0.85;
        quad(p.x, p.y, p.z, p.size, p.r, p.g, p.b, a);
      }
    }
    const verts = f.length / 7;
    if (verts === 0) return;
    this.ensure();
    const data = packColorVerts(f);
    if (!this.vbuf || data.byteLength > this.vcap) {
      if (this.vbuf) this.device.destroyBuffer(this.vbuf);
      this.vcap = Math.max(data.byteLength, 8192);
      this.vbuf = this.device.createBuffer({ sizeBytes: this.vcap, usage: BufferUsage.Vertex, label: "explosion-verts" });
    }
    this.device.writeBuffer(this.vbuf, 0, data);
    if (verts > this.icap) {
      if (this.ibuf) this.device.destroyBuffer(this.ibuf);
      this.icap = verts;
      const idx = sequentialIndexBuffer(verts);
      this.ibuf = this.device.createBuffer({ sizeBytes: idx.byteLength, usage: BufferUsage.Index, label: "explosion-index" });
      this.device.writeBuffer(this.ibuf, 0, new Uint8Array(idx.buffer));
    }
    this.device.writeBuffer(this.uniform!, 0, viewProj);
    const pass = this.device.beginPass({ id: null, width, height }, {}); // LOAD color+depth
    pass.setPipeline(this.pipeline!);
    pass.setBindings(this.bindings!);
    pass.setVertexBuffer(0, this.vbuf);
    pass.setIndexBuffer(this.ibuf!, IndexFormat.Uint32);
    pass.drawIndexed(verts, 0, 0);
    pass.end();
  }

  private ensure(): void {
    if (this.pipeline) return;
    const shader = this.device.createShaderModule(COLOR_VERTEX_WGSL, "explosion");
    this.pipeline = this.pipelines.get({
      label: "explosion",
      shader,
      vertexEntry: "vs",
      fragmentEntry: "fs",
      vertexLayout: {
        strideBytes: COLOR_VERTEX_STRIDE,
        attributes: [
          { location: 0, kind: VertexScalarKind.Float32, components: 3, offsetBytes: 0, asInt: false },
          { location: 1, kind: VertexScalarKind.Unorm8, components: 4, offsetBytes: 12, asInt: false },
        ],
      },
      colorFormat: this.colorFormat,
      depthFormat: this.depthFormat,
      depthCompare: CompareFn.Less,
      pass: { depthTest: true, depthWrite: false, blend: true, cull: CullMode.None, topology: PrimitiveTopology.TriangleList },
      bindingLayout: [{ binding: 0, visibility: [ShaderStage.Vertex], type: { kind: "uniform-buffer" } }],
    });
    this.uniform = this.device.createBuffer({ sizeBytes: 64, usage: BufferUsage.Uniform, label: "explosion-uniform" });
    this.bindings = this.device.createBindings({ pipeline: this.pipeline, group: 0, entries: [{ binding: 0, resource: { buffer: this.uniform, offset: 0, size: 64 } }] });
  }

  dispose(): void {
    if (this.vbuf) this.device.destroyBuffer(this.vbuf);
    if (this.ibuf) this.device.destroyBuffer(this.ibuf);
    if (this.uniform) this.device.destroyBuffer(this.uniform);
    this.pipelines.dispose();
    this.pipeline = null;
    this.bindings = null;
    this.vbuf = this.ibuf = this.uniform = null;
    this.bursts.length = 0;
  }
}

function randUnit(): Vec3 {
  const z = Math.random() * 2 - 1;
  const t = Math.random() * Math.PI * 2;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  return [r * Math.cos(t), z, r * Math.sin(t)];
}
function pushTri(out: number[], a: Vec3, b: Vec3, c: Vec3, r: number, g: number, bl: number, al: number): void {
  out.push(a[0], a[1], a[2], r, g, bl, al);
  out.push(b[0], b[1], b[2], r, g, bl, al);
  out.push(c[0], c[1], c[2], r, g, bl, al);
}
