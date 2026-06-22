// Editor overlays: the hover cursor outline, the placement face-region selector grid, and the rail
// selector. Drawn AFTER terrain + entities in a depth-LOADED pass with depth-test OFF (always visible,
// like the old three.js overlays' depthTest:false). Geometry is ported verbatim from the old renderer's
// showCursorForPick / showFaceRegions / showRailRegions. RENDERER_INTEGRATION_PLAN Step 7.

import type { BindingsHandle, GpuBufferHandle, GraphicsDevice, PipelineHandle, TextureFormat } from "../../core/GraphicsDevice";
import { BufferUsage, CompareFn, CullMode, IndexFormat, PrimitiveTopology, ShaderStage, VertexScalarKind } from "../../core/GraphicsDevice";
import { PipelineCache } from "../../core/PipelineCache";
import type { Mat4 } from "../../camera/Camera";
import type { Vec3 } from "../../types";
import type { FaceRegion, PickResult, Vec3Tuple } from "../../../../schematic-io/types";
import type { OutlineBox } from "../../app/pick";
import { faceBasis, faceRegionForPick } from "../../app/placement";
import { unpackRgb } from "../color";
import { COLOR_VERTEX_STRIDE, COLOR_VERTEX_WGSL, packColorVerts, sequentialIndexBuffer } from "../colorVertex";

const COLD = 0xeaf6ff;
const HOT = 0x29e0ff;
const FILL = 0x29e0ff;
const CURSOR = 0xffd166;
const RAIL_STRAIGHT_SQUARE = (1 / 3) * 2;

// The floor grid uses its own shader so it can fade with XZ distance from the build centre — a port of
// the old renderer's distance-faded ground grid (Grid.ts createGridLines: smoothstep(fade*0.5, fade)).
const GRID_WGSL = /* wgsl */ `
struct GU { viewProj : mat4x4<f32>, params : vec4<f32> }; // params = (centreX, centreZ, fadeStart, fadeEnd)
@group(0) @binding(0) var<uniform> u : GU;
struct VsIn { @location(0) pos : vec3<f32>, @location(1) color : vec4<f32> };
struct VsOut { @builtin(position) clip : vec4<f32>, @location(0) color : vec4<f32>, @location(1) world : vec3<f32> };
@vertex fn vs(in : VsIn) -> VsOut {
  var o : VsOut;
  o.clip = u.viewProj * vec4<f32>(in.pos, 1.0);
  o.color = in.color;
  o.world = in.pos;
  return o;
}
@fragment fn fs(in : VsOut) -> @location(0) vec4<f32> {
  let d = distance(in.world.xz, u.params.xy);
  let fade = 1.0 - smoothstep(u.params.z, u.params.w, d);
  return vec4<f32>(in.color.rgb, in.color.a * fade);
}
`;

export class OverlayRenderer {
  private readonly pipelines: PipelineCache;
  private lineP: PipelineHandle | null = null;
  private triP: PipelineHandle | null = null;
  private uniform: GpuBufferHandle | null = null;
  private bindings: BindingsHandle | null = null;
  private lineBuf: GpuBufferHandle | null = null;
  private triBuf: GpuBufferHandle | null = null;
  private lineVerts = 0;
  private triVerts = 0;
  private lineCap = 0;
  private triCap = 0;
  private idxBuf: GpuBufferHandle | null = null;
  private idxCap = 0;
  private gridP: PipelineHandle | null = null;
  private gridBuf: GpuBufferHandle | null = null;
  private gridUniform: GpuBufferHandle | null = null;
  private gridBindings: BindingsHandle | null = null;
  private gridVerts = 0;
  private gridCap = 0;
  /** Fade params written into the grid uniform: [centreX, centreZ, fadeStart, fadeEnd]. */
  private gridStage = new Float32Array(20);
  private gridCenter: [number, number] = [0, 0];
  private gridFadeStart = 50;
  private gridFadeEnd = 100;

  // Accumulators (cursor + regions are set independently and combined at upload).
  private cursorLines: number[] = [];
  private regionLines: number[] = [];
  private regionFills: number[] = [];
  /** The floor grid (y=0). Depth-TESTED (terrain occludes it), unlike the on-top cursor/region lines. */
  private gridLines: number[] = [];
  private dirty = true;

  constructor(
    private readonly device: GraphicsDevice,
    private readonly colorFormat: TextureFormat,
    private readonly depthFormat: TextureFormat,
  ) {
    this.pipelines = new PipelineCache(device);
  }

  // ── Public API (mirrors EditableWorldRenderer) ───────────────────────────────

  setCursor(pick: PickResult | null, boxes: ReadonlyArray<OutlineBox> | null): void {
    this.cursorLines = [];
    if (pick && pick.type !== "entity") {
      const list = boxes && boxes.length ? boxes : ([[0, 0, 0, 1, 1, 1]] as OutlineBox[]);
      const [ox, oy, oz] = pick.position;
      for (const b of list) boxEdges(this.cursorLines, [ox + b[0], oy + b[1], oz + b[2]], [ox + b[3], oy + b[4], oz + b[5]], CURSOR, 0.92);
    }
    this.dirty = true;
  }
  clearCursor(): void {
    if (this.cursorLines.length) this.dirty = true;
    this.cursorLines = [];
  }

  setFaceRegions(pick: PickResult | null, regionStates?: Partial<Record<FaceRegion, string | null>>): void {
    this.regionLines = [];
    this.regionFills = [];
    if (pick && (pick.type === "block" || pick.type === "ground")) this.buildFaceRegions(pick, regionStates);
    this.dirty = true;
  }

  setRail(pick: PickResult | null, shape: string, canCurve: boolean): void {
    this.regionLines = [];
    this.regionFills = [];
    if (pick && (pick.type === "block" || pick.type === "ground")) this.buildRail(pick, shape, canCurve);
    this.dirty = true;
  }

  clearRegions(): void {
    if (this.regionLines.length || this.regionFills.length) this.dirty = true;
    this.regionLines = [];
    this.regionFills = [];
  }

  /** Build the ground-reference grid at y=0 (null clears it). A faithful port of the old renderer's
   *  grid: lines every `majorStep` (8) blocks in world space, colour 0xaaaaaa, fading with distance
   *  from the build centre (so most of it sits in open space AROUND the build and isn't occluded by it,
   *  which is why a build-bounded grid was invisible). Depth-tested so terrain occludes its near parts. */
  setGrid(bounds: { min: readonly [number, number, number]; max: readonly [number, number, number] } | null): void {
    this.gridLines = [];
    this.dirty = true;
    // Empty world → a default region around the origin so the grid is still visible on a blank canvas.
    const min = bounds?.min ?? [0, 0, 0];
    const max = bounds?.max ?? [16, 0, 16];
    // Minor lines every block; major lines every 16 (world chunk boundaries) — outlines chunks.
    const CHUNK = 16;
    const MINOR = 0x5a6473;
    const MAJOR = 0x9aa6b5;
    const cx = (min[0] + max[0]) / 2;
    const cz = (min[2] + max[2]) / 2;
    const maxDim = Math.max(max[0] - min[0], max[2] - min[2], 16);
    // fadeEnd scales with the build so the grid always surrounds it; classic ~100 for small builds.
    const fadeEnd = Math.max(100, maxDim * 0.9 + 48);
    this.gridCenter = [cx, cz];
    this.gridFadeStart = fadeEnd * 0.5;
    this.gridFadeEnd = fadeEnd;
    // Generate every block out to the fade, snapped to integer (block-aligned) world coords. The fade
    // (shader) hides the far edges; major lines fall on multiples of 16 = chunk grid.
    const R = fadeEnd + 2;
    const x0 = Math.floor(cx - R);
    const x1 = Math.ceil(cx + R);
    const z0 = Math.floor(cz - R);
    const z1 = Math.ceil(cz + R);
    for (let x = x0; x <= x1; x++) {
      const major = ((x % CHUNK) + CHUNK) % CHUNK === 0;
      pushLine(this.gridLines, [x, 0, z0], [x, 0, z1], major ? MAJOR : MINOR, major ? 0.85 : 0.4);
    }
    for (let z = z0; z <= z1; z++) {
      const major = ((z % CHUNK) + CHUNK) % CHUNK === 0;
      pushLine(this.gridLines, [x0, 0, z], [x1, 0, z], major ? MAJOR : MINOR, major ? 0.85 : 0.4);
    }
  }

  /** Hide the ground grid — drop all floor lines so nothing is drawn (the inverse of {@link setGrid}). */
  clearGrid(): void {
    this.gridLines = [];
    this.dirty = true;
  }

  // ── Geometry generators (ported from EditableWorldRenderer) ──────────────────

  private faceFrame(pick: PickResult): { center: Vec3; rv: Vec3; uv: Vec3 } {
    const [bx, byRaw, bz] = pick.position;
    const by = pick.type === "ground" ? byRaw - 1 : byRaw;
    const [nx, ny, nz] = pick.normal;
    const off = 0.502;
    const center: Vec3 = [bx + 0.5 + nx * off, by + 0.5 + ny * off, bz + 0.5 + nz * off];
    const [right, up] = faceBasis(pick.normal as Vec3Tuple);
    return { center, rv: right, uv: up };
  }

  private buildFaceRegions(pick: PickResult, regionStates?: Partial<Record<FaceRegion, string | null>>): void {
    const { center, rv, uv } = this.faceFrame(pick);
    const corner = (su: number, sv: number): Vec3 => [
      center[0] + rv[0] * su * 0.5 + uv[0] * sv * 0.5,
      center[1] + rv[1] * su * 0.5 + uv[1] * sv * 0.5,
      center[2] + rv[2] * su * 0.5 + uv[2] * sv * 0.5,
    ];
    const region = faceRegionForPick(pick);
    const highlight = (r: FaceRegion): boolean =>
      regionStates ? regionStates[r] !== undefined && regionStates[r] === regionStates[region] : r === region;
    const mergeCentral = !!regionStates && regionStates.centermost !== undefined && regionStates.centermost === regionStates.center;
    const seg = (a: Vec3, b: Vec3, hot: boolean): void => pushLine(this.regionLines, a, b, hot ? HOT : COLD, hot ? 1 : 0.7);
    const fillQuad = (a: Vec3, b: Vec3, c: Vec3, d: Vec3): void => pushQuad(this.regionFills, a, b, c, d, FILL, 0.32);

    const inner = [corner(-1 / 3, -1 / 3), corner(1 / 3, -1 / 3), corner(1 / 3, 1 / 3), corner(-1 / 3, 1 / 3)];
    const mid = [corner(-2 / 3, -2 / 3), corner(2 / 3, -2 / 3), corner(2 / 3, 2 / 3), corner(-2 / 3, 2 / 3)];
    const outer = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
    for (let i = 0; i < 4; i++) seg(outer[i], outer[(i + 1) % 4], false);
    for (let i = 0; i < 4; i++) seg(mid[i], mid[(i + 1) % 4], false);
    if (!mergeCentral) for (let i = 0; i < 4; i++) seg(inner[i], inner[(i + 1) % 4], false);
    for (let i = 0; i < 4; i++) seg(mid[i], outer[i], false);
    const edgePairs: Record<"up" | "down" | "left" | "right", [number, number]> = { down: [0, 1], right: [1, 2], up: [2, 3], left: [3, 0] };

    if (mergeCentral && (highlight("centermost") || highlight("center"))) {
      fillQuad(mid[0], mid[1], mid[2], mid[3]);
      for (let i = 0; i < 4; i++) seg(mid[i], mid[(i + 1) % 4], true);
    } else {
      if (highlight("centermost")) {
        fillQuad(inner[0], inner[1], inner[2], inner[3]);
        for (let i = 0; i < 4; i++) seg(inner[i], inner[(i + 1) % 4], true);
      }
      if (highlight("center")) {
        for (let i = 0; i < 4; i++) {
          const j = (i + 1) % 4;
          fillQuad(mid[i], mid[j], inner[j], inner[i]);
        }
        for (let i = 0; i < 4; i++) {
          seg(mid[i], mid[(i + 1) % 4], true);
          seg(inner[i], inner[(i + 1) % 4], true);
        }
      }
    }
    for (const edge of ["up", "down", "left", "right"] as const) {
      if (!highlight(edge)) continue;
      const [a, b] = edgePairs[edge];
      fillQuad(outer[a], outer[b], mid[b], mid[a]);
      seg(outer[a], outer[b], true);
      seg(mid[a], outer[a], true);
      seg(mid[b], outer[b], true);
      seg(mid[a], mid[b], true);
    }
  }

  private buildRail(pick: PickResult, shape: string, canCurve: boolean): void {
    const { center, rv, uv } = this.faceFrame(pick);
    const ny = pick.normal[1];
    const at = (su: number, sv: number): Vec3 => [
      center[0] + rv[0] * su * 0.5 + uv[0] * sv * 0.5,
      center[1] + rv[1] * su * 0.5 + uv[1] * sv * 0.5,
      center[2] + rv[2] * su * 0.5 + uv[2] * sv * 0.5,
    ];
    const seg = (a: Vec3, b: Vec3, hot: boolean): void => pushLine(this.regionLines, a, b, hot ? HOT : COLD, hot ? 1 : 0.7);
    const fillTri = (a: Vec3, b: Vec3, c: Vec3): void => pushTri(this.regionFills, a, b, c, FILL, 0.32);
    const fillRect = (su0: number, sv0: number, su1: number, sv1: number): void => {
      fillTri(at(su0, sv0), at(su1, sv0), at(su1, sv1));
      fillTri(at(su0, sv0), at(su1, sv1), at(su0, sv1));
    };
    const rectOutline = (su0: number, sv0: number, su1: number, sv1: number, hot: boolean): void => {
      seg(at(su0, sv0), at(su1, sv0), hot);
      seg(at(su1, sv0), at(su1, sv1), hot);
      seg(at(su1, sv1), at(su0, sv1), hot);
      seg(at(su0, sv1), at(su0, sv0), hot);
    };

    if (ny !== 0) {
      const sq = canCurve ? RAIL_STRAIGHT_SQUARE : 1;
      rectOutline(-1, -1, 1, 1, false);
      seg(at(-sq, -sq), at(sq, sq), false);
      seg(at(-sq, sq), at(sq, -sq), false);
      if (canCurve) {
        rectOutline(-sq, -sq, sq, sq, false);
        seg(at(0, sq), at(0, 1), false);
        seg(at(0, -sq), at(0, -1), false);
        seg(at(sq, 0), at(1, 0), false);
        seg(at(-sq, 0), at(-1, 0), false);
      }
      if (shape === "north_south") {
        fillTri(at(0, 0), at(-sq, sq), at(sq, sq));
        fillTri(at(0, 0), at(-sq, -sq), at(sq, -sq));
        seg(at(0, 0), at(-sq, sq), true);
        seg(at(0, 0), at(sq, sq), true);
        seg(at(0, 0), at(-sq, -sq), true);
        seg(at(0, 0), at(sq, -sq), true);
      } else if (shape === "east_west") {
        fillTri(at(0, 0), at(sq, sq), at(sq, -sq));
        fillTri(at(0, 0), at(-sq, sq), at(-sq, -sq));
        seg(at(0, 0), at(sq, sq), true);
        seg(at(0, 0), at(sq, -sq), true);
        seg(at(0, 0), at(-sq, sq), true);
        seg(at(0, 0), at(-sq, -sq), true);
      } else {
        const sx = shape.endsWith("east") ? 1 : -1;
        const sy = shape.startsWith("north") ? 1 : -1;
        fillRect(sx * sq, 0, sx, sy);
        fillRect(0, sy * sq, sx * sq, sy);
        rectOutline(sx * sq, 0, sx, sy, true);
        rectOutline(0, sy * sq, sx * sq, sy, true);
      }
    } else {
      seg(at(-1, 0), at(1, 0), false);
      rectOutline(-1, -1, 1, 1, false);
      const ascending = shape.startsWith("ascending_");
      if (ascending) {
        fillRect(-1, 0, 1, 1);
        rectOutline(-1, 0, 1, 1, true);
      } else {
        fillRect(-1, -1, 1, 0);
        rectOutline(-1, -1, 1, 0, true);
      }
    }
  }

  // ── GPU ──────────────────────────────────────────────────────────────────────

  render(viewProj: Mat4, width: number, height: number): void {
    if (this.dirty) {
      this.upload();
      this.dirty = false;
    }
    if (this.gridVerts === 0 && this.lineVerts === 0 && this.triVerts === 0) return;
    this.ensure();
    this.device.writeBuffer(this.uniform!, 0, viewProj);
    const target = { id: null, width, height } as const;
    const pass = this.device.beginPass(target, {}); // LOAD terrain + entity color/depth
    // Floor grid first — depth-TESTED so terrain occludes it (a real ground plane, not an on-top overlay).
    if (this.gridVerts > 0 && this.gridBuf && this.idxBuf) {
      this.gridStage.set(viewProj, 0);
      this.gridStage[16] = this.gridCenter[0];
      this.gridStage[17] = this.gridCenter[1];
      this.gridStage[18] = this.gridFadeStart;
      this.gridStage[19] = this.gridFadeEnd;
      this.device.writeBuffer(this.gridUniform!, 0, this.gridStage);
      pass.setPipeline(this.gridP!);
      pass.setBindings(this.gridBindings!);
      pass.setVertexBuffer(0, this.gridBuf);
      pass.setIndexBuffer(this.idxBuf, IndexFormat.Uint32);
      pass.drawIndexed(this.gridVerts, 0, 0);
    }
    if (this.triVerts > 0 && this.triBuf && this.idxBuf) {
      pass.setPipeline(this.triP!);
      pass.setBindings(this.bindings!);
      pass.setVertexBuffer(0, this.triBuf);
      pass.setIndexBuffer(this.idxBuf, IndexFormat.Uint32);
      pass.drawIndexed(this.triVerts, 0, 0);
    }
    if (this.lineVerts > 0 && this.lineBuf && this.idxBuf) {
      pass.setPipeline(this.lineP!);
      pass.setBindings(this.bindings!);
      pass.setVertexBuffer(0, this.lineBuf);
      pass.setIndexBuffer(this.idxBuf, IndexFormat.Uint32);
      pass.drawIndexed(this.lineVerts, 0, 0);
    }
    pass.end();
  }

  private upload(): void {
    const lines = this.cursorLines.length || this.regionLines.length ? this.cursorLines.concat(this.regionLines) : [];
    const fills = this.regionFills;
    const grid = this.gridLines;
    this.gridVerts = grid.length / 7;
    this.lineVerts = lines.length / 7;
    this.triVerts = fills.length / 7;
    // grid + line-list + triangle-list all draw a plain sequential index buffer [0,1,2,…].
    const maxVerts = Math.max(this.gridVerts, this.lineVerts, this.triVerts);
    if (maxVerts > this.idxCap) {
      if (this.idxBuf) this.device.destroyBuffer(this.idxBuf);
      this.idxCap = maxVerts;
      const idx = sequentialIndexBuffer(maxVerts);
      this.idxBuf = this.device.createBuffer({ sizeBytes: idx.byteLength, usage: BufferUsage.Index, label: "overlay-index" });
      this.device.writeBuffer(this.idxBuf, 0, new Uint8Array(idx.buffer));
    }
    if (this.gridVerts > 0) {
      const data = packColorVerts(grid);
      if (!this.gridBuf || data.byteLength > this.gridCap) {
        if (this.gridBuf) this.device.destroyBuffer(this.gridBuf);
        this.gridCap = Math.max(data.byteLength, 4096);
        this.gridBuf = this.device.createBuffer({ sizeBytes: this.gridCap, usage: BufferUsage.Vertex, label: "overlay-grid" });
      }
      this.device.writeBuffer(this.gridBuf, 0, data);
    }
    if (this.lineVerts > 0) {
      const data = packColorVerts(lines);
      if (!this.lineBuf || data.byteLength > this.lineCap) {
        if (this.lineBuf) this.device.destroyBuffer(this.lineBuf);
        this.lineCap = Math.max(data.byteLength, 4096);
        this.lineBuf = this.device.createBuffer({ sizeBytes: this.lineCap, usage: BufferUsage.Vertex, label: "overlay-lines" });
      }
      this.device.writeBuffer(this.lineBuf, 0, data);
    }
    if (this.triVerts > 0) {
      const data = packColorVerts(fills);
      if (!this.triBuf || data.byteLength > this.triCap) {
        if (this.triBuf) this.device.destroyBuffer(this.triBuf);
        this.triCap = Math.max(data.byteLength, 4096);
        this.triBuf = this.device.createBuffer({ sizeBytes: this.triCap, usage: BufferUsage.Vertex, label: "overlay-tris" });
      }
      this.device.writeBuffer(this.triBuf, 0, data);
    }
  }

  private ensure(): void {
    if (this.lineP) return;
    const shader = this.device.createShaderModule(COLOR_VERTEX_WGSL, "overlay");
    const layout = {
      strideBytes: COLOR_VERTEX_STRIDE,
      attributes: [
        { location: 0, kind: VertexScalarKind.Float32, components: 3, offsetBytes: 0, asInt: false },
        { location: 1, kind: VertexScalarKind.Unorm8, components: 4, offsetBytes: 12, asInt: false },
      ],
    };
    const base = {
      shader,
      vertexEntry: "vs",
      fragmentEntry: "fs",
      vertexLayout: layout,
      colorFormat: this.colorFormat,
      depthFormat: this.depthFormat,
      depthCompare: CompareFn.LessEqual,
      bindingLayout: [{ binding: 0, visibility: [ShaderStage.Vertex], type: { kind: "uniform-buffer" } as const }],
    };
    // Overlays are always-on-top: depth test + write OFF, blend on, no cull.
    this.triP = this.pipelines.get({ ...base, label: "overlay-tri", pass: { depthTest: false, depthWrite: false, blend: true, cull: CullMode.None, topology: PrimitiveTopology.TriangleList } });
    this.lineP = this.pipelines.get({ ...base, label: "overlay-line", pass: { depthTest: false, depthWrite: false, blend: true, cull: CullMode.None, topology: PrimitiveTopology.LineList } });
    this.uniform = this.device.createBuffer({ sizeBytes: 64, usage: BufferUsage.Uniform, label: "overlay-uniform" });
    this.bindings = this.device.createBindings({ pipeline: this.triP, group: 0, entries: [{ binding: 0, resource: { buffer: this.uniform, offset: 0, size: 64 } }] });

    // The floor grid IS depth-tested (depthCompare LessEqual from `base`) so terrain occludes its near
    // parts — it's a ground plane, not an always-on-top editor overlay. Depth-write stays off so it never
    // blocks anything. Its own shader fades the lines with distance from the build centre.
    const gridShader = this.device.createShaderModule(GRID_WGSL, "overlay-grid");
    this.gridP = this.pipelines.get({
      ...base, shader: gridShader, label: "overlay-grid",
      // The grid FRAGMENT shader reads the uniform (fade params), so it must be visible to BOTH stages —
      // the shared `base` layout is Vertex-only (the line/tri shaders only read it in the vertex stage).
      bindingLayout: [{ binding: 0, visibility: [ShaderStage.Vertex, ShaderStage.Fragment], type: { kind: "uniform-buffer" } as const }],
      pass: { depthTest: true, depthWrite: false, blend: true, cull: CullMode.None, topology: PrimitiveTopology.LineList },
    });
    this.gridUniform = this.device.createBuffer({ sizeBytes: 80, usage: BufferUsage.Uniform, label: "overlay-grid-uniform" });
    this.gridBindings = this.device.createBindings({ pipeline: this.gridP, group: 0, entries: [{ binding: 0, resource: { buffer: this.gridUniform, offset: 0, size: 80 } }] });
  }

  dispose(): void {
    if (this.lineBuf) this.device.destroyBuffer(this.lineBuf);
    if (this.triBuf) this.device.destroyBuffer(this.triBuf);
    if (this.gridBuf) this.device.destroyBuffer(this.gridBuf);
    if (this.idxBuf) this.device.destroyBuffer(this.idxBuf);
    if (this.uniform) this.device.destroyBuffer(this.uniform);
    if (this.gridUniform) this.device.destroyBuffer(this.gridUniform);
    this.pipelines.dispose();
    this.lineP = this.triP = this.gridP = null;
    this.bindings = this.gridBindings = null;
    this.lineBuf = this.triBuf = this.gridBuf = this.uniform = this.gridUniform = null;
  }
}

type Col = [number, number, number, number];
function col(rgb: number, a: number): Col {
  const [r, g, b] = unpackRgb(rgb);
  return [r, g, b, a];
}
function pushVert(out: number[], p: Vec3, c: Col): void {
  out.push(p[0], p[1], p[2], c[0], c[1], c[2], c[3]);
}
function pushLine(out: number[], a: Vec3, b: Vec3, rgb: number, alpha: number): void {
  const c = col(rgb, alpha);
  pushVert(out, a, c);
  pushVert(out, b, c);
}
function pushTri(out: number[], a: Vec3, b: Vec3, c: Vec3, rgb: number, alpha: number): void {
  const k = col(rgb, alpha);
  pushVert(out, a, k);
  pushVert(out, b, k);
  pushVert(out, c, k);
}
function pushQuad(out: number[], a: Vec3, b: Vec3, c: Vec3, d: Vec3, rgb: number, alpha: number): void {
  pushTri(out, a, b, c, rgb, alpha);
  pushTri(out, a, c, d, rgb, alpha);
}
function boxEdges(out: number[], min: Vec3, max: Vec3, rgb: number, alpha: number): void {
  const C: Vec3[] = [
    [min[0], min[1], min[2]], [max[0], min[1], min[2]], [max[0], min[1], max[2]], [min[0], min[1], max[2]],
    [min[0], max[1], min[2]], [max[0], max[1], min[2]], [max[0], max[1], max[2]], [min[0], max[1], max[2]],
  ];
  const E: [number, number][] = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  for (const [i, j] of E) pushLine(out, C[i], C[j], rgb, alpha);
}
