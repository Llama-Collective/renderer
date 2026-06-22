// Shared plumbing for the special block-entity effects (beacon beam, end portal, wireframe box).
// RENDERER_PLAN.md §18, Phase 4.5f.
//
// Each effect is its own class with its own shader + pipeline(s) + vertex format + blend/depth state —
// mirroring how vanilla builds one immutable `RenderPipeline` per render type from a shared `Snippet`
// (RenderPipelines.java: BEACON_BEAM_*, END_PORTAL, LINES), and how builds per-pass pipelines from
// one `RenderPipeline.builder()`. What they share lives here: the one depth-loading pass, the one uniform
// (viewProj + time + resolution), and the growable quad mesh — so splitting effects costs no extra pass.

import type { GpuBufferHandle, GraphicsDevice, PassEncoder, TextureFormat } from "../../../core/GraphicsDevice";
import { BufferUsage, IndexFormat } from "../../../core/GraphicsDevice";
import type { PipelineCache } from "../../../core/PipelineCache";
import { quadIndices } from "../../../mesh/VertexFormat";
import { clamp01 } from "../../../types";
import type { BESpecialDraw } from "../blockentities";

/** Shared uniform: mat4 viewProj (64) + time (4) + resX (4) + resY (4) + pad (4) = 80, 16-aligned. */
export const UNIFORM_BYTES = 80;

/** GPU resources every effect draws against. One uniform + one PipelineCache, one pass per frame. */
export interface SpecialContext {
  readonly device: GraphicsDevice;
  readonly colorFormat: TextureFormat;
  readonly depthFormat: TextureFormat;
  readonly pipelines: PipelineCache;
  /** Shared uniform (viewProj + time + resolution). Every effect binds it at @group(0) @binding(0). */
  readonly uniform: GpuBufferHandle;
}

/**
 * One special-BE effect. The orchestrator calls `build` (pack this frame's geometry; returns the quad
 * count) on every effect, then opens ONE pass and calls `encode` on the non-empty ones. Each effect owns
 * its own pipeline(s) — so blend mode, depth state, vertex format and shader vary per effect, exactly as
 * vanilla varies them per `RenderPipeline`.
 */
export interface SpecialEffect {
  build(specials: readonly BESpecialDraw[], clock: number): number;
  encode(pass: PassEncoder): void;
  dispose(): void;
}

/**
 * A self-growing quad mesh: one vertex buffer (re-uploaded each frame, grown as needed) plus the shared
 * `0,1,2,0,2,3` quad index buffer. Format-agnostic — the effect packs its own vertex bytes.
 */
export class QuadBatch {
  private vbuf: GpuBufferHandle | null = null;
  private vcap = 0;
  private ibuf: GpuBufferHandle | null = null;
  private icapQuads = 0;
  private quads = 0;

  constructor(
    private readonly device: GraphicsDevice,
    private readonly label: string,
  ) {}

  get quadCount(): number {
    return this.quads;
  }

  /** Upload `quadCount` quads' packed vertex bytes (4 verts each); (re)builds the quad index buffer. */
  upload(data: Uint8Array, quadCount: number): void {
    this.quads = quadCount;
    if (quadCount === 0) return;
    if (!this.vbuf || data.byteLength > this.vcap) {
      if (this.vbuf) this.device.destroyBuffer(this.vbuf);
      this.vcap = Math.max(data.byteLength, 4096);
      this.vbuf = this.device.createBuffer({ sizeBytes: this.vcap, usage: BufferUsage.Vertex, label: `${this.label}-verts` });
    }
    this.device.writeBuffer(this.vbuf, 0, data);
    if (!this.ibuf || quadCount > this.icapQuads) {
      if (this.ibuf) this.device.destroyBuffer(this.ibuf);
      const idx = quadIndices(quadCount);
      this.ibuf = this.device.createBuffer({ sizeBytes: idx.byteLength, usage: BufferUsage.Index, label: `${this.label}-index` });
      this.device.writeBuffer(this.ibuf, 0, idx);
      this.icapQuads = quadCount;
    }
  }

  /** Bind this batch's buffers and draw. The caller must have set the pipeline + bindings already. */
  draw(pass: PassEncoder): void {
    if (this.quads === 0 || !this.vbuf || !this.ibuf) return;
    pass.setVertexBuffer(0, this.vbuf);
    pass.setIndexBuffer(this.ibuf, IndexFormat.Uint32);
    pass.drawIndexed(this.quads * 6, 0, 0);
  }

  dispose(): void {
    if (this.vbuf) this.device.destroyBuffer(this.vbuf);
    if (this.ibuf) this.device.destroyBuffer(this.ibuf);
    this.vbuf = this.ibuf = null;
  }
}

/** Pack a vertex's RGBA (0..1 floats) into one little-endian unorm8x4 word at byte offset `o`. */
export function packRgba(dv: DataView, o: number, r: number, g: number, b: number, a: number): void {
  dv.setUint8(o, Math.round(clamp01(r) * 255));
  dv.setUint8(o + 1, Math.round(clamp01(g) * 255));
  dv.setUint8(o + 2, Math.round(clamp01(b) * 255));
  dv.setUint8(o + 3, Math.round(clamp01(a) * 255));
}
