// Special block-entity draws orchestrator: beacon beam, end portal/gateway, wireframe boxes.
// RENDERER_PLAN.md §18, Phase 4.5f.
//
// Each effect is its OWN class (special/BeamEffect, PortalEffect, BoxEffect) with its own shader, vertex
// format and pipeline(s) + blend/depth state — mirroring how vanilla builds one `RenderPipeline` per
// render type (RenderPipelines.java) and builds per-pass pipelines, all from a shared base. What
// they share lives here: one depth-loading pass and one uniform (viewProj + time + resolution). So they
// stay batched into a SINGLE pass (drawn opaque→translucent) while diverging in pipeline state.
//
// This pass LOADS terrain+entity color+depth (TRAP 18.B) and runs after EntityRenderer. Per-effect
// geometry is rebuilt CPU-side each frame from `BESpecialDraw` descriptors (counts are tiny).

import type { GpuBufferHandle, GraphicsDevice, TextureFormat } from "../../core/GraphicsDevice";
import { BufferUsage } from "../../core/GraphicsDevice";
import { PipelineCache } from "../../core/PipelineCache";
import type { Mat4 } from "../../mesh/entity/mat4";
import type { BESpecialDraw } from "./blockentities";
import { UNIFORM_BYTES, type SpecialContext, type SpecialEffect } from "./special/shared";
import { BeamEffect } from "./special/BeamEffect";
import { BoxEffect } from "./special/BoxEffect";
import { PortalEffect, type PortalTextures } from "./special/PortalEffect";

export type { PortalTextures, RawTex } from "./special/PortalEffect";

export class SpecialRenderer {
  private readonly pipelines: PipelineCache;
  private readonly uniform: GpuBufferHandle;
  private readonly uniformData = new Float32Array(UNIFORM_BYTES / 4);
  private readonly portal: PortalEffect;
  /** Draw order: opaque portal first, then the box, then the translucent beam glow last. */
  private readonly effects: SpecialEffect[];
  stats = { quads: 0 };

  constructor(
    private readonly device: GraphicsDevice,
    colorFormat: TextureFormat,
    depthFormat: TextureFormat,
  ) {
    this.pipelines = new PipelineCache(device);
    this.uniform = device.createBuffer({ sizeBytes: UNIFORM_BYTES, usage: BufferUsage.Uniform, label: "special-uniform" });
    const ctx: SpecialContext = { device, colorFormat, depthFormat, pipelines: this.pipelines, uniform: this.uniform };
    this.portal = new PortalEffect(ctx);
    this.effects = [this.portal, new BoxEffect(ctx), new BeamEffect(ctx)];
  }

  /** Supply the end-portal shader's textures (Sampler0 = end_sky, Sampler1 = end_portal). Call once. */
  setPortalTextures(tex: PortalTextures): void {
    this.portal.setTextures(tex);
  }

  /** Draw the frame's special BE descriptors. Call after EntityRenderer (loads color+depth). */
  render(specials: readonly BESpecialDraw[], viewProj: Mat4, clock: number, width: number, height: number): void {
    const counts = this.effects.map((e) => e.build(specials, clock));
    const total = counts.reduce((a, b) => a + b, 0);
    this.stats = { quads: total };
    if (total === 0) return;

    this.uniformData.set(viewProj, 0);
    this.uniformData[16] = clock;
    this.uniformData[17] = width;
    this.uniformData[18] = height;
    this.device.writeBuffer(this.uniform, 0, this.uniformData);

    const pass = this.device.beginPass({ id: null, width, height }, {});
    this.effects.forEach((e, i) => {
      if (counts[i] > 0) e.encode(pass);
    });
    pass.end();
  }

  dispose(): void {
    for (const e of this.effects) e.dispose();
    this.device.destroyBuffer(this.uniform);
    this.pipelines.dispose();
  }
}
