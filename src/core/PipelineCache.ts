// Shader/program/pipeline cache. RENDERER_PLAN.md §10.
//
// Pipelines are immutable and expensive to compile, so get-or-create them once and reuse. One
// terrain shader family serves the three passes (the pass differs only in PassStateDesc + a few
// override constants), so most frames hit the cache. WebGPU pipelines are GC'd (no explicit
// destroy), so dispose() just drops references.

import type { GraphicsDevice, PipelineDesc, PipelineHandle, ShaderModuleHandle } from "./GraphicsDevice";

// Shader modules are OPAQUE handles (identity, not value), so the cache key needs a stable per-module id
// to distinguish pipelines built from different shaders. Module-level WeakMap: same module → same id.
const shaderIds = new WeakMap<object, number>();
let nextShaderId = 1;
function shaderId(h: ShaderModuleHandle): number {
  let id = shaderIds.get(h as object);
  if (id === undefined) shaderIds.set(h as object, (id = nextShaderId++));
  return id;
}

/**
 * Identity of a pipeline for caching: EVERYTHING that changes the compiled object — shader module,
 * entry points, vertex layout, bind-group layout, formats, fixed-function state, and override constants.
 * (Omitting shader/vertexLayout/bindingLayout was only safe while every call site used a unique label —
 * a reused label would otherwise return a pipeline built from the wrong shader/attributes. AUDIT M3.)
 */
function keyOf(desc: PipelineDesc): string {
  const p = desc.pass;
  const consts = desc.constants ? JSON.stringify(desc.constants) : "";
  return [
    desc.label,
    `sh${shaderId(desc.shader)}`,
    desc.vertexEntry ?? "vs",
    desc.fragmentEntry ?? "fs",
    JSON.stringify(desc.vertexLayout),
    desc.bindingLayout ? JSON.stringify(desc.bindingLayout) : "auto",
    desc.colorFormat,
    desc.depthFormat ?? "none",
    desc.depthCompare ?? "default",
    `${p.depthTest ? 1 : 0}${p.depthWrite ? 1 : 0}${p.blend ? 1 : 0}`,
    p.cull ?? "back",
    consts,
  ].join("|");
}

export class PipelineCache {
  private readonly pipelines = new Map<string, PipelineHandle>();

  constructor(private readonly device: GraphicsDevice) {}

  /** Get-or-create a pipeline; cached by a key derived from the desc. */
  get(desc: PipelineDesc): PipelineHandle {
    const key = keyOf(desc);
    let p = this.pipelines.get(key);
    if (!p) {
      p = this.device.createPipeline(desc);
      this.pipelines.set(key, p);
    }
    return p;
  }

  get size(): number {
    return this.pipelines.size;
  }

  dispose(): void {
    this.pipelines.clear();
  }
}
