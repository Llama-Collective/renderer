// Opaque/cutout moving-block geometry as transient TERRAIN draws. PISTON_PLAN.md §5 D1.
//
// A moving block (a piston push) renders through the SAME terrain shader as its static self — identical
// per-face directional shading + cutout alpha-test — so there is NO appearance pop when the slide
// completes and the block re-enters terrain meshing. BAKE-ONCE: the block model is packed to a GPU
// vertex buffer ONCE per blockstate (block-local [0,1] coords) and cached; the slide rides the per-draw
// float `origin` uniform (TerrainRenderer adds it to the decoded local position — TerrainRenderer.ts),
// so per frame only the origin changes — no vertex re-upload. This is the cheap, faithful opaque route.
//
// Translucent moving quads are NOT handled here — they need the local back-to-front sort
// (TransientTranslucent, option 2 / TRAP 18.A). Opaque + cutout need no per-quad sort
// (depth-write resolves order), so this is just "bake once, draw at a moving origin".

import type { GpuBufferHandle, GraphicsDevice } from "../../core/GraphicsDevice";
import { BufferUsage } from "../../core/GraphicsDevice";
import { packVertices, type RawVertex } from "../../mesh/VertexFormat";
import { TerrainPass, type Vec3 } from "../../types";
import type { SectionDraw } from "../TerrainRenderer";

type OpaquePass = TerrainPass.Solid | TerrainPass.Cutout;
const OPAQUE_PASSES: readonly OpaquePass[] = [TerrainPass.Solid, TerrainPass.Cutout];

interface OpaqueBuffer {
  buf: GpuBufferHandle;
  quadCount: number;
}

/** Caches baked opaque/cutout block geometry (keyed by blockstate string) and emits per-frame draws. */
export class TransientOpaque {
  private readonly cache = new Map<string, Partial<Record<OpaquePass, OpaqueBuffer>>>();

  constructor(private readonly device: GraphicsDevice) {}

  /** Pack + upload the Solid/Cutout vertex buckets for a blockstate ONCE (idempotent; cached by key). */
  ensure(key: string, solid: readonly RawVertex[] | undefined, cutout: readonly RawVertex[] | undefined): void {
    if (this.cache.has(key)) return;
    const byPass: Partial<Record<OpaquePass, OpaqueBuffer>> = {};
    const buckets: Record<OpaquePass, readonly RawVertex[] | undefined> = {
      [TerrainPass.Solid]: solid,
      [TerrainPass.Cutout]: cutout,
    };
    for (const pass of OPAQUE_PASSES) {
      const verts = buckets[pass];
      if (!verts || verts.length === 0) continue;
      const bytes = packVertices(verts);
      const buf = this.device.createBuffer({ sizeBytes: bytes.byteLength, usage: BufferUsage.Vertex, label: `piston-opaque-${pass}` });
      this.device.writeBuffer(buf, 0, new Uint8Array(bytes));
      byPass[pass] = { buf, quadCount: verts.length / 4 };
    }
    this.cache.set(key, byPass);
  }

  /** True once `ensure(key, …)` has run for this key (even when it produced no opaque geometry). */
  has(key: string): boolean {
    return this.cache.has(key);
  }

  /** Emit terrain draws for a cached blockstate at a (fractional) world origin. */
  draws(key: string, origin: Vec3): SectionDraw[] {
    const byPass = this.cache.get(key);
    if (!byPass) return [];
    const out: SectionDraw[] = [];
    for (const pass of OPAQUE_PASSES) {
      const b = byPass[pass];
      if (b) out.push({ originBlocks: origin, vertex: b.buf, quadCount: b.quadCount, pass });
    }
    return out;
  }

  /** Destroy all cached buffers (call on atlas/scene rebuild — the baked UVs are now stale). */
  clear(): void {
    for (const byPass of this.cache.values()) {
      for (const pass of OPAQUE_PASSES) {
        const b = byPass[pass];
        if (b) this.device.destroyBuffer(b.buf);
      }
    }
    this.cache.clear();
  }

  dispose(): void {
    this.clear();
  }
}
