// Transient moving-translucent path for pistons. RENDERER_PLAN.md §18 (Pistons), Phase 4.
//
// THE ORACLE (+ vanilla): a moving translucent block (glass pushed by a piston) is its OWN
// self-sorted, depth-WRITING translucent pass — it is NEVER merged into terrain geometry and it NEVER
// removes terrain sections. Vanilla's `PistonHeadRenderer` draws the moving block with
// `RenderTypes.translucentMovingBlock()` = `createMovingBlockSetup(TRANSLUCENT_BLOCK, true)` whose
// `sortOnUpload()` sorts ONLY the moving block's own faces, and whose pipeline `TRANSLUCENT_BLOCK` uses
// `DepthStencilState.DEFAULT` = (LESS_EQUAL, writeDepth=true). The depth write is what orders the moving
// glass against static terrain glass: terrain translucent (depth-write OFF) drawn AFTER it is depth-
// rejected where the moving block is in front, and blends over it where it is behind. No merge, no
// suppression — so no terrain glass can ever vanish (BUG_REPORT.md "distant glass flicker").
//
// HOW: each frame we (1) gather the moving block's world-space quads, (2) sort them back-to-front among
// THEMSELVES with the same topological sort terrain uses (TranslucentTopoSort), and (3) emit ONE draw
// (depthWrite ⇒ the renderer routes it through the depth-writing translucent pipeline, BEFORE terrain
// translucent — vanilla Fancy order). The moving block legitimately re-meshes each frame (it moves);
// terrain GPU buffers are NEVER touched (TRAP 12.A).
//
// Positions: the buffer uses a LOCAL origin near the moving block so every corner fits the packed
// vertex's [-8,+24) section-local window (VertexFormat). A single moving block always fits.

import type { GpuBufferHandle, GraphicsDevice } from "../core/GraphicsDevice";
import { BufferUsage } from "../core/GraphicsDevice";
import { encodeVertexInline, type RawVertex, VERTEX_STRIDE_BYTES } from "../mesh/VertexFormat";
import { FULLBRIGHT_LIGHT } from "../mesh/Lighting";
import { makeTQuad, type TQuad } from "../mesh/TranslucentCollector";
import { expandQuadOrder, topoSortOrder } from "../mesh/TranslucentTopoSort";
import { SortType } from "../mesh/SortTypes";
import { dist2, TerrainPass, type Vec3 } from "../types";
import type { SectionDraw } from "./TerrainRenderer";

/** One moving translucent quad: 4 world-space corners (CCW from outside), each a full packed vertex. */
export type TransientQuad = readonly [RawVertex, RawVertex, RawVertex, RawVertex];

const SECTION_LOCAL_BIAS = 8; // center the local corners near +8 so they sit inside the [-8,+24) window

export class TransientTranslucent {
  private vbuf: GpuBufferHandle | null = null;
  private ibuf: GpuBufferHandle | null = null;
  private capacityQuads = 0;

  constructor(private readonly device: GraphicsDevice) {}

  /**
   * Build this frame's moving-translucent draw, or null if there are no transient quads. The moving
   * block's quads are sorted back-to-front among THEMSELVES (no nearby terrain glass is decoded or
   * merged) and uploaded to a per-frame vertex+index pair. The returned draw is flagged `depthWrite`
   * so the renderer draws it with the depth-WRITING translucent pipeline, before terrain translucent
   * (vanilla's `translucentMovingBlock` ordering) — no terrain section is ever suppressed.
   */
  update(camPos: Vec3, transient: readonly TransientQuad[]): SectionDraw | null {
    if (transient.length === 0) return null;

    // 1) The moving block's world-space quads (each = 4 corners). Self-only — never any terrain glass.
    const quads: RawVertex[][] = transient.map((q) => [q[0], q[1], q[2], q[3]]);
    const n = quads.length;

    // 2) Local origin: round(centroid) − BIAS keeps every local corner near +8, inside [-8,+24).
    const u = averageCorners(quads);
    const mergeOrigin: Vec3 = [Math.round(u[0]) - SECTION_LOCAL_BIAS, Math.round(u[1]) - SECTION_LOCAL_BIAS, Math.round(u[2]) - SECTION_LOCAL_BIAS];

    // 3) Sort back-to-front with the SAME topological sort terrain uses (world space; the sort math is
    //    translation-consistent). Cycle → distance fallback. Order is over quad indices.
    const tquads: TQuad[] = quads.map((c) => makeTQuad(posOf(c[0]), posOf(c[1]), posOf(c[2]), posOf(c[3])));
    const order = topoSortOrder(tquads, camPos, false) ?? distanceQuadOrder(tquads, camPos);
    const index = expandQuadOrder(order);

    // 4) Encode vertices in append order (the index references them in sorted order), local = world − origin.
    const vbytes = new ArrayBuffer(n * 4 * VERTEX_STRIDE_BYTES);
    const out = new DataView(vbytes);
    let off = 0;
    for (const corners of quads) {
      for (const rv of corners) {
        // A6: encode straight from rv's fields (local = world − origin) — no per-vertex `{...rv}` spread.
        encodeVertexInline(out, off, rv.x - mergeOrigin[0], rv.y - mergeOrigin[1], rv.z - mergeOrigin[2], rv.u, rv.v, rv.normal, rv.colorRGBA, rv.material, rv.light ?? FULLBRIGHT_LIGHT);
        off += VERTEX_STRIDE_BYTES;
      }
    }

    // 5) Upload to per-frame buffers (grown as needed). Both change every frame (the block moves).
    this.ensureBuffers(n);
    this.device.writeBuffer(this.vbuf!, 0, new Uint8Array(vbytes));
    this.device.writeBuffer(this.ibuf!, 0, index);

    // 6) One translucent draw. sortType StaticTopo ⇒ the renderer draws the index we already sorted (it does
    //    NOT re-sort — no sortQuads carried). depthWrite ⇒ the depth-writing translucent pipeline, drawn before
    //    terrain translucent so depth — not a geometry merge — orders the moving glass against static glass.
    return {
      originBlocks: mergeOrigin,
      vertex: this.vbuf!,
      quadCount: n,
      pass: TerrainPass.Translucent,
      index: this.ibuf!,
      sortType: SortType.StaticTopo,
      depthWrite: true,
    };
  }

  private ensureBuffers(quads: number): void {
    if (this.vbuf && quads <= this.capacityQuads) return;
    const cap = Math.max(quads, 64, this.capacityQuads * 2);
    if (this.vbuf) this.device.destroyBuffer(this.vbuf);
    if (this.ibuf) this.device.destroyBuffer(this.ibuf);
    this.vbuf = this.device.createBuffer({ sizeBytes: cap * 4 * VERTEX_STRIDE_BYTES, usage: BufferUsage.Vertex, label: "transient-translucent-vtx" });
    this.ibuf = this.device.createBuffer({ sizeBytes: cap * 6 * 4, usage: BufferUsage.Index, label: "transient-translucent-idx" });
    this.capacityQuads = cap;
  }

  dispose(): void {
    if (this.vbuf) this.device.destroyBuffer(this.vbuf);
    if (this.ibuf) this.device.destroyBuffer(this.ibuf);
    this.vbuf = null;
    this.ibuf = null;
    this.capacityQuads = 0;
  }
}

const posOf = (v: RawVertex): Vec3 => [v.x, v.y, v.z];

function averageCorners(quads: readonly RawVertex[][]): Vec3 {
  let x = 0, y = 0, z = 0, n = 0;
  for (const c of quads) for (const v of c) { x += v.x; y += v.y; z += v.z; n++; }
  return n > 0 ? [x / n, y / n, z / n] : [0, 0, 0];
}

/** Farthest-centroid-first quad order (distance fallback when the topo sort hits a cycle). */
function distanceQuadOrder(quads: readonly TQuad[], camPos: Vec3): number[] {
  const d2 = (q: TQuad) => dist2(q.centroid as Vec3, camPos);
  return Array.from({ length: quads.length }, (_, i) => i).sort((a, b) => d2(quads[b]) - d2(quads[a]));
}
