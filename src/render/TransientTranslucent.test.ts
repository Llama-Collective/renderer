// Transient moving-translucent draw (RENDERER_PLAN Phase 4 / §18; vanilla `translucentMovingBlock`).
// The moving block's glass is sorted back-to-front AMONG ITSELF and emitted as ONE depth-WRITING draw —
// no nearby terrain glass is decoded or merged, and NO terrain section is suppressed (BUG_REPORT.md).
// GPU-free (FakeGraphicsDevice); the sort + encoding math is what's verified.

import { describe, it, expect } from "vitest";
import { TransientTranslucent, type TransientQuad } from "./TransientTranslucent";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { decodeVertex, type RawVertex, VERTEX_STRIDE_BYTES } from "../mesh/VertexFormat";
import { SortType } from "../mesh/SortTypes";
import { TerrainPass } from "../types";

/** A +Z-facing unit quad at plane z, in world space (CCW from outside → normal +Z). */
function zQuad(z: number, color = 0xffffffff): TransientQuad {
  const v = (x: number, y: number): RawVertex => ({ x, y, z, u: x, v: y, normal: 2, colorRGBA: color, material: 0 });
  return [v(0, 0), v(1, 0), v(1, 1), v(0, 1)];
}

/** Decode the index buffer into the quad VISIT order (first-drawn quad first). */
function visitOrder(dev: FakeGraphicsDevice, draw: { index?: unknown; quadCount: number }): number[] {
  const bytes = dev.read(draw.index as never);
  const u32 = new Uint32Array(bytes.buffer, bytes.byteOffset, draw.quadCount * 6);
  const order: number[] = [];
  for (let q = 0; q < draw.quadCount; q++) order.push(u32[q * 6] / 4); // base vertex / 4 = source quad index
  return order;
}

describe("TransientTranslucent (Phase 4 piston transient)", () => {
  it("returns null when there are no transient quads", () => {
    const dev = new FakeGraphicsDevice();
    expect(new TransientTranslucent(dev).update([0, 0, 10], [])).toBeNull();
  });

  it("emits the moving block's quads ALONE as a depth-writing translucent draw (no terrain merge/suppress)", () => {
    const dev = new FakeGraphicsDevice();
    const merger = new TransientTranslucent(dev);
    const moving = [zQuad(5), zQuad(6)]; // two faces of the moving glass block

    const draw = merger.update([0.5, 0.5, 30], moving);
    expect(draw).not.toBeNull();
    expect(draw!.pass).toBe(TerrainPass.Translucent);
    expect(draw!.quadCount).toBe(2); // exactly the moving quads — nothing pulled from terrain
    expect(draw!.depthWrite).toBe(true); // vanilla `translucentMovingBlock` writes depth (the ordering mechanism)
    expect(draw!.sortType).toBe(SortType.StaticTopo); // pre-sorted ⇒ the renderer draws the index, does NOT re-sort
    expect(draw!.index).toBeDefined();
    // The API carries no `suppress` set at all — there is nothing to suppress.
    expect((draw as unknown as Record<string, unknown>).suppress).toBeUndefined();
  });

  it("orders the moving block's own faces back-to-front (farthest quad drawn first)", () => {
    const dev = new FakeGraphicsDevice();
    const merger = new TransientTranslucent(dev);
    // Camera at z=+30 looking toward −z. Of the block's faces, z=2 is FAR (drawn first), z=5 is NEAR.
    const draw = merger.update([0.5, 0.5, 30], [zQuad(5), zQuad(2)])!;
    const order = visitOrder(dev, draw);
    expect(order).toHaveLength(2);
    expect(order[0]).toBe(1); // far face (append index 1, z=2) drawn first
    expect(order[1]).toBe(0); // near face (append index 0, z=5) drawn last
  });

  it("round-trips the moving quad's world positions through the local-origin encoding", () => {
    const dev = new FakeGraphicsDevice();
    const merger = new TransientTranslucent(dev);
    const draw = merger.update([0.5, 0.5, 30], [zQuad(5)])!;
    const origin = draw.originBlocks;
    const bytes = dev.read(draw.vertex as never);
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const expected = zQuad(5);
    for (let k = 0; k < 4; k++) {
      const got = decodeVertex(dv, k * VERTEX_STRIDE_BYTES);
      expect(got.x + origin[0]).toBeCloseTo(expected[k].x, 2);
      expect(got.y + origin[1]).toBeCloseTo(expected[k].y, 2);
      expect(got.z + origin[2]).toBeCloseTo(expected[k].z, 2);
    }
  });
});
