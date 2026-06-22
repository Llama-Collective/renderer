// TransientOpaque: bake-once caching, per-pass draws at a fractional origin, and clear(). PISTON_PLAN §5 D1.
// GPU-free (FakeGraphicsDevice).

import { describe, it, expect } from "vitest";
import { TransientOpaque } from "./TransientOpaque";
import { FakeGraphicsDevice } from "../../core/testing/FakeGraphicsDevice";
import { TerrainPass, type Vec3 } from "../../types";
import type { RawVertex } from "../../mesh/VertexFormat";

function quadVerts(): RawVertex[] {
  const v = (x: number, y: number): RawVertex => ({ x, y, z: 0, u: x, v: y, normal: 2, colorRGBA: 0xffffffff, material: 0 });
  return [v(0, 0), v(1, 0), v(1, 1), v(0, 1)];
}

describe("TransientOpaque", () => {
  it("emits one Solid + one Cutout draw at the given origin, with correct quad counts", () => {
    const dev = new FakeGraphicsDevice();
    const op = new TransientOpaque(dev);
    op.ensure("k", quadVerts(), [...quadVerts(), ...quadVerts()]); // 1 solid quad, 2 cutout quads
    const origin: Vec3 = [3.5, 1, -2];
    const draws = op.draws("k", origin);
    expect(draws).toHaveLength(2);
    const solid = draws.find((d) => d.pass === TerrainPass.Solid)!;
    const cutout = draws.find((d) => d.pass === TerrainPass.Cutout)!;
    expect(solid.quadCount).toBe(1);
    expect(cutout.quadCount).toBe(2);
    expect(solid.originBlocks).toEqual(origin); // the fractional slide origin rides the draw, not the verts
  });

  it("bakes a key ONCE — a second ensure() does not create new buffers", () => {
    const dev = new FakeGraphicsDevice();
    let created = 0;
    const orig = dev.createBuffer.bind(dev);
    dev.createBuffer = (desc) => {
      created++;
      return orig(desc);
    };
    const op = new TransientOpaque(dev);
    op.ensure("k", quadVerts(), undefined);
    const afterFirst = created;
    op.ensure("k", quadVerts(), undefined); // idempotent — early return on cache hit
    expect(created).toBe(afterFirst);
    expect(op.has("k")).toBe(true);
  });

  it("a key with no opaque geometry caches as empty (no draws, but has() is true)", () => {
    const dev = new FakeGraphicsDevice();
    const op = new TransientOpaque(dev);
    op.ensure("empty", undefined, undefined);
    expect(op.has("empty")).toBe(true);
    expect(op.draws("empty", [0, 0, 0])).toHaveLength(0);
  });

  it("clear() drops the cache so the key must be re-baked", () => {
    const dev = new FakeGraphicsDevice();
    const op = new TransientOpaque(dev);
    op.ensure("k", quadVerts(), undefined);
    expect(op.has("k")).toBe(true);
    op.clear();
    expect(op.has("k")).toBe(false);
    expect(op.draws("k", [0, 0, 0])).toHaveLength(0); // not re-baked → nothing
  });
});
