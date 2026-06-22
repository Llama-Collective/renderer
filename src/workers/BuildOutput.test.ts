// serializeBuildOutput ⇄ makeBuildOutput round-trip — the worker→main wire shape. Phase 5.

import { describe, it, expect } from "vitest";
import { makeBuildOutput, serializeBuildOutput } from "./BuildOutput";
import { SortType } from "../mesh/SortTypes";
import { sectionKey } from "../world/SectionKey";
import type { TQuad } from "../mesh/TranslucentCollector";

const KEY = sectionKey(0, 0, 0);
const tquad = (): TQuad => ({
  extents: new Float32Array([1, 1, 1, 0, 0, 0]),
  facing: 6,
  dot: 0.5,
  normal: [0, 0, 1],
  centroid: [0.5, 0.5, 0.5],
  positions: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
});

function build(sortType: SortType) {
  return makeBuildOutput({
    sectionKey: KEY,
    generation: 1,
    info: { flags: 0 },
    parts: {},
    translucent: { vertexData: new ArrayBuffer(80), quadCount: 1, sortType, quadHash: 7, indexData: new Uint32Array([0, 1, 2, 0, 2, 3]).buffer, quads: [tquad()] },
  });
}

describe("serializeBuildOutput — translucent quads for off-thread dynamic re-sort", () => {
  it("carries quads for DYNAMIC sections so the worker path re-sorts like inline (TRAP 12.A)", () => {
    const { payload } = serializeBuildOutput(build(SortType.Dynamic));
    const out = makeBuildOutput(payload);
    expect(out.translucent?.quads?.length).toBe(1);
    expect(out.translucent?.quads?.[0].facing).toBe(6); // survived structured-clone
  });

  it("omits quads for STATIC sort types (they bake a fixed index and never re-sort)", () => {
    for (const st of [SortType.AnyOrder, SortType.StaticNormal, SortType.StaticTopo]) {
      const { payload } = serializeBuildOutput(build(st));
      expect(makeBuildOutput(payload).translucent?.quads).toBeUndefined();
    }
  });
});
