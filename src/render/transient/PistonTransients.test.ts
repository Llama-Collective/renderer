// Moving-piston driver: state derivation (the 3 vanilla cases), interpolation offset, the 0.5 short/long
// head swap, the opaque-vs-translucent layer split, and the set() rebuild/drop lifecycle. PISTON_PLAN §2/§5.
// GPU-free (FakeGraphicsDevice); the geometry math is what's verified, not pixels.

import { describe, it, expect } from "vitest";
import { PistonTransients, signatureOf, statesOf, type MovingPistonInput } from "./PistonTransients";
import { FakeGraphicsDevice } from "../../core/testing/FakeGraphicsDevice";
import type { BakedBlockModel, BakedRenderQuad } from "../../mesh/model/BakedBlockModel";
import type { BlockProps } from "../../mesh/model/BlockStateResolver";
import { TerrainPass } from "../../types";

function info(over: Partial<MovingPistonInput> = {}): MovingPistonInput {
  return {
    x: 2, y: 1, z: 1,
    movedState: { name: "minecraft:stone", properties: {} },
    direction: "east",
    extending: true,
    isSourcePiston: false,
    progress: 1,
    progressO: 0,
    ...over,
  };
}

/** One unit face on the z=0 plane in the given render layer. */
function faceQuad(layer: TerrainPass): BakedRenderQuad {
  return {
    positions: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    atlasUV: [[0, 0], [1, 0], [1, 1], [0, 1]],
    normal: 2,
    cullface: null,
    layer,
    colorRGBA: 0xffffffff,
    material: layer === TerrainPass.Cutout ? 1 : 0,
  };
}

function model(...quads: BakedRenderQuad[]): BakedBlockModel {
  // Only `quads` is read by bakedModelToEntityVerts; occlusion/skipGroup are inert here.
  return { quads, occlusion: undefined as never, skipGroup: null };
}

/** A bakeModel spy that routes by name → layer and records every (name, props) it bakes. */
function spyBaker(layerFor: (name: string) => TerrainPass | null) {
  const calls: { name: string; props: BlockProps }[] = [];
  const bake = (name: string, props: BlockProps): BakedBlockModel | null => {
    calls.push({ name, props });
    const layer = layerFor(name);
    return layer === null ? null : model(faceQuad(layer));
  };
  return { bake, calls };
}

describe("signatureOf", () => {
  it("excludes progress/progressO but includes state/dir/extending/isSourcePiston", () => {
    const a = info({ progress: 0.5, progressO: 0 });
    const b = info({ progress: 1, progressO: 0.5 });
    expect(signatureOf(a)).toBe(signatureOf(b)); // same identity, different animation phase
    expect(signatureOf(info({ direction: "west" }))).not.toBe(signatureOf(a));
    expect(signatureOf(info({ isSourcePiston: true }))).not.toBe(signatureOf(a));
    expect(signatureOf(info({ movedState: { name: "minecraft:glass", properties: {} } }))).not.toBe(signatureOf(a));
  });
});

describe("statesOf (the 3 vanilla render cases)", () => {
  it("case A — moved piston_head: sliding head, no base, shortUntilHalf", () => {
    const p = statesOf(info({ movedState: { name: "minecraft:piston_head", properties: { type: "sticky" } } }));
    expect(p.headMode).toBe("shortUntilHalf");
    expect(p.base).toBeNull();
    expect(p.movingShort).toBe("minecraft:piston_head[facing=east,short=true,type=sticky]");
    expect(p.movingLong).toBe("minecraft:piston_head[facing=east,short=false,type=sticky]");
  });

  it("case B — source piston retracting: sliding head + STATIC extended base, shortFromHalf", () => {
    const p = statesOf(info({ movedState: { name: "minecraft:sticky_piston", properties: {} }, extending: false, isSourcePiston: true }));
    expect(p.headMode).toBe("shortFromHalf");
    expect(p.movingShort).toBe("minecraft:piston_head[facing=east,short=true,type=sticky]");
    expect(p.movingLong).toBe("minecraft:piston_head[facing=east,short=false,type=sticky]");
    expect(p.base).toBe("minecraft:sticky_piston[facing=east,extended=true]");
  });

  it("case C — any other block: the moved block slides as one piece, no head, no base", () => {
    const p = statesOf(info({ movedState: { name: "minecraft:oak_log", properties: { axis: "x" } } }));
    expect(p.headMode).toBe("none");
    expect(p.base).toBeNull();
    expect(p.moving).toBe("minecraft:oak_log[axis=x]");
  });
});

describe("PistonTransients.frame — interpolation + slide offset", () => {
  it("slides the moved block from one cell back toward the BE cell as progress 0→1 (extending east)", () => {
    const dev = new FakeGraphicsDevice();
    const { bake } = spyBaker(() => TerrainPass.Solid);
    const pt = new PistonTransients(dev, bake);
    pt.set([info({ progress: 1, progressO: 0 })]); // moving_piston BE at x=2

    // partial 0 → progress 0 → extended -1 → one cell back (x=1)
    let f = pt.frame([0, 0, 0], 0);
    expect(f.opaqueDraws).toHaveLength(1);
    expect(f.opaqueDraws[0].originBlocks[0]).toBeCloseTo(1, 6);
    expect(f.translucent).toBeNull();

    // partial 1 → progress 1 → extended 0 → at the BE cell (x=2)
    f = pt.frame([0, 0, 0], 1);
    expect(f.opaqueDraws[0].originBlocks[0]).toBeCloseTo(2, 6);

    // half-tick → progress 0.5 → extended -0.5 → x=1.5
    f = pt.frame([0, 0, 0], 0.5);
    expect(f.opaqueDraws[0].originBlocks[0]).toBeCloseTo(1.5, 6);
    // y/z unaffected (east is the x-axis)
    expect(f.opaqueDraws[0].originBlocks[1]).toBeCloseTo(1, 6);
    expect(f.opaqueDraws[0].originBlocks[2]).toBeCloseTo(1, 6);
  });
});

describe("PistonTransients.frame — layer split (D11)", () => {
  it("opaque blocks → terrain draws only; translucent blocks → the local sort only", () => {
    const dev = new FakeGraphicsDevice();
    const { bake } = spyBaker((name) => (name === "minecraft:glass" ? TerrainPass.Translucent : TerrainPass.Solid));
    const pt = new PistonTransients(dev, bake);

    pt.set([info({ movedState: { name: "minecraft:stone", properties: {} } })]);
    let f = pt.frame([0, 0, 8], 0.5);
    expect(f.opaqueDraws.length).toBeGreaterThan(0);
    expect(f.opaqueDraws[0].pass).toBe(TerrainPass.Solid);
    expect(f.translucent).toBeNull();

    pt.set([info({ movedState: { name: "minecraft:glass", properties: {} } })]);
    f = pt.frame([0, 0, 8], 0.5);
    expect(f.opaqueDraws).toHaveLength(0);
    expect(f.translucent).not.toBeNull();
    expect(f.translucent!.pass).toBe(TerrainPass.Translucent);
  });
});

describe("PistonTransients.frame — head short/long swap at 0.5", () => {
  it("case A bakes the SHORT head at progress ≤ 0.5 and the LONG head after", () => {
    const dev = new FakeGraphicsDevice();
    const { bake, calls } = spyBaker(() => TerrainPass.Solid);
    const pt = new PistonTransients(dev, bake);
    const head = { name: "minecraft:piston_head", properties: { type: "normal" } };

    pt.set([info({ movedState: head, progress: 0.25, progressO: 0.25 })]); // interpolated 0.25 at any partial
    pt.frame([0, 0, 0], 1);
    expect(calls.some((c) => c.name === "minecraft:piston_head" && c.props.short === "true")).toBe(true);
    expect(calls.some((c) => c.name === "minecraft:piston_head" && c.props.short === "false")).toBe(false);

    calls.length = 0;
    pt.set([info({ movedState: head, progress: 0.75, progressO: 0.75 })]);
    pt.frame([0, 0, 0], 1);
    expect(calls.some((c) => c.name === "minecraft:piston_head" && c.props.short === "false")).toBe(true);
  });
});

describe("PistonTransients.frame — case B base placement", () => {
  it("draws the head at the slid origin and the EXTENDED base at the cell with NO offset", () => {
    const dev = new FakeGraphicsDevice();
    const { bake } = spyBaker(() => TerrainPass.Solid);
    const pt = new PistonTransients(dev, bake);
    // retracting source piston at x=2; progress 0.5 → extended (1-0.5)=0.5 → head offset +0.5 on east
    pt.set([info({ movedState: { name: "minecraft:piston", properties: {} }, extending: false, isSourcePiston: true, progress: 0.5, progressO: 0.5 })]);
    const f = pt.frame([0, 0, 0], 1);
    expect(f.opaqueDraws).toHaveLength(2);
    const xs = f.opaqueDraws.map((d) => d.originBlocks[0]);
    expect(xs).toContain(2);            // the static extended base, no offset
    expect(xs.some((x) => Math.abs(x - 2.5) < 1e-6)).toBe(true); // the sliding head, +0.5
  });
});

describe("PistonTransients — set() lifecycle", () => {
  it("active() tracks the entry set; an absent cell is dropped", () => {
    const dev = new FakeGraphicsDevice();
    const { bake } = spyBaker(() => TerrainPass.Solid);
    const pt = new PistonTransients(dev, bake);
    expect(pt.active()).toBe(false);
    pt.set([info()]);
    expect(pt.active()).toBe(true);
    pt.set([]); // piston finished → sim drops it → transient cleared
    expect(pt.active()).toBe(false);
  });

  it("a same-signature update changes the interpolated offset without a rebuild", () => {
    const dev = new FakeGraphicsDevice();
    const { bake } = spyBaker(() => TerrainPass.Solid);
    const pt = new PistonTransients(dev, bake);
    pt.set([info({ progress: 0.5, progressO: 0 })]);
    const before = pt.frame([0, 0, 0], 1).opaqueDraws[0].originBlocks[0];
    pt.set([info({ progress: 1, progressO: 0.5 })]); // same signature, advanced phase
    const after = pt.frame([0, 0, 0], 1).opaqueDraws[0].originBlocks[0];
    expect(after).toBeGreaterThan(before); // slid further toward the BE cell (x=2)
  });
});
