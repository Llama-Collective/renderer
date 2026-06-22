// Parent-chain merge + #ref resolution + blockstate/multipart resolution, against REAL pack JSON
// extracted from public/pack.zip. RENDERER_PLAN §24.3–§24.4, §22.

import { describe, it, expect } from "vitest";
import fixtures from "./__fixtures__/blocks.json";
import { resolveModel, type RawModelProvider } from "./ModelResolver";
import { resolveBlockState } from "./BlockStateResolver";
import type { RawBlockModel, RawBlockState } from "./ModelTypes";

// JSON infers arrays as `number[]`, not our `[number,number,number]` tuples → cast via unknown.
const models = fixtures.models as unknown as Record<string, RawBlockModel>;
const blockstates = fixtures.blockstates as unknown as Record<string, RawBlockState>;
const provider: RawModelProvider = { getModel: (id) => models[id] };

describe("resolveModel (parent merge + #ref)", () => {
  it("oak_stairs inherits elements from block/stairs and resolves textures", () => {
    const m = resolveModel("block/oak_stairs", provider);
    expect(m.elements.length).toBe(2); // bottom slab + back step, from block/stairs
    expect(m.textures.side.sprite).toBe("block/oak_planks");
    expect(m.textures.top.sprite).toBe("block/oak_planks");
    expect(m.ambientocclusion).toBe(true);
  });

  it("grass_block has two elements and grass-tinted slots", () => {
    const m = resolveModel("block/grass_block", provider);
    expect(m.elements.length).toBe(2); // cube + side-overlay
    expect(m.textures.top.sprite).toBe("block/grass_block_top");
    expect(m.textures.overlay.sprite).toBe("block/grass_block_side_overlay");
  });

  it("glass_pane_side resolves the {sprite, force_translucent} object form", () => {
    const m = resolveModel("block/glass_pane_side", provider);
    expect(m.textures.pane.sprite).toBe("block/glass");
    expect(m.textures.pane.forceTranslucent).toBe(true);
    expect(m.textures.edge.forceTranslucent).toBe(true);
    expect(m.elements.length).toBe(1); // from template_glass_pane_side
    expect(m.ambientocclusion).toBe(false); // template sets it false
  });

  it("oak_log resolves #end/#side across the cube_column → cube → block chain", () => {
    const m = resolveModel("block/oak_log", provider);
    expect(m.textures.end.sprite).toBe("block/oak_log_top");
    expect(m.textures.side.sprite).toBe("block/oak_log");
    expect(m.elements.length).toBe(1); // full cube from block/cube
  });

  it("missing model resolves to empty, not a throw", () => {
    expect(resolveModel("block/does_not_exist", provider).elements).toEqual([]);
  });
});

describe("resolveBlockState (variants)", () => {
  it("picks a single stairs variant for a concrete state", () => {
    const parts = resolveBlockState(blockstates.oak_stairs, { facing: "east", half: "bottom", shape: "straight" });
    expect(parts).toHaveLength(1);
    expect(parts[0].model).toContain("oak_stairs");
  });

  it("applies axis rotation for logs", () => {
    expect(resolveBlockState(blockstates.oak_log, { axis: "x" })[0]).toMatchObject({ model: "block/oak_log_horizontal", x: 90, y: 90 });
    expect(resolveBlockState(blockstates.oak_log, { axis: "y" })[0]).toMatchObject({ model: "block/oak_log", x: 0, y: 0 });
  });

  it("picks index 0 of a weighted variant list (grass_block)", () => {
    const parts = resolveBlockState(blockstates.grass_block, { snowy: "false" });
    expect(parts).toHaveLength(1);
    expect(parts[0]).toMatchObject({ model: "block/grass_block", y: 0 });
  });
});

describe("resolveBlockState (multipart)", () => {
  it("oak_fence: post only when unconnected, post+sides per connection", () => {
    const none = { north: "false", east: "false", south: "false", west: "false" };
    expect(resolveBlockState(blockstates.oak_fence, none)).toHaveLength(1); // post
    expect(resolveBlockState(blockstates.oak_fence, { ...none, north: "true" })).toHaveLength(2);
    expect(resolveBlockState(blockstates.oak_fence, { ...none, north: "true", east: "true" })).toHaveLength(3);
  });

  it("glass_pane multipart selects connected sides", () => {
    const all = { north: "true", east: "true", south: "true", west: "true" };
    // post + 4 sides
    expect(resolveBlockState(blockstates.glass_pane, all).length).toBeGreaterThanOrEqual(5);
  });
});

describe("when grammar (OR / AND / negation / pipe)", () => {
  const synthetic: RawBlockState = {
    multipart: [
      { when: { OR: [{ a: "1" }, { b: "2" }] }, apply: { model: "block/x" } },
      { when: { AND: [{ a: "1" }, { b: "2" }] }, apply: { model: "block/y" } },
      { when: { c: "side|up" }, apply: { model: "block/z" } },
      { when: { d: "!none" }, apply: { model: "block/w" } },
    ],
  };
  const count = (props: Record<string, string>) => resolveBlockState(synthetic, props).length;

  it("OR matches if any sub passes", () => expect(count({ a: "1", b: "0", c: "x", d: "none" })).toBe(1));
  it("AND needs all subs", () => expect(count({ a: "1", b: "2", c: "x", d: "none" })).toBe(2)); // OR + AND
  it("pipe is OR-of-values", () => expect(count({ a: "0", b: "0", c: "up", d: "none" })).toBe(1));
  it("negation excludes a value", () => expect(count({ a: "0", b: "0", c: "x", d: "low" })).toBe(1));
});
