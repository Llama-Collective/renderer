import { describe, expect, it } from "vitest";
import { FakeGraphicsDevice } from "../../core/testing/FakeGraphicsDevice";
import { Direction, TerrainPass } from "../../types";
import type { BakedBlockModel, BakedRenderQuad } from "../../mesh/model/BakedBlockModel";
import { computeOcclusion } from "../../mesh/model/OcclusionShape";
import { EntityModelFactory } from "./EntityModelFactory";
import { parseBlockState, bakedModelToEntityVerts } from "./blockDisplay";
import { mul, rotationY, rotationZ, scaling, translation } from "../../mesh/entity/mat4";
import type { RenderEntity } from "./EntityScene";
import type { EntityDraw } from "./EntityRenderer";

const ent = (type: string, position: [number, number, number], properties: Record<string, string> = {}): RenderEntity => ({
  id: "e",
  type,
  position,
  velocity: [0, 0, 0],
  properties,
});

/** A4: buildDraws now emits into a caller array + returns a count; collect into a fresh array for asserts. */
function drawsOf(factory: EntityModelFactory, e: RenderEntity, clock = 0): EntityDraw[] {
  const out: EntityDraw[] = [];
  factory.buildDraws(e, clock, out);
  return out;
}

function solidQuad(): BakedRenderQuad {
  return {
    positions: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    atlasUV: [[0, 0], [1, 0], [1, 1], [0, 1]],
    normal: Direction.North,
    cullface: null,
    layer: TerrainPass.Solid,
    colorRGBA: 0xffffffff,
    material: 0,
  };
}

function model(layer: TerrainPass = TerrainPass.Solid): BakedBlockModel {
  const q = { ...solidQuad(), layer };
  return { quads: [q, q], occlusion: computeOcclusion([], () => true), skipGroup: null };
}

describe("parseBlockState", () => {
  it("parses a bare name", () => {
    expect(parseBlockState("minecraft:sand")).toEqual({ name: "minecraft:sand", props: {} });
  });
  it("parses name + properties", () => {
    expect(parseBlockState("minecraft:oak_log[axis=y,foo=bar]")).toEqual({
      name: "minecraft:oak_log",
      props: { axis: "y", foo: "bar" },
    });
  });
});

describe("bakedModelToEntityVerts", () => {
  it("buckets quads by pass with all faces (4 verts/quad)", () => {
    const v = bakedModelToEntityVerts(model(TerrainPass.Solid));
    expect(v[TerrainPass.Solid]!.length).toBe(8); // 2 quads × 4
    expect(v[TerrainPass.Translucent]).toBeUndefined();
  });
});

describe("EntityModelFactory", () => {
  const make = (bakeBlock: (n: string) => BakedBlockModel | null) => {
    const device = new FakeGraphicsDevice();
    const blockAtlas = device.createTexture({ width: 1, height: 1, format: 0 as never, label: "atlas" });
    const factory = new EntityModelFactory({ device, blockAtlas, bakeBlock: (n) => bakeBlock(n) });
    return { device, factory, blockAtlas };
  };

  it("falling_block: bakes its block and places [0,1] model centered on x,z at the entity y", () => {
    const { factory } = make(() => model());
    const draws = drawsOf(factory, ent("minecraft:falling_block", [10, 5, 3], { block: "minecraft:sand" }), 0);
    expect(draws.length).toBe(1);
    const m = draws[0].model;
    // translation column = (pos.x-0.5, pos.y, pos.z-0.5).
    expect([m[12], m[13], m[14]]).toEqual([9.5, 5, 2.5]);
    expect(draws[0].texture).toBeDefined();
  });

  it("caches geometry: a second falling_block of the same block reuses buffers", () => {
    const { device, factory } = make(() => model());
    drawsOf(factory, ent("minecraft:falling_block", [0, 0, 0], { block: "minecraft:sand" }), 0);
    const after1 = device.liveBufferCount();
    drawsOf(factory, ent("minecraft:falling_block", [5, 0, 0], { block: "minecraft:sand" }), 0);
    expect(device.liveBufferCount()).toBe(after1); // no new GPU buffers for the same variant
  });

  it("A3 byte-identity: primed_tnt pulse-scale matrix equals the allocating reference", () => {
    const { factory } = make(() => model());
    const fuse = 5; // f = clamp(1-5/10)=0.5, s = 1 + 0.5^4·0.3 = 1.01875 → a non-trivial scaleAboutCenter
    const f = Math.max(0, Math.min(1, 1 - fuse / 10));
    const s = 1 + f * f * f * f * 0.3;
    const want = mul(translation(2 - 0.5, 3, 4 - 0.5), mul(translation(0.5, 0.5, 0.5), scaling(s, s, s), translation(-0.5, -0.5, -0.5)));
    const draws = drawsOf(factory, ent("minecraft:tnt", [2, 3, 4], { fuse: String(fuse) }), 0);
    expect(Array.from(draws[0].model)).toEqual(Array.from(want)); // arena path === allocating path, bit-for-bit
  });

  it("A3 byte-identity: minecart contents 7-deep model matrix equals the allocating reference", () => {
    const { factory } = make(() => model());
    // hopper_minecart, no box-model source ⇒ only the block-display CONTENTS draw, off = 1.
    const off = 1, yaw = 0, pitch = 0;
    const want = mul(
      translation(2, 3, 4), translation(0, 0.375, 0),
      rotationY(((180 - yaw) * Math.PI) / 180), rotationZ((-pitch * Math.PI) / 180),
      scaling(0.75, 0.75, 0.75), translation(-0.5, (off - 8) / 16, 0.5), rotationY(Math.PI / 2),
    );
    const draws = drawsOf(factory, ent("minecraft:hopper_minecart", [2, 3, 4]), 0);
    expect(draws.length).toBe(1);
    expect(Array.from(draws[0].model)).toEqual(Array.from(want));
  });

  it("primed_tnt: blinks (flash on) at a fuse that is an even 5-tick bucket", () => {
    const { factory } = make(() => model());
    const blink = drawsOf(factory, ent("minecraft:tnt", [0, 0, 0], { fuse: "80" }), 0); // floor(80/5)=16 even
    expect(blink[0].flash![3]).toBeGreaterThan(0);
    const dark = drawsOf(factory, ent("minecraft:tnt", [0, 0, 0], { fuse: "5" }), 0); // floor(5/5)=1 odd
    expect(dark[0].flash![3]).toBe(0);
  });

  it("hopper_minecart: bakes the contained block via the terrain baker (block-display contents)", () => {
    const seen: string[] = [];
    const { factory } = make((n) => {
      seen.push(n);
      return model();
    });
    const draws = drawsOf(factory, ent("minecraft:hopper_minecart", [0, 0, 0]), 0);
    expect(seen).toContain("minecraft:hopper");
    expect(draws.length).toBeGreaterThan(0);
  });

  it("chest_minecart: does NOT bake the chest via the terrain baker (it's the BE chest box model)", () => {
    // A chest is a block entity with no terrain model, so the box-model source renders it — the terrain
    // baker must never be asked for minecraft:chest (else it bakes empty geometry / a wrong cube).
    const seen: string[] = [];
    const { factory } = make((n) => {
      seen.push(n);
      return model();
    });
    drawsOf(factory, ent("minecraft:chest_minecart", [0, 0, 0]), 0);
    expect(seen).not.toContain("minecraft:chest");
  });

  it("unknown block-display block yields no draw (no crash)", () => {
    const { factory } = make(() => null);
    expect(drawsOf(factory, ent("minecraft:falling_block", [0, 0, 0], { block: "minecraft:unknownium" }), 0)).toEqual([]);
  });
});
