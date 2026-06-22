import { describe, expect, it } from "vitest";
import type { SpriteUv } from "../../../types";
import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";
import { bakeBlockEntity, blockEntitySpecials, hasBlockEntityModel, registeredBETypes } from "./index";
import { transformPoint } from "../../../mesh/entity/mat4";

const FULL: SpriteUv = { u: 0, v: 0, width: 1, height: 1 };
const ctx = { uvFor: () => FULL, sectionOrigin: [0, 0, 0] as [number, number, number] };
const rec = (type: string, props: Record<string, string> = {}): BlockEntityRecord => ({ x: 0, y: 0, z: 0, type, props, animating: false });

describe("block-entity registry coverage", () => {
  const EXPECTED = [
    "chest", "trapped_chest", "ender_chest", "copper_chest",
    "red_shulker_box", "shulker_box", "red_bed",
    "skeleton_skull", "creeper_head", "player_head",
    "bell", "conduit", "decorated_pot",
    "oak_sign", "oak_wall_sign", "oak_hanging_sign",
    "white_banner", "red_banner",
    "copper_golem_statue", "enchanting_table", "lectern",
    "beacon", "spawner", "trial_spawner", "vault", "suspicious_sand",
    "campfire", "oak_shelf", "spruce_shelf", "end_portal", "end_gateway", "structure_block",
    "zombie_head", "wither_skeleton_wall_skull", "piglin_head",
  ];

  it("registers every expected block-entity type", () => {
    for (const t of EXPECTED) expect(hasBlockEntityModel(t), t).toBe(true);
  });

  it("registers a large set of types (variants expand)", () => {
    expect(registeredBETypes().length).toBeGreaterThan(80); // 16 dyes × bed/shulker/banner + woods × signs + …
  });

  it("bakes non-empty box geometry for the box-model BEs", () => {
    const boxes: [string, Record<string, string>][] = [
      ["chest", { facing: "south" }],
      ["red_shulker_box", { facing: "up" }],
      ["red_bed", { part: "head", facing: "north" }],
      ["skeleton_skull", { rotation: "0" }],
      ["bell", {}],
      ["conduit", {}],
      ["decorated_pot", { facing: "south" }],
      ["oak_sign", { rotation: "0" }],
      ["white_banner", { rotation: "0" }],
      ["copper_golem_statue", { facing: "south" }],
      ["enchanting_table", {}],
      ["lectern", { facing: "south" }],
    ];
    for (const [type, props] of boxes) {
      const v = bakeBlockEntity(rec(type, props), ctx, 0, false);
      expect(v, type).not.toBeNull();
      expect(v!.length, type).toBeGreaterThan(0);
    }
  });

  it("beacon emits a beam special; spawner a block, vault an item; end portal a portal; structure a box", () => {
    expect(blockEntitySpecials(rec("beacon", { color: "white" }), 0)[0].kind).toBe("beam");
    expect(blockEntitySpecials(rec("spawner"), 0)[0].kind).toBe("blockModel"); // spawned ENTITY → block placeholder
    expect(blockEntitySpecials(rec("vault"), 0)[0].kind).toBe("item"); // reward ITEM → real item renderer
    expect(blockEntitySpecials(rec("end_portal"), 0)[0].kind).toBe("portal");
    expect(blockEntitySpecials(rec("structure_block"), 0)[0].kind).toBe("lineBox");
    // Campfire cooking items render only when present (empty campfire = just its block model).
    expect(blockEntitySpecials(rec("campfire"), 0).length).toBe(0);
    expect(blockEntitySpecials(rec("campfire", { items: "minecraft:bread,minecraft:cod" }), 0).length).toBe(2);
  });

  it("lays campfire cooking items flat at the fire surface (vanilla CampfireRenderer, AUDIT M6)", () => {
    const draws = blockEntitySpecials(rec("campfire", { facing: "south", items: "minecraft:bread" }), 0);
    expect(draws).toHaveLength(1);
    const d = draws[0];
    if (d.kind !== "item") throw new Error(`expected an item draw, got ${d.kind}`);
    // The item geometry is [0,1]³; its centre maps onto the campfire surface (vanilla y 0.44921875), not
    // the old y 0.5 — confirming both the corrected Y constant and the rotateX(90) flat-lay chain.
    const c = transformPoint(d.model, 0.5, 0.5, 0.5);
    expect(c[1]).toBeCloseTo(0.44921875, 5);
  });

  it("shelf renders up to 3 item slots (vanilla NonNullList.withSize(3))", () => {
    expect(blockEntitySpecials(rec("oak_shelf", { facing: "south" }), 0).length).toBe(0); // empty shelf
    expect(blockEntitySpecials(rec("oak_shelf", { facing: "south", items: "a,b,c,d" }), 0).length).toBe(3); // capped at 3
  });

  it("centers a floor skull on the block (vanilla SkullBlockRenderer, no stray translate)", () => {
    const v = bakeBlockEntity(rec("skeleton_skull", { rotation: "0" }), ctx, 0, false)!;
    const xs = v.map((q) => q.x), zs = v.map((q) => q.z);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cz = (Math.min(...zs) + Math.max(...zs)) / 2;
    expect(cx).toBeCloseTo(0.5, 1); // centred on the block, not at the corner
    expect(cz).toBeCloseTo(0.5, 1);
    expect(Math.min(...xs)).toBeGreaterThan(0.2); // 8px head → ~[0.25, 0.75]
    expect(Math.max(...xs)).toBeLessThan(0.8);
  });

  it("lays the bed FLAT for every facing, not floating upright (AUDIT H4)", () => {
    // Vanilla BedRenderer.modelTransform = T·Rx(90)·Rzc. The earlier T·Rzc·Rx order floated the bed to
    // y≈1.5–1.9 (standing upright) for 3 of 4 facings (only north coincidentally matched). A flat bed's
    // whole model stays below 1 block (head slab sits at bed height ~0.5625; legs hang down to the floor).
    for (const facing of ["south", "west", "north", "east"]) {
      const v = bakeBlockEntity(rec("red_bed", { part: "head", facing }), ctx, 0, false)!;
      expect(v, facing).not.toBeNull();
      const ys = v.map((q) => q.y);
      expect(Math.max(...ys), `${facing} maxY (flat, not floating)`).toBeLessThan(1.0);
      expect(Math.min(...ys), `${facing} minY (on/above floor)`).toBeGreaterThan(-0.3);
    }
  });

  it("copper golem statue stands with its feet at the block base (not floating)", () => {
    const v = bakeBlockEntity(rec("copper_golem_statue", { facing: "south" }), ctx, 0, false)!;
    const ys = v.map((q) => q.y);
    expect(Math.min(...ys)).toBeGreaterThanOrEqual(-0.1);
    expect(Math.min(...ys)).toBeLessThan(0.2); // feet at/near y=0, not floating ~1 block up
    expect(Math.max(...ys)).toBeLessThan(2);
  });
});
