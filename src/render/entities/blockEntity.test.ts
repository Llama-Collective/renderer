import { describe, expect, it } from "vitest";
import { FakeGraphicsDevice } from "../../core/testing/FakeGraphicsDevice";
import { BlockEntityIndex, type BlockEntityRecord } from "../../world/BlockEntityIndex";
import { sectionOfBlock } from "../../world/SectionKey";
import type { SpriteUv } from "../../types";
import { bakeBlockEntity, splitSectionBEs, blockEntityAnimatesPerFrame, hasBlockEntityModel } from "./blockentities";
import { BlockEntitySceneBaker } from "./BlockEntitySceneBaker";

const FULL: SpriteUv = { u: 0, v: 0, width: 1, height: 1 };
const uvFor = () => FULL;
const ctx = (origin: [number, number, number] = [0, 0, 0]) => ({ uvFor, sectionOrigin: origin });
const chest = (x: number, y: number, z: number, props: Record<string, string> = {}, animating = false): BlockEntityRecord => ({ x, y, z, type: "minecraft:chest", props, animating });

describe("bakeBlockEntity (chest, registry-backed, per-FACING)", () => {
  it("bakes non-empty section-local geometry", () => {
    const v = bakeBlockEntity(chest(3, 0, 5), ctx(), 0, false);
    expect(v).not.toBeNull();
    expect(v!.length).toBeGreaterThan(0);
  });

  it("places the BE at its section-local position", () => {
    const v = bakeBlockEntity(chest(3, 0, 5), ctx([0, 0, 0]), 0, false)!;
    const xs = v.map((q) => q.x);
    expect(Math.min(...xs)).toBeGreaterThanOrEqual(2.9);
    expect(Math.max(...xs)).toBeLessThanOrEqual(4.1);
  });

  it("FACING rotates the geometry (north vs east differ)", () => {
    const north = bakeBlockEntity(chest(0, 0, 0, { facing: "north" }), ctx(), 0, false)!;
    const east = bakeBlockEntity(chest(0, 0, 0, { facing: "east" }), ctx(), 0, false)!;
    let differs = false;
    for (let i = 0; i < north.length; i++) if (Math.abs(north[i].x - east[i].x) > 1e-3) differs = true;
    expect(differs).toBe(true);
  });

  it("openness lifts the lid (open geometry differs from the shut rest pose)", () => {
    const rest = bakeBlockEntity(chest(0, 0, 0), ctx(), 0.4, false)!;
    const open = bakeBlockEntity(chest(0, 0, 0), { ...ctx(), openness: 1 }, 0.4, true)!;
    let differs = false;
    for (let i = 0; i < rest.length; i++) if (Math.abs(rest[i].z - open[i].z) > 1e-3 || Math.abs(rest[i].y - open[i].y) > 1e-3) differs = true;
    expect(differs).toBe(true);
  });

  it("a closed (openness 0) bake matches the rest pose — the lid is shut for the static section bake", () => {
    const rest = bakeBlockEntity(chest(0, 0, 0), ctx(), 0, false)!;
    const shut = bakeBlockEntity(chest(0, 0, 0), { ...ctx(), openness: 0 }, 0, true)!;
    for (let i = 0; i < rest.length; i++) {
      expect(Math.abs(rest[i].y - shut[i].y)).toBeLessThan(1e-4);
      expect(Math.abs(rest[i].z - shut[i].z)).toBeLessThan(1e-4);
    }
  });

  it("unknown BE type has no model", () => {
    expect(hasBlockEntityModel("minecraft:furnace")).toBe(false);
    expect(bakeBlockEntity({ x: 0, y: 0, z: 0, type: "minecraft:furnace", props: {}, animating: false }, ctx(), 0, false)).toBeNull();
  });
});

describe("splitSectionBEs (hybrid classifier)", () => {
  it("idle → static set, animating → per-frame set; never both", () => {
    const { idle, animating } = splitSectionBEs([chest(0, 0, 0, {}, false), chest(1, 0, 0, {}, true)]);
    expect(idle.length).toBe(1);
    expect(animating.length).toBe(1);
    expect(idle[0]).not.toBe(animating[0]);
  });
  it("ignores BEs with no registered model", () => {
    const { idle } = splitSectionBEs([{ x: 0, y: 0, z: 0, type: "minecraft:furnace", props: {}, animating: false }]);
    expect(idle.length).toBe(0);
  });
  it("a shulker box is registered and classified", () => {
    expect(hasBlockEntityModel("red_shulker_box")).toBe(true);
    const { idle } = splitSectionBEs([{ x: 0, y: 0, z: 0, type: "red_shulker_box", props: { facing: "up" }, animating: false }]);
    expect(idle.length).toBe(1);
  });
  it("idle-loop BEs (banner / conduit) go to the per-frame set EVEN when not toggled (BBE §5)", () => {
    const banner: BlockEntityRecord = { x: 0, y: 0, z: 0, type: "white_banner", props: {}, animating: false };
    const conduit: BlockEntityRecord = { x: 1, y: 0, z: 0, type: "conduit", props: {}, animating: false };
    const { idle, animating } = splitSectionBEs([banner, conduit]);
    expect(idle.length).toBe(0);
    expect(animating.length).toBe(2);
  });
  it("an event-driven chest stays idle (static) when its record is not toggled", () => {
    const { idle, animating } = splitSectionBEs([chest(0, 0, 0, {}, false)]);
    expect(idle.length).toBe(1);
    expect(animating.length).toBe(0);
  });
});

describe("blockEntityAnimatesPerFrame (render-gate predicate)", () => {
  const r = (type: string, animating = false): BlockEntityRecord => ({ x: 0, y: 0, z: 0, type, props: {}, animating });
  it("idle-loop box BEs always animate, regardless of the toggle", () => {
    expect(blockEntityAnimatesPerFrame(r("white_banner"))).toBe(true);
    expect(blockEntityAnimatesPerFrame(r("conduit"))).toBe(true);
  });
  it("event-driven box BEs animate only when toggled", () => {
    expect(blockEntityAnimatesPerFrame(r("minecraft:chest", false))).toBe(false);
    expect(blockEntityAnimatesPerFrame(r("minecraft:chest", true))).toBe(true);
    expect(blockEntityAnimatesPerFrame(r("red_shulker_box", false))).toBe(false);
  });
  it("always-animated special-only BEs animate; static specials and unknowns do not", () => {
    expect(blockEntityAnimatesPerFrame(r("spawner"))).toBe(true); // spinning embedded mob
    expect(blockEntityAnimatesPerFrame(r("vault"))).toBe(true); // spinning reward item
    expect(blockEntityAnimatesPerFrame(r("end_portal"))).toBe(true); // animated portal shader
    expect(blockEntityAnimatesPerFrame(r("suspicious_sand"))).toBe(false); // static brushable item
    expect(blockEntityAnimatesPerFrame(r("oak_shelf"))).toBe(false); // static held items
    expect(blockEntityAnimatesPerFrame(r("minecraft:furnace"))).toBe(false); // no special renderer
  });
});

describe("BlockEntitySceneBaker (hybrid bake)", () => {
  const setup = () => {
    const device = new FakeGraphicsDevice();
    const tex = device.createTexture({ width: 1, height: 1, format: 0 as never, label: "e" });
    const index = new BlockEntityIndex();
    const baker = new BlockEntitySceneBaker(device, index, tex, uvFor);
    return { device, index, baker };
  };

  it("bakes an idle chest into ONE static section mesh, cached across frames (no re-bake)", () => {
    const { device, index, baker } = setup();
    index.set(chest(2, 0, 2));
    expect(baker.frame(0).length).toBe(1);
    const after1 = device.liveBufferCount();
    baker.frame(0.1);
    expect(device.liveBufferCount()).toBe(after1);
    expect(baker.stats.staticDrawn).toBe(1);
  });

  it("a frustum-culled section's idle chest is NOT drawn (culls with its section)", () => {
    const { index, baker } = setup();
    index.set(chest(2, 0, 2));
    expect(baker.frame(0, () => false).length).toBe(0);
  });

  it("an OPEN chest is drawn per-frame and re-encodes each frame (no churn)", () => {
    const { device, index, baker } = setup();
    index.set(chest(2, 0, 2, {}, false));
    index.setOpen(2, 0, 2, true); // open → lid animates → per-frame path (openness snaps to 1 on first sight)
    expect(baker.frame(0).length).toBe(1);
    expect(baker.stats.animating).toBe(1);
    expect(baker.stats.staticDrawn).toBe(0);
    const live = device.liveBufferCount();
    baker.frame(0.2);
    expect(device.liveBufferCount()).toBe(live);
  });

  it("opening a chest moves it from the static bake to per-frame; closing fully returns it to static", () => {
    const { index, baker } = setup();
    index.set(chest(2, 0, 2, {}, false));
    // Closed: first frame seeds the lid state (openness 0) → baked into the static section mesh.
    expect(baker.frame(0).length).toBe(1);
    expect(baker.stats.staticDrawn).toBe(1);
    expect(baker.stats.animating).toBe(0);
    // Open it: the lid leaves the static bake and draws per-frame (no double-draw — it's in exactly one set).
    const sk = index.setOpen(2, 0, 2, true);
    expect(sk).toBe(sectionOfBlock(2, 0, 2));
    baker.invalidate(sk);
    expect(baker.frame(0.1).length).toBe(1);
    expect(baker.stats.staticDrawn).toBe(0);
    expect(baker.stats.animating).toBe(1);
    // Close it and let the lid fully shut (>0.5s of clock): it returns to the static bake, never double-drawn.
    index.setOpen(2, 0, 2, false);
    baker.frame(0.2);
    baker.frame(1.0); // openness reaches 0 → back to static
    expect(baker.stats.staticDrawn).toBe(1);
    expect(baker.stats.animating).toBe(0);
  });

  it("animates BOTH halves of a double chest when both are opened (the sim opens both)", () => {
    const { index, baker } = setup();
    // A double chest is two BE records (left + right halves) in one section.
    index.set(chest(2, 0, 2, { type: "left", facing: "north" }, false));
    index.set(chest(3, 0, 2, { type: "right", facing: "north" }, false));
    // The sim opens both halves together, so both records get open:true → both lids draw per-frame.
    index.setOpen(2, 0, 2, true);
    index.setOpen(3, 0, 2, true);
    baker.frame(0);
    expect(baker.stats.animating).toBe(2); // both lids animate
    expect(baker.stats.staticDrawn).toBe(0); // neither half is in the static bake while open
  });

  it("a non-openable animated BE still uses the `animating` toggle (bell — hybridTerrain)", () => {
    const { index, baker } = setup();
    index.set({ x: 2, y: 0, z: 2, type: "bell", props: { facing: "south", attachment: "floor" }, animating: false });
    baker.frame(0);
    expect(baker.stats.animating).toBe(0); // idle → static (the bell support box)
    const sk = index.setAnimating(2, 0, 2, true);
    baker.invalidate(sk);
    baker.frame(0.1);
    expect(baker.stats.animating).toBe(1); // toggled → per-frame swing
  });

  it("draws an idle-loop banner on the per-frame path even though its record is not toggled (BBE §5)", () => {
    const { index, baker } = setup();
    index.set({ x: 2, y: 0, z: 2, type: "white_banner", props: { rotation: "0" }, animating: false });
    expect(baker.frame(0).length).toBe(1);
    expect(baker.stats.staticDrawn).toBe(0); // not in the static section bake
    expect(baker.stats.animating).toBe(1); // waves every frame by default
  });
});
