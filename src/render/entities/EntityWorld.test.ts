import { describe, expect, it } from "vitest";
import { FakeGraphicsDevice } from "../../core/testing/FakeGraphicsDevice";
import { TextureFormat } from "../../core/GraphicsDevice";
import { Camera } from "../../camera/Camera";
import { computeOcclusion } from "../../mesh/model/OcclusionShape";
import { Direction, TerrainPass } from "../../types";
import type { BakedBlockModel } from "../../mesh/model/BakedBlockModel";
import { EntityWorld } from "./EntityWorld";

function cubeModel(): BakedBlockModel {
  const q = {
    positions: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]] as [number[], number[], number[], number[]],
    atlasUV: [[0, 0], [1, 0], [1, 1], [0, 1]] as [number[], number[], number[], number[]],
    normal: Direction.North,
    cullface: null,
    layer: TerrainPass.Solid,
    colorRGBA: 0xffffffff,
    material: 0,
  } as BakedBlockModel["quads"][number];
  return { quads: [q], occlusion: computeOcclusion([], () => true), skipGroup: null };
}

const makeWorld = () => {
  const device = new FakeGraphicsDevice();
  const blockAtlas = device.createTexture({ width: 1, height: 1, format: TextureFormat.Rgba8Srgb, label: "b" });
  const entityAtlas = device.createTexture({ width: 1, height: 1, format: TextureFormat.Rgba8Srgb, label: "e" });
  const world = new EntityWorld({
    device,
    colorFormat: TextureFormat.Bgra8Srgb,
    depthFormat: TextureFormat.Depth24Plus,
    bakeBlock: () => cubeModel(),
    blockAtlas,
    entityAtlas,
    entityUvFor: () => ({ u: 0, v: 0, width: 1, height: 1 }),
  });
  const camera = new Camera();
  camera.target = [0, 0, 0];
  camera.distance = 24;
  return { device, world, camera };
};

describe("EntityWorld orchestrator", () => {
  it("ingests + renders a falling block entity (one draw submitted)", () => {
    const { device, world, camera } = makeWorld();
    world.ingestEntities([{ id: "fb", type: "minecraft:falling_block", position: [0, 0, 0], properties: { block: "minecraft:sand" } }]);
    world.render(camera, 0, 0, 256, 256);
    expect(world.stats.entities).toBe(1);
    expect(device.log.drawCalls).toBeGreaterThan(0);
  });

  it("frustum-culls an entity placed far behind the camera", () => {
    const { world, camera } = makeWorld();
    world.ingestEntities([{ id: "fb", type: "minecraft:falling_block", position: [0, 0, 10000], properties: { block: "minecraft:sand" } }]);
    world.render(camera, 0, 0, 256, 256);
    expect(world.stats.entities).toBe(0);
  });

  it("renders an idle block entity as a culled static section mesh", () => {
    const { world, camera } = makeWorld();
    world.setBlockEntity({ x: 0, y: 0, z: 0, type: "minecraft:chest", props: {}, animating: false });
    world.render(camera, 0, 0, 256, 256);
    expect(world.stats.beStatic).toBe(1);
    expect(world.stats.beAnimating).toBe(0);
  });

  it("opens a chest (lid animates per-frame) and reports the lid transition to the render gate", () => {
    const { world, camera } = makeWorld();
    world.setBlockEntity({ x: 0, y: 0, z: 0, type: "minecraft:chest", props: {}, animating: false });
    world.render(camera, 0, 0, 256, 256); // closed → static
    expect(world.stats.beStatic).toBe(1);
    expect(world.stats.beAnimating).toBe(0);
    expect(world.hasContainerLidTransitions()).toBe(false); // a shut chest needs no per-frame redraw
    expect(world.hasActiveAnimations()).toBe(false); // and it isn't a decorative animation either
    world.setBlockEntityOpen(0, 0, 0, true);
    world.render(camera, 0.1, 0.1, 256, 256); // opening → per-frame lid
    expect(world.stats.beStatic).toBe(0);
    expect(world.stats.beAnimating).toBe(1);
    // The lid transition keeps the loop drawing REGARDLESS of the animations master switch (like a piston),
    // so it's reported via hasContainerLidTransitions — NOT the master-switch-gated hasActiveAnimations.
    expect(world.hasContainerLidTransitions()).toBe(true);
    expect(world.hasActiveAnimations()).toBe(false);
  });
});

describe("EntityWorld idle-loop animations + render gate (BBE §5 inverted-trigger)", () => {
  it("animates an idle-loop banner BY DEFAULT (animating:false → per-frame, never baked frozen)", () => {
    const { world, camera } = makeWorld();
    world.setBlockEntity({ x: 0, y: 0, z: 0, type: "white_banner", props: { rotation: "0" }, animating: false });
    world.render(camera, 0, 0, 256, 256);
    expect(world.stats.beAnimating).toBe(1); // the flag waves without any editor toggle
    expect(world.stats.beStatic).toBe(0); // never enters the static section bake
  });

  it("animates an idle-loop conduit by default (the frozen-on-static-path + fixed-eye bug is gone)", () => {
    const { world, camera } = makeWorld();
    world.setBlockEntity({ x: 0, y: 0, z: 0, type: "conduit", props: { active: "true", hunting: "true" }, animating: false });
    world.render(camera, 0, 0, 256, 256);
    expect(world.stats.beAnimating).toBe(1);
    expect(world.stats.beStatic).toBe(0);
  });

  it("keeps an event-driven chest idle (static) until toggled (idle-loop does NOT apply)", () => {
    const { world, camera } = makeWorld();
    world.setBlockEntity({ x: 0, y: 0, z: 0, type: "minecraft:chest", props: {}, animating: false });
    world.render(camera, 0, 0, 256, 256);
    expect(world.stats.beStatic).toBe(1);
    expect(world.stats.beAnimating).toBe(0);
  });

  it("hasActiveAnimations: tracks idle-loop / toggled / special BEs and re-evaluates after each mutation", () => {
    const { world } = makeWorld();
    expect(world.hasActiveAnimations()).toBe(false); // empty scene
    world.setBlockEntity({ x: 0, y: 0, z: 0, type: "minecraft:chest", props: {}, animating: false });
    expect(world.hasActiveAnimations()).toBe(false); // an idle chest does not animate
    world.setBlockEntity({ x: 1, y: 0, z: 0, type: "white_banner", props: {}, animating: false });
    expect(world.hasActiveAnimations()).toBe(true); // an idle-loop banner does (cache invalidated by set)
    world.removeBlockEntity(1, 0, 0);
    expect(world.hasActiveAnimations()).toBe(false); // back to only the idle chest (cache invalidated by remove)
    world.setBlockEntityAnimating(0, 0, 0, true);
    expect(world.hasActiveAnimations()).toBe(true); // toggling the chest preview keeps the loop alive
  });

  it("hasActiveAnimations: an always-animated special-only BE (spinning spawner) counts", () => {
    const { world } = makeWorld();
    world.setBlockEntity({ x: 0, y: 0, z: 0, type: "spawner", props: {}, animating: false });
    expect(world.hasActiveAnimations()).toBe(true);
  });
});
