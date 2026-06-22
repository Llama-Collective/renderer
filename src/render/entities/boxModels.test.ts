import { describe, expect, it } from "vitest";
import { FakeGraphicsDevice } from "../../core/testing/FakeGraphicsDevice";
import { TerrainPass, type SpriteUv } from "../../types";
import { createBoxModelSource, type AtlasRects } from "./boxModels";
import { EntityModelFactory } from "./EntityModelFactory";
import { Mat4Frame } from "../../mesh/entity/mat4";
import type { RenderEntity } from "./EntityScene";
import type { EntityDraw } from "./EntityRenderer";

/** A4: buildDraws emits into a caller array + returns a count; collect for asserts / lifecycle exercises. */
function drawsOf(factory: EntityModelFactory, e: RenderEntity, clock = 0): EntityDraw[] {
  const out: EntityDraw[] = [];
  factory.buildDraws(e, clock, out);
  return out;
}

const FULL: SpriteUv = { u: 0, v: 0, width: 1, height: 1 };
const atlas: AtlasRects = { uvFor: () => FULL, has: () => true };
const FRAME = new Mat4Frame(); // A3: build() now takes the per-frame matrix arena

const sheep = (props: Record<string, string>, velocity: [number, number, number] = [0, 0, 0]): RenderEntity => ({
  id: "s1",
  type: "minecraft:sheep",
  position: [0, 0, 0],
  velocity,
  properties: props,
});

describe("createBoxModelSource — sheep", () => {
  const src = createBoxModelSource(atlas, { id: 1 } as never);

  it("returns body + wool layers, wool tinted by dye colour", () => {
    const layers = src.build(sheep({ color: "red" }), 0, FRAME)!;
    expect(layers.length).toBe(2);
    expect(layers[0].tint).toBeUndefined(); // body untinted
    const wool = layers[1].tint!;
    expect(wool[0]).toBeGreaterThan(wool[2]); // red: R > B
  });

  it("different dye colours give different wool tints (same geometry)", () => {
    const red = src.build(sheep({ color: "red" }), 0, FRAME)![1].tint!;
    const blue = src.build(sheep({ color: "blue" }), 0, FRAME)![1].tint!;
    expect(red).not.toEqual(blue);
  });

  it("a resting sheep is static (cacheable); a moving sheep is dynamic", () => {
    expect(src.build(sheep({ color: "white" }), 0, FRAME)![0].dynamic).toBeFalsy();
    expect(src.build(sheep({ color: "white" }, [0.3, 0, 0]), 0, FRAME)![0].dynamic).toBe(true);
  });

  it("bakes non-empty cutout geometry", () => {
    const layers = src.build(sheep({ color: "white" }), 0, FRAME)!;
    // `verts` is a lazy thunk now (ENT-1) — invoke it to bake.
    expect(layers[0].verts()[TerrainPass.Cutout]!.length).toBeGreaterThan(0);
  });
});

describe("EntityModelFactory + box source (dynamic buffer lifecycle)", () => {
  const setup = () => {
    const device = new FakeGraphicsDevice();
    const blockAtlas = device.createTexture({ width: 1, height: 1, format: 0 as never, label: "b" });
    const entityAtlas = device.createTexture({ width: 1, height: 1, format: 0 as never, label: "e" });
    const factory = new EntityModelFactory({ device, blockAtlas, bakeBlock: () => null, boxModels: createBoxModelSource(atlas, entityAtlas) });
    return { device, factory };
  };

  it("ENT-1: a resting (static) layer bakes ONCE across repeated frames; dynamic re-bakes each frame", () => {
    const device = new FakeGraphicsDevice();
    const tex = device.createTexture({ width: 1, height: 1, format: 0 as never, label: "e" });
    let staticBakes = 0;
    let dynBakes = 0;
    const cutout = { [TerrainPass.Cutout]: [{ x: 0, y: 0, z: 0, u: 0, v: 0, normal: 0, colorRGBA: 0xffffffff, material: 0 }, { x: 1, y: 0, z: 0, u: 1, v: 0, normal: 0, colorRGBA: 0xffffffff, material: 0 }, { x: 1, y: 1, z: 0, u: 1, v: 1, normal: 0, colorRGBA: 0xffffffff, material: 0 }, { x: 0, y: 1, z: 0, u: 0, v: 1, normal: 0, colorRGBA: 0xffffffff, material: 0 }] };
    const boxModels = {
      texture: tex,
      build: (): import("./EntityModelFactory").BoxModelLayer[] => [
        { geomKey: "test:static", verts: () => { staticBakes++; return cutout; }, model: new Float32Array(16) as never },
        { geomKey: "test:dyn", verts: () => { dynBakes++; return cutout; }, model: new Float32Array(16) as never, dynamic: true },
      ],
    };
    const factory = new EntityModelFactory({ device, blockAtlas: tex, bakeBlock: () => null, boxModels });
    const e = sheep({});
    for (let f = 0; f < 5; f++) {
      factory.beginFrame();
      drawsOf(factory, e, 0);
      factory.endFrame();
    }
    expect(staticBakes).toBe(1); // baked once, served from cache afterwards (ENT-1)
    expect(dynBakes).toBe(5); // re-posed every frame
  });

  it("renders a walking sheep via dynamic re-encode (body + wool draws)", () => {
    const { factory } = setup();
    factory.beginFrame();
    const draws = drawsOf(factory, sheep({ color: "lime" }, [0.3, 0, 0]), 0.1);
    factory.endFrame();
    expect(draws.length).toBe(2); // body + wool, one cutout pass each
    expect(draws.every((d) => d.quadCount > 0)).toBe(true);
  });

  it("frees a despawned sheep's dynamic buffers on the next frame", () => {
    const { device, factory } = setup();
    factory.beginFrame();
    drawsOf(factory, sheep({ color: "white" }, [0.3, 0, 0]), 0);
    factory.endFrame();
    const withSheep = device.liveBufferCount();
    // Next frame: no sheep built → its dynamic buffers reclaimed.
    factory.beginFrame();
    factory.endFrame();
    expect(device.liveBufferCount()).toBeLessThan(withSheep);
  });

  it("reuses dynamic buffers across frames for the same entity (no churn)", () => {
    const { device, factory } = setup();
    factory.beginFrame();
    drawsOf(factory, sheep({ color: "white" }, [0.3, 0, 0]), 0);
    factory.endFrame();
    const after1 = device.liveBufferCount();
    factory.beginFrame();
    drawsOf(factory, sheep({ color: "white" }, [0.3, 0, 0]), 0.2);
    factory.endFrame();
    expect(device.liveBufferCount()).toBe(after1); // same buffers, rewritten
  });
});
