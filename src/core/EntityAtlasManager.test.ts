import { describe, expect, it } from "vitest";
import { packEntityAtlas, EntityAtlasManager, type EntitySprite } from "./EntityAtlasManager";
import { FakeGraphicsDevice } from "./testing/FakeGraphicsDevice";

const sprite = (name: string, w: number, h: number): EntitySprite => ({ name, width: w, height: h, rgba: new Uint8Array(w * h * 4) });

describe("packEntityAtlas", () => {
  it("packs without overlap and within the atlas bounds", () => {
    const sprites = [sprite("a", 64, 64), sprite("b", 64, 32), sprite("c", 32, 32), sprite("d", 16, 48)];
    const pack = packEntityAtlas(sprites);
    expect(pack.placements.length).toBe(4);
    for (const p of pack.placements) {
      expect(p.x + p.width).toBeLessThanOrEqual(pack.width);
      expect(p.y + p.height).toBeLessThanOrEqual(pack.height);
    }
    // No two placements overlap.
    for (let i = 0; i < pack.placements.length; i++) {
      for (let j = i + 1; j < pack.placements.length; j++) {
        expect(overlaps(pack.placements[i], pack.placements[j])).toBe(false);
      }
    }
  });

  it("produces power-of-two dimensions", () => {
    const pack = packEntityAtlas([sprite("a", 100, 50)]);
    expect(isPow2(pack.width)).toBe(true);
    expect(isPow2(pack.height)).toBe(true);
  });

  it("is deterministic (stable order)", () => {
    const s = [sprite("z", 32, 32), sprite("a", 32, 32), sprite("m", 32, 32)];
    expect(packEntityAtlas(s)).toEqual(packEntityAtlas([...s].reverse()));
  });
});

describe("EntityAtlasManager", () => {
  it("builds an atlas and returns normalized rects inside [0,1]", () => {
    const device = new FakeGraphicsDevice();
    const atlas = new EntityAtlasManager(device);
    atlas.build([sprite("chest", 64, 64), sprite("sheep", 64, 32)]);
    expect(atlas.has("chest")).toBe(true);
    const r = atlas.uvFor("sheep");
    expect(r.u).toBeGreaterThanOrEqual(0);
    expect(r.u + r.width).toBeLessThanOrEqual(1);
    expect(r.v + r.height).toBeLessThanOrEqual(1);
  });

  it("throws for unknown sprites", () => {
    const atlas = new EntityAtlasManager(new FakeGraphicsDevice());
    atlas.build([sprite("chest", 64, 64)]);
    expect(() => atlas.uvFor("nope")).toThrow();
  });
});

function overlaps(a: { x: number; y: number; width: number; height: number }, b: typeof a): boolean {
  return a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
}
function isPow2(n: number): boolean {
  return (n & (n - 1)) === 0;
}
