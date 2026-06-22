import { describe, expect, it } from "vitest";
import { EntityScene, type EntitySnapshot } from "./EntityScene";

const snap = (id: string, pos: [number, number, number], extra: Partial<EntitySnapshot> = {}): EntitySnapshot => ({
  id,
  type: extra.type ?? "minecraft:falling_block",
  position: pos,
  velocity: extra.velocity,
  properties: extra.properties,
});

describe("EntityScene", () => {
  it("a freshly spawned entity does not interpolate (prev == pos)", () => {
    const s = new EntityScene();
    s.ingest([snap("a", [10, 5, 3])]);
    expect(s.size).toBe(1);
    // At any partialTick it sits exactly at its spawn position — no slide-in from origin.
    for (const t of [0, 0.5, 1]) {
      expect(s.frame(t)[0].position).toEqual([10, 5, 3]);
    }
  });

  it("interpolates prev→pos by partialTick on a moved entity", () => {
    const s = new EntityScene();
    s.ingest([snap("a", [0, 0, 0])]); // tick 0
    s.ingest([snap("a", [10, 20, -4])]); // tick 1: travels from (0,0,0)
    expect(s.frame(0)[0].position).toEqual([0, 0, 0]);
    expect(s.frame(0.5)[0].position).toEqual([5, 10, -2]);
    expect(s.frame(1)[0].position).toEqual([10, 20, -4]);
  });

  it("clamps partialTick to [0,1] (a long frame never overshoots the target tick)", () => {
    const s = new EntityScene();
    s.ingest([snap("a", [0, 0, 0])]);
    s.ingest([snap("a", [10, 0, 0])]);
    expect(s.frame(-0.5)[0].position).toEqual([0, 0, 0]);
    expect(s.frame(2)[0].position).toEqual([10, 0, 0]);
  });

  it("removes despawned ids", () => {
    const s = new EntityScene();
    s.ingest([snap("a", [0, 0, 0]), snap("b", [1, 1, 1])]);
    s.ingest([], ["a"]);
    expect(s.size).toBe(1);
    expect(s.frame(0).map((e) => e.id)).toEqual(["b"]);
  });

  it("carries velocity + properties from the latest snapshot", () => {
    const s = new EntityScene();
    s.ingest([snap("sheep", [0, 0, 0], { type: "minecraft:sheep", velocity: [0.2, 0, 0], properties: { color: "red" } })]);
    const e = s.frame(0)[0];
    expect(e.type).toBe("minecraft:sheep");
    expect(e.velocity).toEqual([0.2, 0, 0]);
    expect(e.properties.color).toBe("red");
  });

  it("an unchanged entity (absent from the next diff) holds its position", () => {
    const s = new EntityScene();
    s.ingest([snap("a", [0, 0, 0]), snap("b", [5, 5, 5])]);
    s.ingest([snap("a", [1, 0, 0])]); // only 'a' changed this tick
    // 'b' is not in the diff → stays put at any partialTick.
    const b = s.frame(1).find((e) => e.id === "b")!;
    expect(b.position).toEqual([5, 5, 5]);
  });

  it("settles a moved-then-stopped entity at its rest position (AUDIT H6)", () => {
    const s = new EntityScene();
    s.ingest([snap("a", [0, 0, 0])]); // tick 0: spawn
    s.ingest([snap("a", [10, 0, 0])]); // tick 1: moves 0 → 10
    expect(s.frame(0.5)[0].position).toEqual([5, 0, 0]); // mid-travel during the moving tick
    s.ingest([]); // tick 2: stops → omitted from the sparse diff
    // prev must advance to pos, so it sits at rest for every partialTick (not lerping back toward 5).
    for (const t of [0, 0.5, 1]) {
      expect(s.frame(t)[0].position, `t=${t}`).toEqual([10, 0, 0]);
    }
  });

  it("keeps interpolating a continuously moving entity each tick (fix doesn't stick prev)", () => {
    const s = new EntityScene();
    s.ingest([snap("a", [0, 0, 0])]);
    s.ingest([snap("a", [10, 0, 0])]); // 0 → 10
    expect(s.frame(0.5)[0].position).toEqual([5, 0, 0]);
    s.ingest([snap("a", [20, 0, 0])]); // 10 → 20 (prev advanced to 10, not stuck at 0)
    expect(s.frame(0)[0].position).toEqual([10, 0, 0]);
    expect(s.frame(0.5)[0].position).toEqual([15, 0, 0]);
    expect(s.frame(1)[0].position).toEqual([20, 0, 0]);
  });

  it("clear() drops all tracked entities", () => {
    const s = new EntityScene();
    s.ingest([snap("a", [0, 0, 0])]);
    s.clear();
    expect(s.size).toBe(0);
    expect(s.frame(0)).toEqual([]);
  });
});
