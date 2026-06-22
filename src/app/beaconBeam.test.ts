import { describe, expect, it } from "vitest";
import { computeBeaconBeam } from "./beaconBeam";

const key = (x: number, y: number, z: number): string => `${x},${y},${z}`;
const world = (blocks: Record<string, string>) => (x: number, y: number, z: number): string | null => blocks[key(x, y, z)] ?? null;

/** Build a complete `level`-layer beacon base pyramid below (0,0,0) of `base` blocks. */
function pyramid(level: number, base = "minecraft:iron_block"): Record<string, string> {
  const out: Record<string, string> = {};
  for (let step = 1; step <= level; step++)
    for (let lx = -step; lx <= step; lx++)
      for (let lz = -step; lz <= step; lz++) out[key(lx, -step, lz)] = base;
  return out;
}

describe("computeBeaconBeam (vanilla BeaconBlockEntity beam visibility)", () => {
  it("no base pyramid → NO beam (the bug: a non-activated beacon must not beam)", () => {
    expect(computeBeaconBeam(world({}), 0, 0, 0, 16).active).toBe(false);
  });

  it("a complete 1-layer iron base + clear column → an active white beam to the sky", () => {
    const beam = computeBeaconBeam(world(pyramid(1)), 0, 0, 0, 16);
    expect(beam.active).toBe(true);
    expect(beam.color).toBe(0xffffff);
    expect(beam.height).toBeGreaterThan(16); // reaches above the highest block
  });

  it("an incomplete base (one block missing) → no beam", () => {
    const b = pyramid(1);
    delete b[key(1, -1, 1)];
    expect(computeBeaconBeam(world(b), 0, 0, 0, 16).active).toBe(false);
  });

  it("any mix of beacon-base blocks counts toward the pyramid", () => {
    const b = pyramid(1);
    b[key(0, -1, 0)] = "minecraft:gold_block";
    b[key(1, -1, 1)] = "minecraft:diamond_block";
    b[key(-1, -1, -1)] = "minecraft:emerald_block";
    b[key(1, -1, -1)] = "minecraft:netherite_block";
    expect(computeBeaconBeam(world(b), 0, 0, 0, 16).active).toBe(true);
  });

  it("a non-base block in the pyramid layer breaks activation", () => {
    const b = pyramid(1);
    b[key(0, -1, 0)] = "minecraft:stone";
    expect(computeBeaconBeam(world(b), 0, 0, 0, 16).active).toBe(false);
  });

  it("an opaque block ANYWHERE in the column kills the whole beam (vanilla clears all sections)", () => {
    const b = { ...pyramid(1), [key(0, 10, 0)]: "minecraft:stone" };
    expect(computeBeaconBeam(world(b), 0, 0, 0, 16).active).toBe(false);
  });

  it("stained glass tints the beam; clear glass passes through untinted", () => {
    const b = { ...pyramid(1), [key(0, 5, 0)]: "minecraft:glass", [key(0, 6, 0)]: "minecraft:red_stained_glass" };
    const beam = computeBeaconBeam(world(b), 0, 0, 0, 16);
    expect(beam.active).toBe(true);
    expect(beam.color).toBe(0xb02e26); // DYE_DIFFUSE.red
  });

  it("stained-glass PANES tint the beam too (they are BeaconBeamBlocks)", () => {
    const b = { ...pyramid(1), [key(0, 5, 0)]: "minecraft:blue_stained_glass_pane" };
    expect(computeBeaconBeam(world(b), 0, 0, 0, 16).color).toBe(0x3c44aa); // DYE_DIFFUSE.blue
  });
});
