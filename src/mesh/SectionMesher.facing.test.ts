// FS-2 — opaque/cutout facing partition. `partitionFacing` regroups the SAME quads into 7 facing runs
// (winding preserved); the default path stays byte-identical. Proven here: same multiset, correct
// per-facing counts, and every quad lands in the run matching its geometric facing.

import { describe, it, expect } from "vitest";
import { meshSection, type BakedModelProvider } from "./SectionMesher";
import { ModelBaker, type BakedBlockModel, type BlockEntry } from "./model/BakedBlockModel";
import { TintProvider } from "./model/TintProvider";
import { fixtureBlockstates, fixtureProvider } from "./model/__fixtures__/load";
import { SnapshotSource, type BlockSource } from "../world/SnapshotSource";
import { sectionKey } from "../world/SectionKey";
import { DirtyReason, TerrainPass } from "../types";
import { decodeVertex } from "./VertexFormat";
import { quadFacing } from "./ModelQuadFacing";

const STONE = 1;
const PALETTE: Record<number, BlockEntry> = { [STONE]: { name: "stone", props: {} } };
const opacityOf = () => ({ hasTransparent: false, hasTranslucent: false });

function makeProvider(): BakedModelProvider {
  const baker = new ModelBaker({
    blockstate: (name) => fixtureBlockstates[name],
    models: fixtureProvider,
    uvFor: () => ({ u: 0, v: 0, width: 1, height: 1 }),
    opacityOf,
    tint: new TintProvider(),
  });
  const cache = new Map<number, BakedBlockModel | null>();
  return { get(id) { if (id === 0) return null; if (!cache.has(id)) cache.set(id, PALETTE[id] ? baker.bake(PALETTE[id]) : null); return cache.get(id) ?? null; } };
}

function meshWorld(solid: Map<string, number>, partitionFacing: boolean) {
  const src: BlockSource = { getBlock: (x, y, z) => solid.get(`${x},${y},${z}`) ?? 0 };
  const snap = new SnapshotSource(src).buildSnapshot(sectionKey(0, 0, 0), 1, DirtyReason.InitialLoad);
  return meshSection(snap, makeProvider(), undefined, { partitionFacing });
}

/** Every vertex, JSON-stringified + sorted — the order-independent multiset for an equality check. */
function vertexMultiset(buf: ArrayBuffer): string[] {
  const dv = new DataView(buf);
  const out: string[] = [];
  for (let i = 0; i < buf.byteLength; i += 20) out.push(JSON.stringify(decodeVertex(dv, i)));
  return out.sort();
}

describe("SectionMesher facing partition (FS-2)", () => {
  it("off path is unchanged — no facingVertexCounts, single run", () => {
    const off = meshWorld(new Map([["0,0,0", STONE]]), false);
    expect(off.parts[TerrainPass.Solid]!.quadCount).toBe(6);
    expect(off.parts[TerrainPass.Solid]!.facingVertexCounts).toBeUndefined();
  });

  it("a stone cube partitions into one quad per axis facing, none unassigned", () => {
    const on = meshWorld(new Map([["0,0,0", STONE]]), true);
    const part = on.parts[TerrainPass.Solid]!;
    expect(part.quadCount).toBe(6);
    // 6 axis facings (PosX..NegZ) each get one quad (4 verts); UNASSIGNED gets none.
    expect(Array.from(part.facingVertexCounts!)).toEqual([4, 4, 4, 4, 4, 4, 0]);
    // The run lengths tile the whole buffer.
    expect(Array.from(part.facingVertexCounts!).reduce((a, b) => a + b, 0) * 20).toBe(part.vertexData.byteLength);
  });

  it("partitioned bytes are the SAME vertex multiset as the off path (reordered, not re-encoded)", () => {
    const world = new Map<string, number>([["0,0,0", STONE], ["2,0,0", STONE], ["0,2,0", STONE]]); // separated cubes
    const off = meshWorld(world, false);
    const on = meshWorld(world, true);
    expect(vertexMultiset(on.parts[TerrainPass.Solid]!.vertexData)).toEqual(vertexMultiset(off.parts[TerrainPass.Solid]!.vertexData));
  });

  it("every quad lands in the run matching its geometric facing", () => {
    const world = new Map<string, number>([["0,0,0", STONE], ["2,0,0", STONE]]);
    const part = meshWorld(world, true).parts[TerrainPass.Solid]!;
    const dv = new DataView(part.vertexData);
    const pos = (i: number) => { const r = decodeVertex(dv, i * 20); return [r.x, r.y, r.z] as [number, number, number]; };
    let vstart = 0;
    Array.from(part.facingVertexCounts!).forEach((vcount, facing) => {
      for (let q = 0; q < vcount; q += 4) {
        const i = vstart + q;
        expect(quadFacing(pos(i), pos(i + 1), pos(i + 2), pos(i + 3))).toBe(facing); // run `facing` holds only that facing
      }
      vstart += vcount;
    });
  });
});
