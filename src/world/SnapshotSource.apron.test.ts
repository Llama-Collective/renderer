// SL-3 — widening the snapshot apron to 2 (for smooth-AO corner reads) must NOT change the flat geometry:
// the apron only widens sampling REACH, so the emitted opaque quads are byte-identical at apron 1 vs 2.

import { describe, it, expect } from "vitest";
import { SnapshotSource, snapshotStride, type BlockSource } from "./SnapshotSource";
import { meshSection, type BakedModelProvider } from "../mesh/SectionMesher";
import { ModelBaker, type BakedBlockModel, type BlockEntry } from "../mesh/model/BakedBlockModel";
import { TintProvider } from "../mesh/model/TintProvider";
import { fixtureBlockstates, fixtureProvider } from "../mesh/model/__fixtures__/load";
import { sectionKey } from "./SectionKey";
import { DirtyReason, TerrainPass } from "../types";

const STONE = 1;
function makeProvider(): BakedModelProvider {
  const baker = new ModelBaker({
    blockstate: (name) => fixtureBlockstates[name],
    models: fixtureProvider,
    uvFor: () => ({ u: 0, v: 0, width: 1, height: 1 }),
    opacityOf: () => ({ hasTransparent: false, hasTranslucent: false }),
    tint: new TintProvider(),
  });
  const palette: Record<number, BlockEntry> = { [STONE]: { name: "stone", props: {} } };
  const cache = new Map<number, BakedBlockModel | null>();
  return { get(id) { if (id === 0) return null; if (!cache.has(id)) cache.set(id, palette[id] ? baker.bake(palette[id]) : null); return cache.get(id) ?? null; } };
}

// A solid 16³ stone section against air → its outer shell faces read the apron neighbours (air at ±1/16).
const solidWorld: BlockSource = { getBlock: (x, y, z) => (x >= 0 && x < 16 && y >= 0 && y < 16 && z >= 0 && z < 16 ? STONE : 0) };

describe("SnapshotSource apron widen (SL-3)", () => {
  it("snapshotStride(2) === 20 (the apron-2 cube side)", () => {
    expect(snapshotStride(2)).toBe(20);
    expect(snapshotStride(1)).toBe(18);
  });

  it("apron 1 vs apron 2 mesh to BYTE-identical opaque geometry (apron only widens sampling reach)", () => {
    const provider = makeProvider();
    const snap1 = new SnapshotSource(solidWorld, 1).buildSnapshot(sectionKey(0, 0, 0), 1, DirtyReason.InitialLoad);
    const snap2 = new SnapshotSource(solidWorld, 2).buildSnapshot(sectionKey(0, 0, 0), 1, DirtyReason.InitialLoad);
    expect(snap1.apron).toBe(1);
    expect(snap2.apron).toBe(2);

    const out1 = meshSection(snap1, provider);
    const out2 = meshSection(snap2, provider);
    expect(out2.parts[TerrainPass.Solid]!.quadCount).toBe(out1.parts[TerrainPass.Solid]!.quadCount);
    expect(new Uint8Array(out2.parts[TerrainPass.Solid]!.vertexData)).toEqual(new Uint8Array(out1.parts[TerrainPass.Solid]!.vertexData));
  });
});
