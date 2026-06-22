// P6/LM-1: the mesher CONSUMES `snapshot.light` as flat per-face lighting (vanilla non-smooth); with no
// light array every face bakes full-bright (so the editor's default render is unchanged).

import { describe, it, expect } from "vitest";
import { meshSection, type BakedModelProvider } from "./SectionMesher";
import { ModelBaker, type BakedBlockModel, type BlockEntry } from "./model/BakedBlockModel";
import { TintProvider } from "./model/TintProvider";
import { fixtureBlockstates, fixtureProvider } from "./model/__fixtures__/load";
import { snapshotIndex, snapshotStride, type SectionSnapshot } from "../world/SnapshotSource";
import { sectionKey } from "../world/SectionKey";
import { DirtyReason, Direction, TerrainPass } from "../types";
import { decodeVertex, VERTEX_STRIDE_BYTES } from "./VertexFormat";
import { FULLBRIGHT_LIGHT, unpackLight } from "./Lighting";

const STONE = 1;
const PALETTE: Record<number, BlockEntry> = { [STONE]: { name: "stone", props: {} } };

function makeProvider(): BakedModelProvider {
  const baker = new ModelBaker({
    blockstate: (name) => fixtureBlockstates[name],
    models: fixtureProvider,
    uvFor: () => ({ u: 0, v: 0, width: 1, height: 1 }),
    opacityOf: () => ({ hasTransparent: false, hasTranslucent: false }),
    tint: new TintProvider(),
  });
  const cache = new Map<number, BakedBlockModel | null>();
  return { get: (id) => (id === 0 ? null : (cache.has(id) ? cache.get(id)! : (cache.set(id, baker.bake(PALETTE[id])), cache.get(id)!))) };
}

const APRON = 1;
/** A 16³ snapshot with one stone cube at local (0,0,0) and an optional per-cell nibble-light array. */
function stoneSnapshot(light?: Uint8Array): SectionSnapshot {
  const s = snapshotStride(APRON);
  const blocks = new Uint32Array(s * s * s);
  blocks[snapshotIndex(0, 0, 0, APRON)] = STONE;
  return { sectionKey: sectionKey(0, 0, 0), generation: 1, origin: [0, 0, 0], size: 16, apron: APRON, blocks, light, changedReason: DirtyReason.InitialLoad };
}

/** Decode every solid vertex and bucket its light word by face normal. */
function lightByNormal(out: ReturnType<typeof meshSection>): Map<number, Set<number>> {
  const part = out.parts[TerrainPass.Solid]!;
  const dv = new DataView(part.vertexData);
  const byNormal = new Map<number, Set<number>>();
  for (let i = 0; i < part.quadCount * 4; i++) {
    const v = decodeVertex(dv, i * VERTEX_STRIDE_BYTES);
    if (!byNormal.has(v.normal)) byNormal.set(v.normal, new Set());
    byNormal.get(v.normal)!.add(v.light! >>> 0);
  }
  return byNormal;
}

describe("SectionMesher light consumption (P6/LM-1)", () => {
  it("no light array → every face is full-bright (render unchanged)", () => {
    const out = meshSection(stoneSnapshot(), makeProvider());
    const byNormal = lightByNormal(out);
    expect(byNormal.size).toBe(6); // 6 faces of the cube
    for (const words of byNormal.values()) {
      expect([...words]).toEqual([FULLBRIGHT_LIGHT]);
    }
  });

  it("flat per-face: each face samples the neighbour cell it looks into", () => {
    const s = snapshotStride(APRON);
    const light = new Uint8Array(s * s * s);
    light.fill((5 << 4) | 0); // sky 5 everywhere by default
    // The Up neighbour (0,1,0) is bright sky 15; the East neighbour (1,0,0) carries block light 12.
    light[snapshotIndex(0, 1, 0, APRON)] = (15 << 4) | 0;
    light[snapshotIndex(1, 0, 0, APRON)] = (0 << 4) | 12;

    const out = meshSection(stoneSnapshot(light), makeProvider());
    const byNormal = lightByNormal(out);

    const up = [...byNormal.get(Direction.Up)!].map(unpackLight);
    expect(up).toHaveLength(1);
    expect(up[0]).toMatchObject({ skyLight: 15, blockLight: 0 });

    const east = [...byNormal.get(Direction.East)!].map(unpackLight);
    expect(east).toHaveLength(1);
    expect(east[0]).toMatchObject({ skyLight: 0, blockLight: 12 });

    // A face whose neighbour wasn't overridden keeps the default sky 5.
    const down = [...byNormal.get(Direction.Down)!].map(unpackLight);
    expect(down[0]).toMatchObject({ skyLight: 5, blockLight: 0 });
  });
});
