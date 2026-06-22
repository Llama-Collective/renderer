// SL-2/SL-4 end-to-end: smooth lighting reuses the EXACT geometry (position/uv/normal/colour unchanged) and
// touches ONLY the per-vertex light word — darkening the AO byte at concave corners. Flag OFF is byte-identical.

import { describe, it, expect } from "vitest";
import { meshSection, type BakedModelProvider } from "./SectionMesher";
import { ModelBaker, type BakedBlockModel, type BlockEntry } from "./model/BakedBlockModel";
import { TintProvider } from "./model/TintProvider";
import { fixtureBlockstates, fixtureProvider } from "./model/__fixtures__/load";
import { SnapshotSource, type BlockSource } from "../world/SnapshotSource";
import { sectionKey } from "../world/SectionKey";
import { DirtyReason, TerrainPass } from "../types";
import { decodeVertex } from "./VertexFormat";
import { unpackLight } from "./Lighting";

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

// A stone block at the origin with a diagonal stone neighbour at (1,1,1) → the origin block's top/side faces
// have a concave corner that smooth AO must darken.
const world: BlockSource = { getBlock: (x, y, z) => ((x === 0 && y === 0 && z === 0) || (x === 1 && y === 1 && z === 1) ? STONE : 0) };

function solidVerts(out: ReturnType<typeof meshSection>) {
  const part = out.parts[TerrainPass.Solid]!;
  const dv = new DataView(part.vertexData);
  return { dv, count: part.quadCount * 4, bytes: new Uint8Array(part.vertexData) };
}

describe("SectionMesher smooth lighting (SL-2/SL-4)", () => {
  const provider = makeProvider();
  // Smooth lighting needs the wider apron (SL-3) for corner reads.
  const snap = new SnapshotSource(world, 2).buildSnapshot(sectionKey(0, 0, 0), 1, DirtyReason.InitialLoad);

  it("flag OFF ⇒ byte-identical to flat (the baseline golden)", () => {
    const flat = solidVerts(meshSection(snap, provider));
    const off = solidVerts(meshSection(snap, provider, undefined, { smoothLighting: false }));
    expect(off.bytes).toEqual(flat.bytes);
  });

  it("flag ON ⇒ identical geometry, only the light word changes; AO darkens at the concave corner", () => {
    const flat = solidVerts(meshSection(snap, provider));
    const smooth = solidVerts(meshSection(snap, provider, undefined, { smoothLighting: true }));
    expect(smooth.count).toBe(flat.count); // same geometry shape

    let aoDarkened = 0;
    for (let i = 0; i < flat.count; i++) {
      const f = decodeVertex(flat.dv, i * 20);
      const s = decodeVertex(smooth.dv, i * 20);
      // Position / UV / normal / colour are UNTOUCHED — smooth lighting is light-only.
      expect([s.x, s.y, s.z, s.u, s.v, s.normal, s.colorRGBA]).toEqual([f.x, f.y, f.z, f.u, f.v, f.normal, f.colorRGBA]);
      expect(unpackLight(f.light!).ao).toBe(255); // flat is always fully open
      if (unpackLight(s.light!).ao < 255) aoDarkened++;
    }
    expect(aoDarkened).toBeGreaterThan(0); // the concave corner got darkened
  });

  it("is deterministic (inline == off-thread): same input ⇒ identical bytes", () => {
    const a = solidVerts(meshSection(snap, provider, undefined, { smoothLighting: true }));
    const b = solidVerts(meshSection(snap, provider, undefined, { smoothLighting: true }));
    expect(a.bytes).toEqual(b.bytes);
  });
});
