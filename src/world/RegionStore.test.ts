// MEM-1 — RegionStore: per-region AABBs + live counts maintained across the two SectionStore mutation
// sites (getOrCreate add / dispose remove), the count→0 contiguous-reclaim hook firing EXACTLY ONCE, and
// `regionRev` bumping ONLY on a drawable-presence change (mirror of graphRev). Driven through a real
// SectionStore so the wiring (the hooks behind the optional `regions?` injection) is what's under test.

import { describe, it, expect } from "vitest";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { GpuSectionUploader } from "../render/GpuSectionUploader";
import { SectionStore } from "./SectionStore";
import { RegionStore } from "./RegionStore";
import { packRegionOfSection, packRegionCoord } from "./RegionKey";
import { sectionKey, type SectionKey } from "./SectionKey";
import { DirtyReason, TerrainPass } from "../types";
import { makeBuildOutput } from "../workers/BuildOutput";

function setup(onRegionEmpty?: (k: number) => void) {
  const device = new FakeGraphicsDevice();
  const uploader = new GpuSectionUploader(device);
  const regions = new RegionStore(onRegionEmpty);
  const store = new SectionStore(uploader, regions);
  return { device, uploader, regions, store };
}

/** markDirty → accept → commit a solid build (drawable geometry) at a section. */
function commitSolid(store: SectionStore, key: SectionKey, bytes = 64): void {
  const gen = store.markDirty(key, DirtyReason.InitialLoad);
  const build = makeBuildOutput({ sectionKey: key, generation: gen, info: { flags: 0 }, parts: { [TerrainPass.Solid]: { vertexData: new ArrayBuffer(bytes), quadCount: 1 } } });
  store.acceptBuild(build);
  store.commit(store.get(key)!);
}

/** Commit an EMPTY build (no geometry) — section exists + presented cleared (drawable → not drawable). */
function commitEmpty(store: SectionStore, key: SectionKey): void {
  const gen = store.markDirty(key, DirtyReason.Edit);
  const build = makeBuildOutput({ sectionKey: key, generation: gen, info: { flags: 0 }, parts: {} });
  store.acceptBuild(build);
  store.commit(store.get(key)!);
}

describe("RegionStore — per-region AABB + live count across getOrCreate / dispose", () => {
  it("registers a section into its region with a single-section inclusive AABB", () => {
    const { store, regions } = setup();
    commitSolid(store, sectionKey(2, 1, 3)); // region (0,0,0)
    const r = regions.get(packRegionOfSection(2, 1, 3))!;
    expect(r).toBeDefined();
    expect([r.loX, r.loY, r.loZ]).toEqual([2, 1, 3]);
    expect([r.hiX, r.hiY, r.hiZ]).toEqual([2, 1, 3]);
    expect(r.liveCount).toBe(1);
    expect(r.drawableCount).toBe(1);
  });

  it("extends a region's AABB as more sections of the SAME region are added", () => {
    const { store, regions } = setup();
    commitSolid(store, sectionKey(1, 0, 1)); // all in region (0,0,0): x∈[0,8) y∈[0,4) z∈[0,8)
    commitSolid(store, sectionKey(6, 2, 5));
    commitSolid(store, sectionKey(0, 3, 7));
    const r = regions.get(packRegionOfSection(1, 0, 1))!;
    expect([r.loX, r.loY, r.loZ]).toEqual([0, 0, 1]);
    expect([r.hiX, r.hiY, r.hiZ]).toEqual([6, 3, 7]);
    expect(r.liveCount).toBe(3);
    expect(regions.regionCount).toBe(1); // all one region
  });

  it("separates sections that fall into DIFFERENT regions", () => {
    const { store, regions } = setup();
    commitSolid(store, sectionKey(0, 0, 0)); // region (0,0,0)
    commitSolid(store, sectionKey(8, 0, 0)); // region (1,0,0) — x span is 8
    expect(regions.regionCount).toBe(2);
    expect(regions.get(packRegionCoord(0, 0, 0))!.liveCount).toBe(1);
    expect(regions.get(packRegionCoord(1, 0, 0))!.liveCount).toBe(1);
  });

  it("decrements live count on dispose; the region survives while other sections remain", () => {
    const { store, regions } = setup();
    commitSolid(store, sectionKey(0, 0, 0));
    commitSolid(store, sectionKey(1, 0, 0)); // same region
    store.dispose(sectionKey(0, 0, 0));
    const r = regions.get(packRegionCoord(0, 0, 0))!;
    expect(r.liveCount).toBe(1);
    expect(regions.regionCount).toBe(1);
  });
});

describe("RegionStore — contiguous-reclaim hook fires EXACTLY ONCE on count→0", () => {
  it("fires onRegionEmpty once when a region's LAST section is disposed", () => {
    const fired: number[] = [];
    const { store } = setup((k) => fired.push(k));
    commitSolid(store, sectionKey(0, 0, 0));
    commitSolid(store, sectionKey(1, 0, 0)); // same region (0,0,0)
    store.dispose(sectionKey(0, 0, 0)); // count 2 → 1: no fire yet
    expect(fired).toEqual([]);
    store.dispose(sectionKey(1, 0, 0)); // count 1 → 0: fire once
    expect(fired).toEqual([packRegionCoord(0, 0, 0)]);
  });

  it("does not fire twice if a section in an already-emptied region is re-disposed", () => {
    const fired: number[] = [];
    const { store } = setup((k) => fired.push(k));
    commitSolid(store, sectionKey(0, 0, 0));
    store.dispose(sectionKey(0, 0, 0)); // → 0, fires once, region dropped
    store.dispose(sectionKey(0, 0, 0)); // already gone — no section, no-op
    expect(fired).toEqual([packRegionCoord(0, 0, 0)]);
  });

  it("fires independently per region", () => {
    const fired: number[] = [];
    const { store } = setup((k) => fired.push(k));
    commitSolid(store, sectionKey(0, 0, 0)); // region (0,0,0)
    commitSolid(store, sectionKey(8, 0, 0)); // region (1,0,0)
    store.dispose(sectionKey(0, 0, 0));
    store.dispose(sectionKey(8, 0, 0));
    expect(fired.sort()).toEqual([packRegionCoord(0, 0, 0), packRegionCoord(1, 0, 0)].sort());
  });
});

describe("RegionStore — regionRev bumps ONLY on drawable-presence change", () => {
  it("bumps when a section first gains geometry (commit), not on getOrCreate alone", () => {
    const { store, regions } = setup();
    const rev0 = regions.regionRev;
    store.markDirty(sectionKey(0, 0, 0), DirtyReason.Edit); // getOrCreate runs — section added, NOT drawable
    expect(regions.regionRev).toBe(rev0); // no drawable presence yet ⇒ no bump
    commitSolid(store, sectionKey(0, 0, 0)); // gains geometry ⇒ one bump
    expect(regions.regionRev).toBe(rev0 + 1);
  });

  it("does NOT bump on a connectivity-neutral re-mesh that stays drawable", () => {
    const { store, regions } = setup();
    commitSolid(store, sectionKey(0, 0, 0)); // +1
    const rev1 = regions.regionRev;
    commitSolid(store, sectionKey(0, 0, 0)); // re-mesh, still drawable — drawable presence unchanged
    expect(regions.regionRev).toBe(rev1); // no bump (mirror of graphRev discipline)
  });

  it("bumps when a drawn section is emptied to no geometry (presence flip true→false)", () => {
    const { store, regions } = setup();
    commitSolid(store, sectionKey(0, 0, 0)); // +1
    const rev1 = regions.regionRev;
    commitEmpty(store, sectionKey(0, 0, 0)); // drawable → not drawable
    expect(regions.regionRev).toBe(rev1 + 1);
  });

  it("bumps when a drawn section is disposed (presence flip via removal)", () => {
    const { store, regions } = setup();
    commitSolid(store, sectionKey(0, 0, 0)); // +1
    const rev1 = regions.regionRev;
    store.dispose(sectionKey(0, 0, 0)); // a drawn section vanished
    expect(regions.regionRev).toBe(rev1 + 1);
  });

  it("does NOT bump when a never-drawn (empty) section is disposed", () => {
    const { store, regions } = setup();
    store.markDirty(sectionKey(0, 0, 0), DirtyReason.Edit); // added, never committed geometry
    const rev0 = regions.regionRev;
    store.dispose(sectionKey(0, 0, 0)); // not drawable at removal ⇒ no presence change
    expect(regions.regionRev).toBe(rev0);
  });
});

describe("RegionStore — DEFAULT-OFF (no injection) is a pure no-op", () => {
  it("a SectionStore with no RegionStore exposes no region tier and behaves unchanged", () => {
    const store = new SectionStore(new GpuSectionUploader(new FakeGraphicsDevice()));
    expect(store.regionStore).toBeUndefined();
    commitSolid(store, sectionKey(0, 0, 0)); // exercises the same hooks, all `?.` no-ops
    store.dispose(sectionKey(0, 0, 0)); // no throw, no region bookkeeping
    expect(store.regionStore).toBeUndefined();
  });
});
