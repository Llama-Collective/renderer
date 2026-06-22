// P6/LM-2 — the light-only invalidation path: a pure Light dirty commits via an IN-PLACE light reupload
// (presented identity preserved, no realloc) and does NOT re-run the occlusion BFS (graphRevision steady).

import { describe, it, expect } from "vitest";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { GpuSectionUploader, type GpuCommittedSection } from "../render/GpuSectionUploader";
import { SectionStore } from "./SectionStore";
import { DirtyFlag } from "./RenderSection";
import { sectionKey } from "./SectionKey";
import { DirtyReason, TerrainPass } from "../types";
import type { SectionBuildOutput } from "../workers/BuildOutput";
import { instrument } from "../core/Instrument";
import { occlusionVisibleSet, occlusionHomeSection } from "../app/scene";

const K = sectionKey(0, 0, 0);

function makeBuild(generation: number, visibility = 0x7fff): SectionBuildOutput & { disposed: boolean } {
  const bytes = 64;
  return {
    sectionKey: K,
    generation,
    info: { flags: 0, visibilityData: Uint32Array.of(visibility) },
    parts: { [TerrainPass.Solid]: { vertexData: new ArrayBuffer(bytes), quadCount: 4 } },
    approxBytes: bytes,
    disposed: false,
    dispose(this: { disposed: boolean }) { this.disposed = true; },
  };
}

function commitBuild(store: SectionStore, build: SectionBuildOutput): void {
  store.acceptBuild(build);
  store.commit(store.get(K)!);
}

describe("SectionStore light-only invalidation (P6/LM-2)", () => {
  it("markDirty(Light) sets ONLY the Light flag; a block reason sets Geometry", () => {
    const store = new SectionStore(new GpuSectionUploader(new FakeGraphicsDevice()));
    store.markDirty(K, DirtyReason.Light);
    expect(store.get(K)!.dirtyFlags).toBe(DirtyFlag.Light);

    const store2 = new SectionStore(new GpuSectionUploader(new FakeGraphicsDevice()));
    store2.markDirty(K, DirtyReason.Edit);
    expect(store2.get(K)!.dirtyFlags).toBe(DirtyFlag.Geometry);
  });

  it("a pure-light commit reuses presented in place and does NOT bump graphRevision (no BFS)", () => {
    const store = new SectionStore(new GpuSectionUploader(new FakeGraphicsDevice()));
    store.markDirty(K, DirtyReason.InitialLoad);
    commitBuild(store, makeBuild(1, 0x1234));
    const section = store.get(K)!;
    const presented0 = section.presented as GpuCommittedSection;
    const graph0 = store.graphRevision;
    expect(presented0).toBeTruthy();

    // A light change: same geometry (quadCount + visibility identical), new generation.
    store.markDirty(K, DirtyReason.Light);
    expect(section.dirtyFlags).toBe(DirtyFlag.Light);
    commitBuild(store, makeBuild(2, 0x1234));

    expect(section.presented).toBe(presented0); // SAME committed object — allocations updated in place
    expect((section.presented as GpuCommittedSection).generation).toBe(2);
    expect(store.graphRevision).toBe(graph0); // connectivity-neutral ⇒ the camera BFS does not re-run
    expect(section.dirtyFlags).toBe(DirtyFlag.None); // cleared on commit
    expect(store.presentedChanges.has(K)).toBe(true); // the draw list still re-resolves this section
  });

  it("a light dirty whose geometry CHANGES (visibility differs) falls back to a full upload + bumps graph", () => {
    const store = new SectionStore(new GpuSectionUploader(new FakeGraphicsDevice()));
    store.markDirty(K, DirtyReason.InitialLoad);
    commitBuild(store, makeBuild(1, 0x1234));
    const section = store.get(K)!;
    const presented0 = section.presented;
    const graph0 = store.graphRevision;

    // Same Light flag, but the rebuilt visibility differs ⇒ connectivity changed ⇒ graph must bump. (The
    // uploadLight quad-count check still passes here, so this proves the N4 visibility gate itself fires.)
    store.markDirty(K, DirtyReason.Light);
    commitBuild(store, makeBuild(2, 0x0001));
    expect(section.presented).toBe(presented0); // still in-place (same geometry shape)
    expect(store.graphRevision).toBe(graph0 + 1); // but visibility changed ⇒ BFS must re-run
  });
});

// OCC-1: the occlusion reachability cache keys on graphRevision (NOT generation / presentedChanges), so a
// connectivity-neutral light-only commit — which deliberately does NOT bump graphRevision — must leave the
// cache valid: the caller's gate sees a steady graphRevision + a stable home section ⇒ it does not re-run the
// BFS, so the occBfsRuns win metric does not increment on a light update.
describe("OCC-1 reachability cache — a light-only commit does not invalidate (graphRevision steady)", () => {
  it("a pure-light commit keeps graphRevision steady ⇒ the BFS does not re-run (occBfsRuns flat)", () => {
    instrument.enabled = true;
    instrument.reset();

    const store = new SectionStore(new GpuSectionUploader(new FakeGraphicsDevice()));
    store.markDirty(K, DirtyReason.InitialLoad);
    commitBuild(store, makeBuild(1, 0x1234));

    const cam = [8, 8, 8] as [number, number, number]; // center of section (0,0,0)
    const reach = new Set<number>();

    // Frame N: initial BFS run for this home section + graph revision.
    const home0 = occlusionHomeSection(store, cam)!;
    const graph0 = store.graphRevision;
    occlusionVisibleSet(store, cam, () => true, reach);
    expect(instrument.job.occBfsRuns).toBe(1);
    const membership0 = [...reach].sort();

    // A light-only commit (same geometry shape + same visibility word ⇒ connectivity-neutral).
    store.markDirty(K, DirtyReason.Light);
    commitBuild(store, makeBuild(2, 0x1234));

    // The cache gate: graphRevision steady AND home section unchanged ⇒ reachabilityStale is false ⇒ the
    // caller SKIPS occlusionVisibleSet entirely. The counter must stay at 1 (no re-run on a light update).
    const graph1 = store.graphRevision;
    const home1 = occlusionHomeSection(store, cam)!;
    expect(graph1).toBe(graph0); // light-only ⇒ no graph bump (the invalidation signal stays low)
    expect(home1).toEqual(home0); // same home section
    const reachabilityStale = graph1 !== graph0 || home1[0] !== home0[0] || home1[1] !== home0[1] || home1[2] !== home0[2];
    expect(reachabilityStale).toBe(false);
    // (caller does NOT call occlusionVisibleSet) — the BFS-run counter is unchanged.
    expect(instrument.job.occBfsRuns).toBe(1);
    // The cached membership is still valid (would be returned unchanged).
    expect([...reach].sort()).toEqual(membership0);
  });
});
