// Stable Presentation Invariant tests — T-PRES-1..5 + the GATE 4.1 cases.
// RENDERER_PLAN.md §4, §5. These must stay green forever (§22 REVIEW GATE 22.1).
//
// Exercises the real GpuSectionUploader against FakeGraphicsDevice so "old buffers freed
// only after a current-gen replacement commits" is observable via liveBufferCount().

import { describe, it, expect } from "vitest";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { GpuSectionUploader, type GpuCommittedSection } from "../render/GpuSectionUploader";
import { SectionStore, AcceptResult } from "./SectionStore";
import { sectionKey } from "./SectionKey";
import { DirtyReason, TerrainPass } from "../types";
import { SortType } from "../mesh/SortTypes";
import type { SectionBuildOutput, TranslucentMeshPart } from "../workers/BuildOutput";
import type { SortOutput } from "../workers/SortOutput";
import { VIS_ALL } from "./SectionVisibility";

const K = sectionKey(0, 0, 0);

function setup(budgetBytes?: number) {
  const device = new FakeGraphicsDevice();
  const uploader = new GpuSectionUploader(device, budgetBytes);
  const store = new SectionStore(uploader);
  return { device, uploader, store };
}

type TestBuild = SectionBuildOutput & { disposed: boolean };

function makeBuild(
  generation: number,
  opts: { translucent?: boolean; empty?: boolean; bytes?: number; visibilityData?: Uint32Array } = {},
  key = K,
): TestBuild {
  const bytes = opts.bytes ?? 64;
  const empty = opts.empty ?? false;
  const parts: SectionBuildOutput["parts"] = {};
  let translucent: TranslucentMeshPart | undefined;
  if (!empty) {
    parts[TerrainPass.Solid] = { vertexData: new ArrayBuffer(bytes), quadCount: 4 };
    if (opts.translucent) {
      translucent = {
        vertexData: new ArrayBuffer(bytes),
        quadCount: 2,
        sortType: SortType.StaticNormal,
        indexData: new ArrayBuffer(24),
        quadHash: 123,
      };
    }
  }
  return {
    sectionKey: key,
    generation,
    info: { flags: 0, visibilityData: opts.visibilityData },
    parts,
    translucent,
    approxBytes: empty ? 0 : bytes,
    disposed: false,
    dispose(this: TestBuild) {
      this.disposed = true;
    },
  };
}

type TestSort = SortOutput & { disposed: boolean };

function makeSort(generation: number): TestSort {
  return {
    sectionKey: K,
    generation,
    indexData: new ArrayBuffer(24),
    disposed: false,
    dispose(this: TestSort) {
      this.disposed = true;
    },
  };
}

describe("SectionStore.sectionBounds — populated-region AABB (occlusion traversal bound)", () => {
  /** markDirty + accept + commit a build at a section (geometry by default, empty optional). */
  function commit(store: SectionStore, kx: number, ky: number, kz: number, empty = false) {
    const key = sectionKey(kx, ky, kz);
    const s = store.getOrCreate(key);
    store.markDirty(key, DirtyReason.InitialLoad);
    store.acceptBuild(makeBuild(s.generation, { empty }, key));
    store.commit(s);
  }

  it("is null for an empty store (no drawable geometry ⇒ the BFS short-circuits)", () => {
    const { store } = setup();
    expect(store.sectionBounds()).toBeNull();
  });

  it("is the inclusive AABB over drawable sections (incl. negatives)", () => {
    const { store } = setup();
    commit(store, 2, 3, 4);
    commit(store, -1, 1, 6);
    expect(store.sectionBounds()).toEqual({ lo: [-1, 1, 4], hi: [2, 3, 6] });
  });

  it("ignores empty (no-geometry) sections — they add nothing to draw", () => {
    const { store } = setup();
    commit(store, 0, 0, 0);
    commit(store, 9, 9, 9, /* empty */ true); // committed but carries no geometry
    expect(store.sectionBounds()).toEqual({ lo: [0, 0, 0], hi: [0, 0, 0] });
  });

  it("shrinks when a drawable section is disposed (graphRevision bumps ⇒ cache refreshes)", () => {
    const { store } = setup();
    commit(store, 0, 0, 0);
    commit(store, 5, 0, 0);
    expect(store.sectionBounds()).toEqual({ lo: [0, 0, 0], hi: [5, 0, 0] });
    store.dispose(sectionKey(5, 0, 0));
    expect(store.sectionBounds()).toEqual({ lo: [0, 0, 0], hi: [0, 0, 0] });
  });
});

describe("SectionStore.getByCoords — S1 numeric BFS lookup mirror", () => {
  it("returns the same section as get(sectionKey(...)) for created sections, incl. negatives", () => {
    const { store } = setup();
    for (const [x, y, z] of [[0, 0, 0], [2, -3, 4], [-7, 11, -1]]) {
      const s = store.getOrCreate(sectionKey(x, y, z));
      expect(store.getByCoords(x, y, z)).toBe(s);
      expect(store.getByCoords(x, y, z)).toBe(store.get(sectionKey(x, y, z)));
    }
  });

  it("is undefined before create and after dispose (mirror stays in sync)", () => {
    const { store } = setup();
    expect(store.getByCoords(3, 4, 5)).toBeUndefined();
    store.getOrCreate(sectionKey(3, 4, 5));
    expect(store.getByCoords(3, 4, 5)).toBeDefined();
    store.dispose(sectionKey(3, 4, 5));
    expect(store.getByCoords(3, 4, 5)).toBeUndefined();
  });
});

describe("SectionStore — happy path (GATE 4.1c)", () => {
  it("commits a current-generation build and replaces presented", () => {
    const { device, store } = setup();
    const s = store.getOrCreate(K);
    store.markDirty(K, DirtyReason.InitialLoad); // generation -> 1
    const out = makeBuild(1);

    expect(store.acceptBuild(out)).toBe(AcceptResult.Accepted);
    expect(s.presented).toBeNull(); // not presented until committed

    expect(store.commit(s)).toBe(true);
    expect(s.presented).not.toBeNull();
    expect(s.lastBuiltGeneration).toBe(1);
    expect(out.disposed).toBe(true); // output consumed + disposed after commit
    expect(device.liveBufferCount()).toBe(1); // one solid vertex buffer
  });
});

describe("Stable Presentation Invariant", () => {
  it("T-PRES-1: starting a new job (re-dirty) never mutates presented before commit", () => {
    const { device, store } = setup();
    const s = store.getOrCreate(K);
    store.markDirty(K, DirtyReason.InitialLoad);
    store.acceptBuild(makeBuild(1));
    store.commit(s);

    const presentedBefore = s.presented;
    const liveBefore = device.liveBufferCount();

    // Re-dirty (a new job starts) — the store bumps generation but MUST NOT touch `presented` until the
    // next commit (the Stable Presentation Invariant; job lifecycle itself lives in the Scheduler now).
    store.markDirty(K, DirtyReason.Simulation);

    expect(s.presented).toBe(presentedBefore);
    expect(device.liveBufferCount()).toBe(liveBefore);
  });

  it("T-PRES-2: stale build/sort outputs are discarded, disposed, presentation untouched", () => {
    const { device, store } = setup();
    const s = store.getOrCreate(K);
    store.markDirty(K, DirtyReason.InitialLoad); // gen 1
    store.acceptBuild(makeBuild(1, { translucent: true }));
    store.commit(s);

    const presentedBefore = s.presented;
    const liveBefore = device.liveBufferCount();

    store.markDirty(K, DirtyReason.Edit); // gen 2 -> gen-1 outputs are now stale

    const staleBuild = makeBuild(1);
    expect(store.acceptBuild(staleBuild)).toBe(AcceptResult.DiscardedStale);
    expect(staleBuild.disposed).toBe(true);

    const staleSort = makeSort(1);
    expect(store.acceptSort(staleSort)).toBe(AcceptResult.DiscardedStale);
    expect(staleSort.disposed).toBe(true);

    expect(s.presented).toBe(presentedBefore);
    expect(device.liveBufferCount()).toBe(liveBefore); // nothing created or freed
  });

  it("T-PRES-3: an empty result clears geometry only when its generation is current", () => {
    const { device, store } = setup();
    const s = store.getOrCreate(K);
    store.markDirty(K, DirtyReason.InitialLoad); // gen 1
    store.acceptBuild(makeBuild(1));
    store.commit(s);
    expect(device.liveBufferCount()).toBe(1);

    // A STALE empty result must NOT clear.
    store.markDirty(K, DirtyReason.Edit); // gen 2
    const staleEmpty = makeBuild(1, { empty: true });
    expect(store.acceptBuild(staleEmpty)).toBe(AcceptResult.DiscardedStale);
    expect(s.presented).not.toBeNull();
    expect(device.liveBufferCount()).toBe(1);

    // A CURRENT empty result clears (and frees old buffers).
    const currentEmpty = makeBuild(2, { empty: true });
    expect(store.acceptBuild(currentEmpty)).toBe(AcceptResult.Accepted);
    expect(store.commit(s)).toBe(true);
    expect(s.presented).not.toBeNull(); // an empty CommittedSection, not null
    // Old geometry's vertex ALLOCATION is freed within the pooled arena; the arena buffer itself is
    // retained for reuse (§11), so liveBufferCount stays 1 (the solid arena) rather than dropping to 0.
    expect((s.presented as GpuCommittedSection).vertexAlloc[TerrainPass.Solid]).toBeUndefined();
    expect(device.liveBufferCount()).toBe(1);
  });

  it("T-PRES-4: when the upload budget is exhausted, old data is kept and retried next frame", () => {
    const { device, uploader, store } = setup();
    const s = store.getOrCreate(K);
    store.markDirty(K, DirtyReason.InitialLoad);
    store.acceptBuild(makeBuild(1));
    store.commit(s); // presented P1

    const presentedBefore = s.presented;
    const liveBefore = device.liveBufferCount(); // 1

    store.markDirty(K, DirtyReason.Edit); // gen 2
    const out = makeBuild(2, { bytes: 128 });
    expect(store.acceptBuild(out)).toBe(AcceptResult.Accepted);

    // Frame 1: no budget.
    uploader.beginFrame(0);
    expect(store.commit(s)).toBe(false);
    expect(s.presented).toBe(presentedBefore); // unchanged
    expect(s.pendingBuild).toBe(out); // retained for retry
    expect(out.disposed).toBe(false);
    expect(device.liveBufferCount()).toBe(liveBefore);

    // Frame 2: budget restored.
    uploader.beginFrame();
    expect(store.commit(s)).toBe(true);
    expect(s.presented).not.toBe(presentedBefore);
    expect(out.disposed).toBe(true);
    expect(device.liveBufferCount()).toBe(1); // old freed, new uploaded
  });

  it("OCC-2: a 4-element visibilityData stores the union into section.visibility AND the raw sets, no graphRev bump on a connectivity-neutral remesh", () => {
    const { store } = setup();
    const s = store.getOrCreate(K);
    store.markDirty(K, DirtyReason.InitialLoad); // gen 1
    // Union [0] = 0b101; the per-quadrant tail differs from [0] (a true refinement).
    const sets1 = Uint32Array.of(0b101, 0b101, 0b100, 0b001);
    store.acceptBuild(makeBuild(1, { visibilityData: sets1 }));
    store.commit(s);
    expect(s.visibility).toBe(0b101); // union word stored on .visibility (off-readers unaffected)
    expect(s.visibilitySets).toBe(sets1); // raw 4-element sets retained for the camera-quadrant join
    const revAfterFirst = store.graphRevision;

    // A connectivity-neutral remesh: SAME union word [0], but a DIFFERENT raw-sets tail (a lamp toggling in a
    // still-solid section recomputes the same union). graphRev must NOT bump (TRAP OCC-2.C — compare the union
    // word, never the raw sets array), so a still-connectivity-stable remesh does not re-trigger the camera BFS.
    store.markDirty(K, DirtyReason.Edit); // gen 2
    const sets2 = Uint32Array.of(0b101, 0b001, 0b100, 0b101); // same [0]=0b101, tail reshuffled
    store.acceptBuild(makeBuild(2, { visibilityData: sets2 }));
    store.commit(s);
    expect(s.visibility).toBe(0b101);
    expect(s.visibilitySets).toBe(sets2);
    expect(store.graphRevision).toBe(revAfterFirst); // NO bump — union word unchanged

    // A 1-element (off-path) build clears visibilitySets back to undefined and behaves exactly as OCC-1.
    store.markDirty(K, DirtyReason.Edit); // gen 3
    store.acceptBuild(makeBuild(3, { visibilityData: Uint32Array.of(0b101) }));
    store.commit(s);
    expect(s.visibility).toBe(0b101);
    expect(s.visibilitySets).toBeUndefined();
    expect(store.graphRevision).toBe(revAfterFirst); // still no bump (same union)

    // An absent visibilityData falls back to VIS_ALL exactly as the legacy off-path (no sets).
    store.markDirty(K, DirtyReason.Edit); // gen 4
    store.acceptBuild(makeBuild(4)); // no visibilityData
    store.commit(s);
    expect(s.visibility).toBe(VIS_ALL); // union changed 0b101 → VIS_ALL ⇒ this commit DOES bump
    expect(s.visibilitySets).toBeUndefined();
    expect(store.graphRevision).toBe(revAfterFirst + 1);
  });

  it("T-PRES-5: a sort rewrites indices only — vertices and geometry are untouched", () => {
    const { device, store } = setup();
    const s = store.getOrCreate(K);
    store.markDirty(K, DirtyReason.InitialLoad);
    store.acceptBuild(makeBuild(1, { translucent: true }));
    store.commit(s);

    const before = s.presented as GpuCommittedSection;
    const vtxBefore = before.vertexAlloc[TerrainPass.Translucent]; // stable arena allocation id (§11)
    const idxAllocBefore = before.indexAlloc[TerrainPass.Translucent]; // stable index-arena allocation id
    const liveBefore = device.liveBufferCount(); // solid arena + translucent vertex arena + index arena = 3

    expect(store.acceptSort(makeSort(1))).toBe(AcceptResult.Accepted);
    expect(store.commit(s)).toBe(true);

    const after = s.presented as GpuCommittedSection;
    expect(after.vertexAlloc[TerrainPass.Translucent]).toBe(vtxBefore); // vertices reused (same allocation)
    // U6: a same-size re-sort updates the index allocation IN PLACE — same id, contents rewritten, no churn.
    expect(after.indexAlloc[TerrainPass.Translucent]).toBe(idxAllocBefore);
    expect(after.quadCount[TerrainPass.Translucent]).toBe(before.quadCount[TerrainPass.Translucent]);
    expect(device.liveBufferCount()).toBe(liveBefore); // no buffer freed or created (in-place update)
  });
});
