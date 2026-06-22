// UploadScheduler — per-frame budget, deferral, draining, aging. RENDERER_PLAN.md §11, §16.

import { describe, it, expect } from "vitest";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { GpuSectionUploader } from "./GpuSectionUploader";
import { UploadScheduler } from "./UploadScheduler";
import { SectionStore } from "../world/SectionStore";
import { sectionKey, type SectionKey } from "../world/SectionKey";
import { DirtyReason, TerrainPass } from "../types";
import type { SectionBuildOutput } from "../workers/BuildOutput";
import type { SortOutput } from "../workers/SortOutput";

function setup() {
  const device = new FakeGraphicsDevice();
  const store = new SectionStore(new GpuSectionUploader(device)); // unbounded uploader; scheduler gates
  return { device, store };
}

describe("UploadScheduler single-budget authority (INFRA-5)", () => {
  it("accepts an unbounded (default Infinity) uploader", () => {
    const store = new SectionStore(new GpuSectionUploader(new FakeGraphicsDevice()));
    expect(() => new UploadScheduler(store)).not.toThrow();
  });

  it("rejects a scheduler-driven uploader that has its own finite budget", () => {
    const store = new SectionStore(new GpuSectionUploader(new FakeGraphicsDevice(), 10_000));
    expect(() => new UploadScheduler(store)).toThrow(/single upload-budget authority|INFRA-5/);
  });
});

type TestBuild = SectionBuildOutput & { disposed: boolean };

function makeBuild(key: SectionKey, generation: number, bytes = 64): TestBuild {
  return {
    sectionKey: key,
    generation,
    info: { flags: 0 },
    parts: { [TerrainPass.Solid]: { vertexData: new ArrayBuffer(bytes), quadCount: 4 } },
    approxBytes: bytes,
    disposed: false,
    dispose(this: TestBuild) {
      this.disposed = true;
    },
  };
}

function makeSort(key: SectionKey, generation: number): SortOutput & { disposed: boolean } {
  return {
    sectionKey: key,
    generation,
    indexData: new ArrayBuffer(24),
    disposed: false,
    dispose() {
      (this as { disposed: boolean }).disposed = true;
    },
  };
}

const A = sectionKey(0, 0, 0);
const B = sectionKey(1, 0, 0);
const C = sectionKey(2, 0, 0);

function dirtyAt(store: SectionStore, key: SectionKey): void {
  store.markDirty(key, DirtyReason.InitialLoad);
}

describe("UploadScheduler", () => {
  it("drains and commits all pending sections within budget", () => {
    const { store } = setup();
    const sched = new UploadScheduler(store, { now: () => 0 });
    [A, B, C].forEach((k) => dirtyAt(store, k));
    sched.submitBuild(makeBuild(A, 1));
    sched.submitBuild(makeBuild(B, 1));
    sched.submitBuild(makeBuild(C, 1));

    const stats = sched.drainAndCommit(0);

    expect(stats.committed).toBe(3);
    expect(stats.discardedStale).toBe(0);
    expect(sched.pendingCount).toBe(0);
    expect(store.get(A)!.presented).not.toBeNull();
    expect(store.get(B)!.presented).not.toBeNull();
    expect(store.get(C)!.presented).not.toBeNull();
  });

  it("pendingCount reflects staged-but-undrained inbox items (render-on-demand liveness)", () => {
    const { store } = setup();
    const sched = new UploadScheduler(store, { now: () => 0 });
    [A, B].forEach((k) => dirtyAt(store, k));
    // A worker reply stages into the inbox WITHOUT a drain. Before this counted, pendingCount was 0 here, so
    // the dynamic-FPS loop would idle and never run drainAndCommit — the build would never draw.
    sched.submitBuild(makeBuild(A, 1));
    sched.submitBuild(makeBuild(B, 1));
    expect(sched.pendingCount).toBe(2); // inbox.length (undrained), even though `pending` is still empty
    sched.drainAndCommit(0);
    expect(sched.pendingCount).toBe(0); // committed → both inbox and pending now empty
  });

  it("discards stale outputs while draining (no commit, output disposed)", () => {
    const { store } = setup();
    const sched = new UploadScheduler(store, { now: () => 0 });
    dirtyAt(store, A); // gen 1
    store.markDirty(A, DirtyReason.Edit); // gen 2 -> a gen-1 output is stale
    const stale = makeBuild(A, 1);
    sched.submitBuild(stale);

    const stats = sched.drainAndCommit(0);

    expect(stats.discardedStale).toBe(1);
    expect(stats.committed).toBe(0);
    expect(stale.disposed).toBe(true);
    expect(store.get(A)!.presented).toBeNull();
  });

  it("setBudget retunes the per-frame budget at runtime (ABP-1)", () => {
    const { store } = setup();
    const sched = new UploadScheduler(store, { now: () => 0 }); // default 4 MB budget
    [A, B].forEach((k) => dirtyAt(store, k));
    sched.submitBuild(makeBuild(A, 1, 64));
    sched.submitBuild(makeBuild(B, 1, 64));
    // Tighten the byte budget BEFORE the drain — only the emergency-first commit gets through.
    sched.setBudget({ maxBytesPerFrame: 100 });
    expect(sched.currentBudget.maxBytesPerFrame).toBe(100);
    const f1 = sched.drainAndCommit(1);
    expect(f1.committed).toBe(1);
    expect(f1.deferredBudget).toBe(1); // B deferred under the tightened budget
    // Open the budget back up → B commits next frame.
    sched.setBudget({ maxBytesPerFrame: 4 * 1024 * 1024 });
    expect(sched.drainAndCommit(2).committed).toBe(1);
    expect(sched.pendingCount).toBe(0);
  });

  it("defers over-byte-budget sections and commits them on a later frame", () => {
    const { store } = setup();
    let deferredReported = 0;
    const sched = new UploadScheduler(store, {
      now: () => 0, // time never a factor
      budget: { maxBytesPerFrame: 100, maxMillisPerFrame: 1000 },
      onDeferred: () => deferredReported++,
    });
    [A, B].forEach((k) => dirtyAt(store, k));
    sched.submitBuild(makeBuild(A, 1, 64));
    sched.submitBuild(makeBuild(B, 1, 64));

    // Frame 1: A commits (first always allowed), B exceeds remaining 36 -> deferred.
    const f1 = sched.drainAndCommit(1);
    expect(f1.committed).toBe(1);
    expect(f1.deferredBudget).toBe(1);
    expect(deferredReported).toBe(1);
    expect(store.get(A)!.presented).not.toBeNull();
    expect(store.get(B)!.presented).toBeNull(); // old (none) retained, not blanked-wrongly
    expect(store.get(B)!.pendingBuild).not.toBeNull(); // still staged
    expect(sched.pendingCount).toBe(1);

    // Frame 2: B now commits.
    const f2 = sched.drainAndCommit(2);
    expect(f2.committed).toBe(1);
    expect(store.get(B)!.presented).not.toBeNull();
    expect(sched.pendingCount).toBe(0);
  });

  it("defers remaining sections when the time budget is exceeded", () => {
    const { store } = setup();
    // now() sequence: frameStart=0, check A -> 0 (ok), check B -> 5 (>=2, defer rest)
    const times = [0, 0, 5];
    let i = 0;
    const sched = new UploadScheduler(store, {
      now: () => times[Math.min(i++, times.length - 1)],
      budget: { maxBytesPerFrame: Infinity, maxMillisPerFrame: 2 },
    });
    [A, B].forEach((k) => dirtyAt(store, k));
    sched.submitBuild(makeBuild(A, 1));
    sched.submitBuild(makeBuild(B, 1));

    const stats = sched.drainAndCommit(0);

    expect(stats.committed).toBe(1);
    expect(stats.deferredTime).toBe(1);
    expect(store.get(A)!.presented).not.toBeNull();
    expect(store.get(B)!.presented).toBeNull();
    expect(sched.pendingCount).toBe(1);
  });

  it("applies a submitted sort after the build is committed", () => {
    const { store } = setup();
    const sched = new UploadScheduler(store, { now: () => 0 });
    dirtyAt(store, A);
    sched.submitBuild(makeBuild(A, 1));
    sched.drainAndCommit(0);
    expect(store.get(A)!.presented).not.toBeNull();

    sched.submitSort(makeSort(A, 1));
    const stats = sched.drainAndCommit(1);
    expect(stats.committed).toBe(1);
    expect(store.get(A)!.pendingSort).toBeNull();
  });

  it("ages deferred work so a busy queue cannot starve it (TRAP 16.B)", () => {
    const { store } = setup();
    // Budget fits exactly one 64-byte commit per frame.
    const sched = new UploadScheduler(store, {
      now: () => 0,
      budget: { maxBytesPerFrame: 64, maxMillisPerFrame: 1000 },
    });
    dirtyAt(store, A);
    sched.submitBuild(makeBuild(A, 1));
    sched.drainAndCommit(0); // A waiting since frame 0

    // New higher-volume work arrives every frame; A must still get committed via aging.
    let committedA = false;
    for (let frame = 1; frame <= 3 && !committedA; frame++) {
      dirtyAt(store, sectionKey(10 + frame, 0, 0));
      sched.submitBuild(makeBuild(sectionKey(10 + frame, 0, 0), 1));
      sched.drainAndCommit(frame);
      committedA = store.get(A)!.presented !== null;
    }
    expect(committedA).toBe(true);
  });
});
