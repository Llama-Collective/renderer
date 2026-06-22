import { describe, it, expect } from "vitest";
import { Scheduler } from "./Scheduler";
import { SectionStore } from "../world/SectionStore";
import { GpuSectionUploader } from "../render/GpuSectionUploader";
import { UploadScheduler } from "../render/UploadScheduler";
import { SnapshotSource, type BlockSource } from "../world/SnapshotSource";
import { FakeGraphicsDevice } from "../core/testing/FakeGraphicsDevice";
import { sectionKey } from "../world/SectionKey";
import { DirtyReason, TerrainPass } from "../types";
import type { BakedModelProvider } from "../mesh/SectionMesher";
import { WorkerPool } from "../workers/WorkerPool";
import { makeBuildOutput, serializeBuildOutput } from "../workers/BuildOutput";
import type { SectionSnapshot } from "../world/SnapshotSource";

// Everything is block id 1 → every section's core is non-air (so isAllAirCore doesn't skip), but the
// provider returns no geometry → cheap empty meshes. We assert on DISPATCH counts, not pixels.
const blocks: BlockSource = { getBlock: () => 1 };
const provider: BakedModelProvider = { get: () => null };

function makeScheduler(meshBudgetMs: number, now?: () => number) {
  const device = new FakeGraphicsDevice();
  const store = new SectionStore(new GpuSectionUploader(device));
  const uploads = new UploadScheduler(store); // generous default budget → never the limiter here
  const snapshots = new SnapshotSource(blocks);
  const sched = new Scheduler({ store, snapshots, provider, uploads, meshBudgetMs, now });
  return { sched, store };
}

describe("Scheduler (N1 coalescing + SCHED-1 supersede + SCHED-3 budget)", () => {
  it("N1: re-dirtying a queued section collapses to ONE mesh job", () => {
    const { sched } = makeScheduler(1000);
    const k = sectionKey(0, 0, 0);
    for (let i = 0; i < 5; i++) sched.markDirty(k, DirtyReason.Simulation);
    expect(sched.pending).toBe(1); // 5 dirties → 1 queued task
    expect(sched.stats.coalesced).toBe(4);
    const { meshed } = sched.tick(0);
    expect(meshed).toBe(1); // dispatched once, not five times
    expect(sched.pending).toBe(0);
  });

  it("SCHED-1: a re-dirty while queued meshes the LATEST generation (supersede)", () => {
    const { sched, store } = makeScheduler(1000);
    const k = sectionKey(0, 0, 0);
    const g1 = sched.markDirty(k, DirtyReason.Simulation);
    const g2 = sched.markDirty(k, DirtyReason.Simulation); // bumps generation while queued
    expect(g2).toBeGreaterThan(g1);
    sched.tick(0);
    // The section's committed/last generation reflects g2, never the stale g1.
    expect(store.get(k)!.generation).toBe(g2);
    expect(store.get(k)!.pendingUpdateType).toBe(0); // cleared on dispatch
    expect(store.get(k)!.queued).toBe(false);
  });

  it("SCHED-3: a burst beyond the CPU budget spreads across frames (≥1 dispatched per tick)", () => {
    let c = 0;
    const now = () => c++ * 1000; // each call jumps 1000ms — budget 10ms allows exactly 1 mesh/tick
    const { sched } = makeScheduler(10, now);
    for (let i = 0; i < 3; i++) sched.markDirty(sectionKey(i, 0, 0), DirtyReason.Simulation);
    expect(sched.pending).toBe(3);
    expect(sched.tick(0).meshed).toBe(1);
    expect(sched.pending).toBe(2);
    expect(sched.tick(1).meshed).toBe(1);
    expect(sched.tick(2).meshed).toBe(1);
    expect(sched.pending).toBe(0); // all eventually meshed, none dropped (TRAP 16.A)
  });

  it("ABP-1: meshBudgetOf sizes the CPU mesh budget per tick (overrides the fixed literal)", () => {
    const device = new FakeGraphicsDevice();
    const store = new SectionStore(new GpuSectionUploader(device));
    const uploads = new UploadScheduler(store);
    const snapshots = new SnapshotSource(blocks);
    let c = 0;
    const now = () => c++ * 1000; // each now() call jumps 1000ms
    let budget = 10; // tiny adaptive budget → 1 mesh/tick
    // The fixed meshBudgetMs is huge; meshBudgetOf MUST win, proving the adaptive provider governs.
    const sched = new Scheduler({ store, snapshots, provider, uploads, meshBudgetMs: 99_999, now, meshBudgetOf: () => budget });
    for (let i = 0; i < 3; i++) sched.markDirty(sectionKey(i, 0, 0), DirtyReason.Simulation);
    expect(sched.tick(0).meshed).toBe(1); // throttled by the adaptive 10ms, not the 99999 literal
    expect(sched.pending).toBe(2);
    budget = Infinity; // headroom opens up → drain the rest in one tick
    expect(sched.tick(1).meshed).toBe(2);
    expect(sched.pending).toBe(0);
  });

  it("no budget (Infinity) meshes the whole burst in one tick", () => {
    const { sched } = makeScheduler(Infinity);
    for (let i = 0; i < 4; i++) sched.markDirty(sectionKey(i, 0, 0), DirtyReason.Simulation);
    expect(sched.tick(0).meshed).toBe(4);
    expect(sched.pending).toBe(0);
  });

  it("ABP-2: a section marked immediate+queued meshes ONCE inline and leaves the queue (no double-mesh)", () => {
    let c = 0;
    const now = () => c++ * 1000; // a tiny budget would normally throttle the burst…
    const { sched } = makeScheduler(0, now); // …but the immediate drain ignores the budget
    const k = sectionKey(0, 0, 0);
    sched.markDirty(k, DirtyReason.Simulation); // queued normally
    sched.markDirty(k, DirtyReason.Simulation, 0, true); // ALSO flagged immediate
    expect(sched.queue.has(k)).toBe(true); // still queued before the tick
    const t = sched.tick(0);
    expect(t.meshed).toBe(1); // meshed exactly once (inline immediate), not again by the normal loop
    expect(sched.queue.has(k)).toBe(false); // drainImmediate removed it from the queue
    expect(sched.pending).toBe(0);
  });

  it("ABP-2: an immediate edit caps inline work per tick; the excess falls back to the normal path", () => {
    const { sched } = makeScheduler(Infinity);
    // 6 immediate sections, cap is 4 → 4 mesh inline, 2 spill to the (here unbudgeted) normal loop.
    for (let i = 0; i < 6; i++) sched.markDirty(sectionKey(i, 0, 0), DirtyReason.Simulation, 0, true);
    const t = sched.tick(0);
    expect(t.meshed).toBe(6); // 4 immediate + 2 normal, all meshed (none dropped — TRAP 16.A)
    expect(sched.pending).toBe(0);
  });

  it("SCHED-4: under a tight CPU budget, the near-camera section meshes first", () => {
    let c = 0;
    const now = () => c++ * 1000; // budget 10ms → exactly 1 mesh dispatched per tick
    const { sched, store } = makeScheduler(10, now);
    const near = sectionKey(0, 0, 0);
    const far = sectionKey(20, 0, 0);
    sched.markDirty(far, DirtyReason.Simulation, 1000); // enqueued first, but FAR
    sched.markDirty(near, DirtyReason.Simulation, 1000);
    sched.tick(0, [0, 0, 0]); // camera at section 0 → near dispatches before far
    expect(store.get(near)!.presented).not.toBeNull(); // near meshed + committed this tick
    expect(store.get(far)!.presented).toBeNull(); // far deferred to a later tick
  });

  it("re-dirtying a queued section RAISES its priority and never lowers it (no inversion)", () => {
    // Re-dirties route through enqueue's OR-join (not a generation-only supersede), so a higher-priority
    // re-dirty correctly raises the queued task — the latent priority-inversion the audit flagged.
    const { sched } = makeScheduler(1000);
    const k = sectionKey(0, 0, 0);
    sched.markDirty(k, DirtyReason.Simulation, 100); // first dirty, priority 100
    sched.markDirty(k, DirtyReason.Simulation, 1000); // higher → must raise
    sched.markDirty(k, DirtyReason.Simulation, 50); // lower → must NOT lower
    expect(sched.pending).toBe(1); // still one coalesced task (N1)
    expect(sched.queue.pop(0)!.basePriority).toBe(1000);
  });
});

// Phase 5: meshing through a WorkerPool (here an INLINE pool — async via microtask, deterministic with
// flush) instead of synchronous inline. Dispatch and commit are separated in time; stale results drop.
const flush = (): Promise<void> => new Promise((r) => setTimeout(r));

function makePoolScheduler() {
  const device = new FakeGraphicsDevice();
  const store = new SectionStore(new GpuSectionUploader(device));
  const uploads = new UploadScheduler(store);
  const snapshots = new SnapshotSource(blocks);
  // A mesh fn that produces one solid quad tagged with the snapshot's generation.
  const mesh = (snap: SectionSnapshot) =>
    makeBuildOutput({
      sectionKey: snap.sectionKey,
      generation: snap.generation,
      info: { flags: 0 },
      parts: { [TerrainPass.Solid]: { vertexData: new ArrayBuffer(80), quadCount: 1 } },
    });
  const pool = new WorkerPool({ mode: "inline", mesh });
  const sched = new Scheduler({ store, snapshots, provider, uploads, meshBudgetMs: Infinity, pool });
  return { sched, store, pool };
}

describe("Scheduler (Phase 5 off-thread dispatch/drain)", () => {
  it("dispatches asynchronously, then commits the result on a later tick", async () => {
    const { sched, store } = makePoolScheduler();
    const k = sectionKey(0, 0, 0);
    sched.markDirty(k, DirtyReason.Simulation);

    const t0 = sched.tick(0);
    expect(t0.meshed).toBe(1); // dispatched to the pool
    expect(t0.committed).toBe(0); // result not back yet (async)
    expect(store.get(k)!.presented).toBeNull();

    await flush(); // inline pool delivers buildDone → onWorkerMessage → submitBuild
    const t1 = sched.tick(1);
    expect(t1.committed).toBe(1); // drained + committed
    expect(store.get(k)!.presented).not.toBeNull();
  });

  it("hasPendingWork tracks queue+inFlight, and onWork nudges the loop when a reply stages a commit", async () => {
    const device = new FakeGraphicsDevice();
    const store = new SectionStore(new GpuSectionUploader(device));
    const uploads = new UploadScheduler(store);
    const snapshots = new SnapshotSource(blocks);
    const mesh = (snap: SectionSnapshot) =>
      makeBuildOutput({
        sectionKey: snap.sectionKey,
        generation: snap.generation,
        info: { flags: 0 },
        parts: { [TerrainPass.Solid]: { vertexData: new ArrayBuffer(80), quadCount: 1 } },
      });
    const pool = new WorkerPool({ mode: "inline", mesh });
    let nudges = 0;
    const sched = new Scheduler({ store, snapshots, provider, uploads, meshBudgetMs: Infinity, pool, onWork: () => nudges++ });
    const k = sectionKey(0, 0, 0);

    expect(sched.hasPendingWork()).toBe(false); // idle scene → loop may idle
    sched.markDirty(k, DirtyReason.Simulation);
    expect(sched.hasPendingWork()).toBe(true); // queued
    sched.tick(0); // dispatched off-thread → inFlight (queue now empty)
    expect(sched.hasPendingWork()).toBe(true); // still pending (in-flight)
    expect(nudges).toBe(0); // reply not back yet

    await flush(); // inline pool delivers buildDone → onWorkerMessage → submitBuild + onWork
    expect(nudges).toBe(1); // the loop was nudged to run the next drainAndCommit
    expect(sched.hasPendingWork()).toBe(false); // inFlight drained; result now staged in the uploads inbox
    expect(uploads.pendingCount).toBe(1); // ...where the viewer's liveness gate still sees it (render-on-demand)
  });

  it("stale-drops a result whose generation was superseded while meshing (Invariant)", async () => {
    const { sched, store } = makePoolScheduler();
    const k = sectionKey(0, 0, 0);
    sched.markDirty(k, DirtyReason.Simulation); // gen 1
    sched.tick(0); // dispatch gen 1
    const g2 = sched.markDirty(k, DirtyReason.Simulation); // gen 2 — supersede while "meshing"

    await flush(); // the gen-1 result arrives → stale (section is gen 2) → dropped, not presented
    const t1 = sched.tick(1); // dispatches gen 2
    expect(t1.discarded).toBe(1); // the stale gen-1 drop is surfaced (Bug 3 — was pinned at 0)
    expect(store.get(k)!.presented).toBeNull(); // gen-1 commit was dropped

    await flush(); // gen-2 result arrives
    sched.tick(2);
    expect(store.get(k)!.presented).not.toBeNull();
    expect(store.get(k)!.lastBuiltGeneration).toBe(g2); // committed the CURRENT generation only
  });

  it("flags the prior in-flight job for SAB cancellation when a section is re-dispatched (INFRA-3)", () => {
    const { sched, pool } = makePoolScheduler();
    const cancels: number[] = [];
    const orig = pool.cancelSharedJob.bind(pool);
    pool.cancelSharedJob = (jobId: number) => { cancels.push(jobId); orig(jobId); };
    const k = sectionKey(0, 0, 0);

    sched.markDirty(k, DirtyReason.Simulation); // gen 1
    sched.tick(0); // dispatch gen-1 (jobId 1) — first build, nothing prior to cancel
    expect(cancels).toEqual([]);

    sched.markDirty(k, DirtyReason.Simulation); // gen 2 — supersede while gen-1 is still in flight (no flush)
    sched.tick(1); // dispatch gen-2 (jobId 2) → flags the prior jobId 1
    expect(cancels).toEqual([1]);
  });

  it("WRK-3: re-dispatch issues the Model-A cooperative cancel for the PRIOR jobId, never the current", () => {
    // A worker-mode pool with cooperativeCancelA ON: supersede(prior) routes to pool.cancel(prior). We spy
    // pool.cancel to prove the PRIOR in-flight jobId is the one cancelled (the just-bumped current is never
    // touched — TRAP WRK-3.C). The fake worker replies buildDone so the surviving current-gen build commits.
    const device = new FakeGraphicsDevice();
    const store = new SectionStore(new GpuSectionUploader(device));
    const uploads = new UploadScheduler(store);
    const snapshots = new SnapshotSource(blocks);
    let onmsg: ((e: MessageEvent) => void) | null = null;
    const pending: { jobId: number; snapshot: SectionSnapshot }[] = [];
    // Mesh fn producing one solid quad tagged with the snapshot's generation (matches makePoolScheduler).
    const mesh = (snap: SectionSnapshot) =>
      makeBuildOutput({ sectionKey: snap.sectionKey, generation: snap.generation, info: { flags: 0 }, parts: { [TerrainPass.Solid]: { vertexData: new ArrayBuffer(80), quadCount: 1 } } });
    const fakeWorker = {
      get onmessage() { return onmsg; },
      set onmessage(fn: ((e: MessageEvent) => void) | null) { onmsg = fn; },
      postMessage(req: { type: string; jobId?: number; snapshot?: SectionSnapshot }) {
        if (req.type === "build") pending.push({ jobId: req.jobId!, snapshot: req.snapshot! });
      },
      terminate() {},
      flushOne() {
        const job = pending.shift();
        if (!job) return;
        const payload = serializeBuildOutput(mesh(job.snapshot)).payload; // gen-tagged from the snapshot
        onmsg?.({ data: { type: "buildDone", jobId: job.jobId, payload } } as MessageEvent);
      },
    };
    const pool = new WorkerPool({ mode: "worker", init: {}, size: 1, cooperativeCancelA: true, createWorker: () => fakeWorker as unknown as Worker });
    const cancels: number[] = [];
    const origCancel = pool.cancel.bind(pool);
    pool.cancel = (jobId: number) => { cancels.push(jobId); origCancel(jobId); };
    const sched = new Scheduler({ store, snapshots, provider, uploads, meshBudgetMs: Infinity, pool });
    const k = sectionKey(0, 0, 0);

    sched.markDirty(k, DirtyReason.Simulation); // gen 1
    sched.tick(0); // dispatch gen-1 (jobId 1) — nothing prior to cancel
    expect(cancels).toEqual([]);

    sched.markDirty(k, DirtyReason.Simulation); // gen 2 — supersede while gen-1 still in flight
    sched.tick(1); // dispatch gen-2 (jobId 2) → cooperatively cancels the PRIOR jobId 1 (not 2)
    expect(cancels).toEqual([1]); // the PRIOR jobId — the current (2) is never cancelled

    // The surviving current-gen build still commits (cancel is a pure CPU-savings hint; T-PRES intact).
    fakeWorker.flushOne(); // reply jobId 1 (stale gen-1) — dropped by the generation stale-drop
    fakeWorker.flushOne(); // reply jobId 2 (current gen-2) — committed
    sched.tick(2);
    expect(store.get(k)!.presented).not.toBeNull();
  });

  it("ABP-2: an immediate near edit presents THIS tick even behind a pool; a normal edit defers", () => {
    const { sched, store } = makePoolScheduler();
    const near = sectionKey(0, 0, 0);
    const far = sectionKey(1, 0, 0);
    sched.markDirty(near, DirtyReason.Simulation, 0, true); // ABP-2 immediate → inline mesh + same-tick commit
    sched.markDirty(far, DirtyReason.Simulation, 0, false); // normal → dispatched to the pool (async)
    sched.tick(0); // NO flush: the pool result for `far` is not back yet
    expect(store.get(near)!.presented).not.toBeNull(); // immediate presented this very tick
    expect(store.get(far)!.presented).toBeNull(); // async edit still in flight (commits a later tick)
  });
});

// WRK-1: per-frame queued-duration admission gate on the POOL dispatch path. A burst (~explosion) must
// dispatch only a BOUNDED number of jobs and DEFER the rest into the aging TaskQueue (never dropped, never
// starved). The Model-A path (inline pool) is exercised here — the worst case for clone spikes.
//
// Each snapshot is an apron-1 cube → 18³ uint32 = 23328 bytes (snap.blocks.byteLength). We size the gate via
// a transparent budget = workers * frameMs * bytesPerMs so the admitted count is exact.
const SNAP_BYTES = 18 * 18 * 18 * 4; // 23328 — matches the live SnapshotSource(blocks) at APRON=1

/** A pool-backed scheduler with a WRK-1 admission gate whose per-frame budget is `budgetSnapshots` snapshots
 *  (Model-A path: no queuedBytesOf ⇒ the Scheduler's own inFlightBytes accumulator gates). */
function makeGatedScheduler(budgetSnapshots: number, queuedBytesOf?: () => number) {
  const device = new FakeGraphicsDevice();
  const store = new SectionStore(new GpuSectionUploader(device));
  const uploads = new UploadScheduler(store);
  const snapshots = new SnapshotSource(blocks);
  const mesh = (snap: SectionSnapshot) =>
    makeBuildOutput({
      sectionKey: snap.sectionKey,
      generation: snap.generation,
      info: { flags: 0 },
      parts: { [TerrainPass.Solid]: { vertexData: new ArrayBuffer(80), quadCount: 1 } },
    });
  const pool = new WorkerPool({ mode: "inline", mesh });
  const sched = new Scheduler({
    store, snapshots, provider, uploads, meshBudgetMs: Infinity, pool,
    // budget = 1 * (budgetSnapshots * SNAP_BYTES) * 1 = budgetSnapshots snapshots' worth of bytes.
    admissionGate: { frameMsOf: () => budgetSnapshots * SNAP_BYTES, workers: 1, bytesPerMs: 1, queuedBytesOf },
  });
  return { sched, store, pool };
}

describe("Scheduler (WRK-1 worker-effort admission gate)", () => {
  it("an explosion-size burst dispatches a BOUNDED number of jobs and defers the rest (no loss)", async () => {
    const { sched } = makeGatedScheduler(3); // budget = 3 snapshots/frame
    const N = 40;
    for (let i = 0; i < N; i++) sched.markDirty(sectionKey(i, 0, 0), DirtyReason.Simulation);
    expect(sched.pending).toBe(N);

    // Frame 0: gate admits exactly 3 (queuedBytes hits 3*SNAP at the 4th), defers the other 37.
    const t0 = sched.tick(0);
    expect(t0.meshed).toBe(3); // bounded — NOT 40 (no ~920 KB single-frame clone spike)
    expect(sched.pending).toBe(N - 3); // the rest stay queued (the deferral tier)

    // Drain across frames: each frame the inline pool returns 3 builds (inFlightBytes falls), so the gate
    // re-opens and admits 3 more. Eventually every section meshes — zero work lost, no stale-drop inflation.
    let cumulative = t0.meshed;
    let totalDiscarded = t0.discarded;
    for (let f = 1; f < 40 && sched.pending > 0; f++) {
      await flush(); // prior frame's builds retire → inFlightBytes returns toward 0
      const t = sched.tick(f);
      cumulative += t.meshed;
      totalDiscarded += t.discarded;
    }
    expect(sched.pending).toBe(0);
    expect(cumulative).toBe(N); // cumulative meshed == sections dirtied — nothing dropped (TRAP 16.A)
    expect(totalDiscarded).toBe(0); // no stale-drop inflation
  });

  it("never starves — admits ≥1 even when the in-flight estimate already exceeds the budget", () => {
    const { sched } = makeGatedScheduler(1); // budget = 1 snapshot/frame
    // Dispatch one (fills inFlightBytes to 1 snapshot, == budget) WITHOUT flushing → it stays in flight.
    sched.markDirty(sectionKey(0, 0, 0), DirtyReason.Simulation);
    expect(sched.tick(0).meshed).toBe(1); // first frame admits ≥1
    expect(sched.inFlightBytesEstimate).toBe(SNAP_BYTES); // one snapshot still in flight (no flush)

    // Now the frame-start estimate already == budget. A new burst still admits ≥1 (meshed>0 guard) — the
    // gate trips only AFTER the always-admitted first job, never starving the queue.
    for (let i = 1; i <= 5; i++) sched.markDirty(sectionKey(i, 0, 0), DirtyReason.Simulation);
    const t = sched.tick(1);
    expect(t.meshed).toBeGreaterThanOrEqual(1);
  });

  it("flag-off (no admissionGate) is byte-identical: the whole burst meshes in one tick", () => {
    // Same scenario as 'no budget meshes the whole burst', but through the POOL path WITHOUT a gate — proves
    // absent admissionGate == the exact current dispatch path (no bounding, no deferral).
    const { sched } = makePoolScheduler(); // no admissionGate
    for (let i = 0; i < 40; i++) sched.markDirty(sectionKey(i, 0, 0), DirtyReason.Simulation);
    expect(sched.tick(0).meshed).toBe(40); // all dispatched in one tick — no gate in the absent-config path
    expect(sched.pending).toBe(0);
  });

  it("a deferred (gated) task is RE-ENQUEUED — not dropped — and keeps its ORIGINAL age", () => {
    const { sched } = makeGatedScheduler(1); // budget = 1 snapshot/frame → admit 1, defer the 2nd
    const a = sectionKey(0, 0, 0);
    const b = sectionKey(5, 0, 0);
    sched.markDirty(a, DirtyReason.Simulation); // enqueued at frame 0
    sched.markDirty(b, DirtyReason.Simulation); // enqueued at frame 0
    // Tick at a LATER frame so the original enqueuedFrame (0) is distinguishable from currentFrame (7).
    const t = sched.tick(7);
    expect(t.meshed).toBe(1); // exactly one admitted
    expect(sched.pending).toBe(1); // the other deferred — still in the queue, NOT dropped (no work loss)
    // The deferred task kept its ORIGINAL enqueuedFrame (0) so AGE_WEIGHT keeps aging it (currentFrame would
    // reset its age). pop() exposes the surviving task; its age boost reflects frame 0, not frame 7.
    const survivor = sched.queue.pop(7)!;
    expect(survivor.enqueuedFrame).toBe(0); // original age preserved (TRAP WRK-1.B)
  });

  it("in-flight bytes are released on stale-drop and cancel, not just buildDone (no leak)", async () => {
    const { sched } = makeGatedScheduler(10); // generous budget so the gate never blocks this test
    const k = sectionKey(0, 0, 0);
    sched.markDirty(k, DirtyReason.Simulation); // gen 1
    sched.tick(0); // dispatch gen-1 → inFlightBytes = 1 snapshot
    expect(sched.inFlightBytesEstimate).toBe(SNAP_BYTES);
    sched.markDirty(k, DirtyReason.Simulation); // gen 2 — supersede while gen-1 is "meshing" (no flush)

    await flush(); // the gen-1 result arrives → STALE-DROPPED in onWorkerMessage (not committed)
    const t = sched.tick(1); // also dispatches gen-2 (+1 snapshot in flight)
    expect(t.discarded).toBe(1); // the stale gen-1 build was dropped (its bytes must still be retired)
    // Bytes for the stale-dropped gen-1 job were released on its terminal path (the shared delete), leaving
    // only the freshly-dispatched gen-2 job in flight — the estimate did NOT leak upward (TRAP WRK-1.C).
    expect(sched.inFlightBytesEstimate).toBe(SNAP_BYTES);

    await flush(); // gen-2 result arrives + commits → estimate returns fully to 0 (idle scene)
    sched.tick(2);
    expect(sched.inFlightBytesEstimate).toBe(0);
  });

  it("Model B: queuedBytesOf is the single source of truth (no double-count with inFlightBytes)", () => {
    // When a Model-B queuedBytesOf is wired, the Scheduler must NOT also accumulate inFlightBytes on post
    // (that would double-count the same bytes and over-throttle the gate permanently).
    let ringBytes = 0;
    const { sched } = makeGatedScheduler(10, () => ringBytes); // gate reads the (mock) SAB ring estimate
    sched.markDirty(sectionKey(0, 0, 0), DirtyReason.Simulation);
    sched.tick(0); // dispatch — Model-B path must leave the main-thread accumulator at 0
    expect(sched.inFlightBytesEstimate).toBe(0); // the SAB ring is the source of truth, not inFlightBytes
  });
});
