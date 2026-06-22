// Inline WorkerPool meshing + cancellation. RENDERER_PLAN §6, §22.

import { describe, it, expect } from "vitest";
import { WorkerPool } from "./WorkerPool";
import { SharedMeshQueue } from "./SharedMeshQueue";
import { WorkerCancelFlags } from "./CancelFlags";
import { makeBuildOutput, serializeBuildOutput, type SectionBuildOutput } from "./BuildOutput";
import type { WorkerRequest, WorkerResponse } from "./protocol";
import { TerrainPass, type Generation } from "../types";
import { sectionKey } from "../world/SectionKey";
import type { SectionSnapshot } from "../world/SnapshotSource";

const KEY = sectionKey(0, 0, 0);

/** A trivial mesh fn: one solid quad's worth of (empty) geometry, tagged with the generation. */
function fakeMesh(snapshot: SectionSnapshot): SectionBuildOutput {
  const part = { vertexData: new ArrayBuffer(8), quadCount: 1 };
  return makeBuildOutput({ sectionKey: snapshot.sectionKey, generation: snapshot.generation, info: { flags: 0 }, parts: { [TerrainPass.Solid]: part } });
}

function snapshot(generation: Generation): SectionSnapshot {
  return { sectionKey: KEY, generation, origin: [0, 0, 0], size: 16, apron: 1, blocks: new Uint32Array(0), changedReason: 0 };
}

const flush = (): Promise<void> => new Promise((r) => setTimeout(r));

describe("WorkerPool (inline)", () => {
  it("runs the mesh fn for a build job and delivers a wrappable BuildPayload", async () => {
    const pool = new WorkerPool({ mode: "inline", mesh: fakeMesh });
    const got: WorkerResponse[] = [];
    pool.onMessage((r) => got.push(r));

    pool.post({ type: "build", jobId: 7, snapshot: snapshot(3) });
    await flush();

    expect(got).toHaveLength(1);
    const res = got[0];
    expect(res.type).toBe("buildDone");
    if (res.type !== "buildDone") return;
    expect(res.jobId).toBe(7);
    const out: SectionBuildOutput = makeBuildOutput(res.payload);
    expect(out.parts[TerrainPass.Solid]?.quadCount).toBe(1);
    expect(out.generation).toBe(3);
  });

  it("converts a build into `cancelled` when the job was cancelled first", async () => {
    const pool = new WorkerPool({ mode: "inline", mesh: fakeMesh });
    const got: WorkerResponse[] = [];
    pool.onMessage((r) => got.push(r));

    pool.post({ type: "build", jobId: 9, snapshot: snapshot(1) });
    pool.cancel(9);
    await flush();

    expect(got).toHaveLength(1);
    expect(got[0].type).toBe("cancelled");
  });

  it("THROWS on a `sort` request instead of silently dropping it (P1.5 / TRAP 16.A)", () => {
    const pool = new WorkerPool({ mode: "inline", mesh: fakeMesh });
    expect(() =>
      pool.post({ type: "sort", jobId: 1, sectionKey: KEY, generation: 0 as Generation, sortData: new ArrayBuffer(0), cameraPos: [0, 0, 0] }),
    ).toThrow(/sort|main thread/i);
  });

  it("WRK-1: workerCount is 1 for an inline pool (no real workers, but one mesh thread)", () => {
    const pool = new WorkerPool({ mode: "inline", mesh: fakeMesh });
    expect(pool.workerCount).toBe(1);
  });
});

// An in-process fake of the real mesher.worker: replies to `build` with a transferable buildDone,
// records `init`, honors `cancel`. Lets us test the pool's spawn/init/round-robin/deliver glue without a
// real DOM Worker (the actual meshing is tested in MeshInit.test).
function makeFakeWorker(log: { inits: number; built: number[] }): Worker {
  let onmessage: ((e: MessageEvent) => void) | null = null;
  const w = {
    get onmessage() { return onmessage; },
    set onmessage(fn: ((e: MessageEvent) => void) | null) { onmessage = fn; },
    postMessage(req: WorkerRequest) {
      if (req.type === "init") { log.inits++; return; }
      if (req.type !== "build") return; // cancel/addModels ignored — matches the real worker (no cancel set)
      const { jobId, snapshot } = req;
      queueMicrotask(() => {
        log.built.push(jobId);
        const payload = serializeBuildOutput(fakeMesh(snapshot)).payload;
        onmessage?.({ data: { type: "buildDone", jobId, payload } } as MessageEvent);
      });
    },
    terminate() {},
  };
  return w as unknown as Worker;
}

describe("WorkerPool (worker mode)", () => {
  it("spawns workers, inits each, and delivers a wrappable BuildPayload from a build", async () => {
    const log = { inits: 0, built: [] as number[] };
    const pool = new WorkerPool({ mode: "worker", init: { models: [] }, size: 2, createWorker: () => makeFakeWorker(log) });
    expect(log.inits).toBe(2); // every spawned worker got the init payload

    const got: WorkerResponse[] = [];
    pool.onMessage((r) => got.push(r));
    pool.post({ type: "build", jobId: 7, snapshot: snapshot(3) });
    await flush();

    expect(got).toHaveLength(1);
    expect(got[0].type).toBe("buildDone");
    if (got[0].type !== "buildDone") return;
    expect(got[0].jobId).toBe(7);
    expect(makeBuildOutput(got[0].payload).generation).toBe(3);
  });

  it("round-robins builds across the worker pool", async () => {
    const logs = [
      { inits: 0, built: [] as number[] },
      { inits: 0, built: [] as number[] },
    ];
    let i = 0;
    const pool = new WorkerPool({ mode: "worker", init: {}, size: 2, createWorker: () => makeFakeWorker(logs[i++]) });
    pool.onMessage(() => {});
    pool.post({ type: "build", jobId: 1, snapshot: snapshot(1) });
    pool.post({ type: "build", jobId: 2, snapshot: snapshot(1) });
    await flush();
    expect(logs[0].built).toEqual([1]); // job 1 → worker 0
    expect(logs[1].built).toEqual([2]); // job 2 → worker 1
  });

  it("a cancelled build is meshed but its result is converted to `cancelled` main-side (leak-free)", async () => {
    // Workers don't drop cancelled builds (that silent drop was the source of two confirmed set leaks).
    // The pool marks the jobId and converts the always-arriving result to `cancelled` in deliver() — the
    // same model as the Scheduler's generation stale-drop. `cancelled` self-clears, no jobId leaks.
    const log = { inits: 0, built: [] as number[] };
    const pool = new WorkerPool({ mode: "worker", init: {}, size: 1, createWorker: () => makeFakeWorker(log) });
    const got: WorkerResponse[] = [];
    pool.onMessage((r) => got.push(r));
    pool.post({ type: "build", jobId: 9, snapshot: snapshot(1) });
    pool.cancel(9);
    await flush();
    expect(log.built).toEqual([9]); // worker still meshed it (no silent drop)
    expect(got).toHaveLength(1);
    expect(got[0].type).toBe("cancelled"); // result converted to cancelled by the pool, not committed
  });

  it("WRK-1: workerCount reports the spawned worker count (sizes the admission budget)", () => {
    const pool = new WorkerPool({ mode: "worker", init: {}, size: 3, createWorker: () => makeFakeWorker({ inits: 0, built: [] }) });
    expect(pool.workerCount).toBe(3);
    // After dispose the pool degenerates to 1 (the gate never divides by 0 / multiplies by 0 workers).
    pool.dispose();
    expect(pool.workerCount).toBe(1);
  });

  it("dispatching to a pool whose workers all failed to spawn throws (no silent drop — TRAP 16.A)", () => {
    const pool = new WorkerPool({ mode: "worker", init: {}, size: 1, createWorker: () => makeFakeWorker({ inits: 0, built: [] }) });
    // Force the empty-pool state (e.g. spawn failure) and confirm a build fails loudly rather than vanishing.
    (pool as unknown as { workers: Worker[] }).workers.length = 0;
    expect(() => pool.post({ type: "build", jobId: 1, snapshot: snapshot(1) })).toThrow(/no workers/i);
  });
});

// WRK-3: a fake worker that attaches the cooperative-cancel SAB from its init message (recording its slot
// index), and on `build` polls its CANCEL_REQUEST word EXACTLY like the real mesher.worker — replying
// `cancelled` WITHOUT meshing when the request matches its active job, else meshing + buildDone. Builds are
// MANUAL (queued until flushOne) so a cancel can land between the dispatch and the "mesh", reproducing the
// redstone-clock supersede-before-mesh ordering deterministically.
function makeCancelAwareWorker(log: { inits: number; built: number[]; gotSab: number; slot: number }) {
  let onmessage: ((e: MessageEvent) => void) | null = null;
  let flags: WorkerCancelFlags | null = null;
  const pending: number[] = []; // queued jobIds
  const w = {
    get onmessage() { return onmessage; },
    set onmessage(fn: ((e: MessageEvent) => void) | null) { onmessage = fn; },
    postMessage(req: WorkerRequest & { cancelFlags?: SharedArrayBuffer; workerIndex?: number }) {
      if (req.type === "init") {
        log.inits++;
        if (req.cancelFlags) { log.gotSab++; log.slot = req.workerIndex ?? -1; flags = WorkerCancelFlags.attach(req.cancelFlags, req.workerIndex ?? 0); }
        return;
      }
      if (req.type !== "build") return;
      pending.push(req.jobId);
    },
    terminate() {},
    /** "Run" the next queued build: publish ACTIVE_JOBID, poll the cancel word, then either reply cancelled
     *  (no mesh) or mesh + buildDone — mirroring mesher.worker's build case. */
    flushOne(): boolean {
      const jobId = pending.shift();
      if (jobId === undefined) return false;
      if (flags) {
        flags.setActive(jobId);
        const cancelled = flags.isCancelled(jobId);
        flags.clearActive();
        if (cancelled) {
          onmessage?.({ data: { type: "cancelled", jobId } } as MessageEvent);
          return true;
        }
      }
      log.built.push(jobId); // only reached when NOT cancelled
      const payload = serializeBuildOutput(fakeMesh(snapshot(1))).payload;
      onmessage?.({ data: { type: "buildDone", jobId, payload } } as MessageEvent);
      return true;
    },
  };
  return w;
}

describe("WorkerPool (WRK-3 cooperative cancel — Model A)", () => {
  it("flag ON: a superseded build is observed via CANCEL_REQUEST and replied `cancelled` WITHOUT meshing", () => {
    const log = { inits: 0, built: [] as number[], gotSab: 0, slot: -1 };
    const workers: ReturnType<typeof makeCancelAwareWorker>[] = [];
    const pool = new WorkerPool({
      mode: "worker", init: {}, size: 1, cooperativeCancelA: true,
      createWorker: () => { const w = makeCancelAwareWorker(log); workers.push(w); return w as unknown as Worker; },
    });
    expect(log.gotSab).toBe(1); // the worker received the cancel SAB at init
    expect(log.slot).toBe(0);
    const got: WorkerResponse[] = [];
    pool.onMessage((r) => got.push(r));

    pool.post({ type: "build", jobId: 9, snapshot: snapshot(1) }); // dispatched, not yet "meshed"
    pool.cancel(9); // main supersedes job 9 BEFORE the worker meshes it (writes CANCEL_REQUEST)
    workers[0].flushOne(); // worker runs: sees its CANCEL_REQUEST == ACTIVE_JOBID → cancelled, no mesh

    expect(log.built).not.toContain(9); // NEVER meshed (the whole WRK-3 win)
    expect(got).toHaveLength(1);
    expect(got[0].type).toBe("cancelled"); // exactly one reply, of type cancelled
  });

  it("flag ON: a build NOT cancelled meshes + replies buildDone (no false-positive cancel)", () => {
    const log = { inits: 0, built: [] as number[], gotSab: 0, slot: -1 };
    const workers: ReturnType<typeof makeCancelAwareWorker>[] = [];
    const pool = new WorkerPool({
      mode: "worker", init: {}, size: 1, cooperativeCancelA: true,
      createWorker: () => { const w = makeCancelAwareWorker(log); workers.push(w); return w as unknown as Worker; },
    });
    const got: WorkerResponse[] = [];
    pool.onMessage((r) => got.push(r));
    pool.post({ type: "build", jobId: 1, snapshot: snapshot(1) });
    workers[0].flushOne(); // no cancel requested → meshes normally
    expect(log.built).toEqual([1]);
    expect(got).toHaveLength(1);
    expect(got[0].type).toBe("buildDone");
  });

  it("cancel targets ONLY the worker holding the jobId — per-worker scoping, no cross-cancel (TRAP WRK-3.B)", () => {
    const logs = [
      { inits: 0, built: [] as number[], gotSab: 0, slot: -1 },
      { inits: 0, built: [] as number[], gotSab: 0, slot: -1 },
    ];
    const workers: ReturnType<typeof makeCancelAwareWorker>[] = [];
    let i = 0;
    const pool = new WorkerPool({
      mode: "worker", init: {}, size: 2, cooperativeCancelA: true,
      createWorker: () => { const w = makeCancelAwareWorker(logs[i++]); workers.push(w); return w as unknown as Worker; },
    });
    const got: WorkerResponse[] = [];
    pool.onMessage((r) => got.push(r));

    // job 1 → w0 (rr), job 2 → w1 (rr). Cancel ONLY job 1.
    pool.post({ type: "build", jobId: 1, snapshot: snapshot(1) });
    pool.post({ type: "build", jobId: 2, snapshot: snapshot(1) });
    pool.cancel(1); // writes CANCEL_REQUEST on w0's slot only

    workers[0].flushOne(); // w0: job 1 → cancelled, not meshed
    workers[1].flushOne(); // w1: job 2 → meshed normally (its slot was never written)

    expect(logs[0].built).not.toContain(1); // job 1 cancelled
    expect(logs[1].built).toEqual([2]); // job 2 (the live one on the OTHER worker) survives
    const cancelled = got.filter((r) => r.type === "cancelled");
    const done = got.filter((r) => r.type === "buildDone");
    expect(cancelled.map((r) => r.jobId)).toEqual([1]);
    expect(done.map((r) => r.jobId)).toEqual([2]);
  });

  it("flag OFF: no SAB is sent, cancel meshes + converts main-side (the existing leak-free path, unchanged)", () => {
    const log = { inits: 0, built: [] as number[], gotSab: 0, slot: -1 };
    const workers: ReturnType<typeof makeCancelAwareWorker>[] = [];
    // cooperativeCancelA omitted (default OFF).
    const pool = new WorkerPool({
      mode: "worker", init: {}, size: 1,
      createWorker: () => { const w = makeCancelAwareWorker(log); workers.push(w); return w as unknown as Worker; },
    });
    expect(log.gotSab).toBe(0); // NO cancel SAB in the init message (byte-identical init)
    const got: WorkerResponse[] = [];
    pool.onMessage((r) => got.push(r));
    pool.post({ type: "build", jobId: 9, snapshot: snapshot(1) });
    pool.cancel(9);
    workers[0].flushOne(); // no flags attached → the worker meshes (built contains 9)
    expect(log.built).toEqual([9]); // still meshed (no silent drop — the leak-free backstop)
    expect(got).toHaveLength(1);
    expect(got[0].type).toBe("cancelled"); // converted main-side by deliver() exactly as today
  });
});

// A fake worker whose build replies are MANUAL: builds queue until the test calls flushOne(). Lets us hold
// one worker "slow" (busy) while another drains the backlog, to observe ABP-3 work-stealing deterministically.
function makeManualWorker(log: { inits: number; built: number[] }) {
  let onmessage: ((e: MessageEvent) => void) | null = null;
  const pending: { jobId: number; snapshot: SectionSnapshot }[] = [];
  const w = {
    get onmessage() { return onmessage; },
    set onmessage(fn: ((e: MessageEvent) => void) | null) { onmessage = fn; },
    postMessage(req: WorkerRequest) {
      if (req.type === "init") { log.inits++; return; }
      if (req.type !== "build") return;
      pending.push({ jobId: req.jobId, snapshot: req.snapshot }); // HOLD — replied only on flushOne()
    },
    terminate() {},
    flushOne(): boolean {
      const job = pending.shift();
      if (!job) return false;
      log.built.push(job.jobId);
      const payload = serializeBuildOutput(fakeMesh(job.snapshot)).payload;
      onmessage?.({ data: { type: "buildDone", jobId: job.jobId, payload } } as MessageEvent);
      return true;
    },
    pendingCount() { return pending.length; },
  };
  return w;
}

describe("WorkerPool (Model A work-stealing — ABP-3)", () => {
  it("an idle worker pulls extra builds while a busy worker holds its single in-flight build", () => {
    const workers: ReturnType<typeof makeManualWorker>[] = [];
    const logs = [{ inits: 0, built: [] as number[] }, { inits: 0, built: [] as number[] }];
    let i = 0;
    const pool = new WorkerPool({
      mode: "worker", init: {}, size: 2,
      createWorker: () => { const w = makeManualWorker(logs[i++]); workers.push(w); return w as unknown as Worker; },
    });
    pool.workStealing = true;
    pool.onMessage(() => {});

    // 4 builds: b0→w0, b1→w1 (one each, both busy); b2,b3 wait in the deque.
    for (let j = 0; j < 4; j++) pool.post({ type: "build", jobId: j, snapshot: snapshot(1) });
    expect(workers[0].pendingCount()).toBe(1); // ≤1 in-flight per worker
    expect(workers[1].pendingCount()).toBe(1);

    // w0 is "fast": each reply frees it and it steals the next queued build. w1 stays slow (never replies).
    workers[0].flushOne(); // b0 done → steal b2
    workers[0].flushOne(); // b2 done → steal b3
    workers[0].flushOne(); // b3 done → deque empty
    expect(logs[0].built).toEqual([0, 2, 3]); // the fast worker did 3 of 4 (> N/workers = 2)
    expect(workers[1].pendingCount()).toBe(1); // w1 still holds only its original single build

    workers[1].flushOne(); // w1 finally replies b1
    expect(logs[1].built).toEqual([1]);
    // Every jobId yielded exactly one reply — no leak, payloads worker-independent.
    expect([...logs[0].built, ...logs[1].built].sort((a, b) => a - b)).toEqual([0, 1, 2, 3]);
  });

  it("default (round-robin) splits builds evenly — work-stealing is opt-in", () => {
    const logs = [{ inits: 0, built: [] as number[] }, { inits: 0, built: [] as number[] }];
    let i = 0;
    const pool = new WorkerPool({ mode: "worker", init: {}, size: 2, createWorker: () => makeManualWorker(logs[i++]) as unknown as Worker });
    // workStealing left false → exact rr++ path.
    for (let j = 0; j < 4; j++) pool.post({ type: "build", jobId: j, snapshot: snapshot(1) });
    // rr alternates regardless of reply timing: jobs 0,2 → w0; 1,3 → w1 (held, but assigned immediately).
    expect((pool as unknown as { pendingBuilds: unknown[] }).pendingBuilds.length).toBe(0); // no deque used
  });

  it("dispose clears the work-stealing bookkeeping", () => {
    const pool = new WorkerPool({ mode: "worker", init: {}, size: 1, createWorker: () => makeManualWorker({ inits: 0, built: [] }) as unknown as Worker });
    pool.workStealing = true;
    pool.post({ type: "build", jobId: 1, snapshot: snapshot(1) }); // assigned (busy)
    pool.post({ type: "build", jobId: 2, snapshot: snapshot(1) }); // queued in pendingBuilds
    pool.dispose();
    expect((pool as unknown as { pendingBuilds: unknown[] }).pendingBuilds.length).toBe(0);
    expect((pool as unknown as { busy: unknown[] }).busy.length).toBe(0);
  });
});

// WRK-2: a fake worker that records every `addModels` (and the epoch it carried) AND every dispatched
// build's posted epoch — so a test can assert (a) the steal path stamps a build with the required epoch and
// (b) the rr/flag-off path attaches NO epoch (byte-identity). Builds are MANUAL (held until flushOne), so a
// build can be observed as STOLEN by an idle worker before either replies.
function makeModelAwareWorker(log: {
  inits: number;
  built: number[];
  addModels: number[]; // epochs received via addModels (undefined → -1 so it's representable)
  buildEpochs: (number | undefined)[]; // the epoch field on each dispatched build, in dispatch order
}) {
  let onmessage: ((e: MessageEvent) => void) | null = null;
  const pending: { jobId: number; snapshot: SectionSnapshot }[] = [];
  const w = {
    get onmessage() { return onmessage; },
    set onmessage(fn: ((e: MessageEvent) => void) | null) { onmessage = fn; },
    postMessage(req: WorkerRequest) {
      if (req.type === "init") { log.inits++; return; }
      if (req.type === "addModels") { log.addModels.push(req.epoch ?? -1); return; }
      if (req.type !== "build") return;
      log.buildEpochs.push(req.epoch); // exactly what the pool stamped (undefined on rr / flag-off)
      pending.push({ jobId: req.jobId, snapshot: req.snapshot });
    },
    terminate() {},
    flushOne(): boolean {
      const job = pending.shift();
      if (!job) return false;
      log.built.push(job.jobId);
      const payload = serializeBuildOutput(fakeMesh(job.snapshot)).payload;
      onmessage?.({ data: { type: "buildDone", jobId: job.jobId, payload } } as MessageEvent);
      return true;
    },
    pendingCount() { return pending.length; },
  };
  return w;
}

function emptyModelLog() {
  return { inits: 0, built: [] as number[], addModels: [] as number[], buildEpochs: [] as (number | undefined)[] };
}

describe("WorkerPool (WRK-2 work-stealing + addModels epoch gate — Model A)", () => {
  it("steal + addModels race: the addModels broadcast carries modelEpochA and a STOLEN build is stamped with it", () => {
    const logs = [emptyModelLog(), emptyModelLog()];
    const workers: ReturnType<typeof makeModelAwareWorker>[] = [];
    let i = 0;
    const pool = new WorkerPool({
      mode: "worker", init: {}, size: 2, workStealing: true,
      createWorker: () => { const w = makeModelAwareWorker(logs[i++]); workers.push(w); return w as unknown as Worker; },
    });
    pool.onMessage(() => {});

    // A new palette id is minted → addModels broadcast to BOTH workers, bumping modelEpochA to 1.
    pool.post({ type: "addModels", models: [] });
    expect(logs[0].addModels).toEqual([1]);
    expect(logs[1].addModels).toEqual([1]); // every worker grew its palette + recorded the epoch

    // 4 builds enter post() AFTER the addModels: b0→w0, b1→w1 (both busy), b2,b3 wait in the deque.
    for (let j = 0; j < 4; j++) pool.post({ type: "build", jobId: j, snapshot: snapshot(1) });
    expect(workers[0].pendingCount()).toBe(1);
    expect(workers[1].pendingCount()).toBe(1);

    // w0 is fast: each reply frees it and it STEALS the next queued build off the deque.
    workers[0].flushOne(); // b0 done → steal b2
    workers[0].flushOne(); // b2 done → steal b3
    workers[0].flushOne(); // b3 done → deque empty
    expect(logs[0].built).toEqual([0, 2, 3]); // fast worker did 3 of 4 (the steal happened)

    // EVERY build it ran — including the stolen b2/b3 — was stamped with the captured epoch 1, so the worker
    // gate (req.epoch > appliedModelEpoch) holds until that addModels is applied → never meshes the new id as AIR.
    expect(logs[0].buildEpochs).toEqual([1, 1, 1]);
    workers[1].flushOne();
    expect(logs[1].buildEpochs).toEqual([1]); // b1 also carried the captured epoch
  });

  it("a build that enters post() BEFORE any addModels captures epoch 0 (waits for nothing)", () => {
    const log = emptyModelLog();
    const pool = new WorkerPool({
      mode: "worker", init: {}, size: 1, workStealing: true,
      createWorker: () => makeModelAwareWorker(log) as unknown as Worker,
    });
    pool.onMessage(() => {});
    pool.post({ type: "build", jobId: 1, snapshot: snapshot(1) }); // no addModels yet → modelEpochA still 0
    expect(log.buildEpochs).toEqual([0]); // captured 0 ⇒ the worker gate (0 > 0 is false) never blocks
  });

  it("capture-at-enqueue: a LATER addModels does NOT retroactively raise an already-queued build's epoch (TRAP WRK-2.C)", () => {
    const log = emptyModelLog();
    const workers: ReturnType<typeof makeModelAwareWorker>[] = [];
    const pool = new WorkerPool({
      mode: "worker", init: {}, size: 1, workStealing: true,
      createWorker: () => { const w = makeModelAwareWorker(log); workers.push(w); return w as unknown as Worker; },
    });
    pool.onMessage(() => {});
    pool.post({ type: "addModels", models: [] }); // modelEpochA → 1
    pool.post({ type: "build", jobId: 1, snapshot: snapshot(1) }); // dispatched immediately, stamped epoch 1
    pool.post({ type: "build", jobId: 2, snapshot: snapshot(1) }); // worker busy → queued in the deque, captured epoch 1
    expect(log.buildEpochs).toEqual([1]); // only b1 dispatched so far (1 worker, ≤1 in-flight)
    // The queued b2 already holds its CAPTURED epoch (1) — frozen at enqueue, on the pendingBuilds entry.
    const pending = (pool as unknown as { pendingBuilds: { jobId: number; epoch?: number }[] }).pendingBuilds;
    expect(pending.map((b) => ({ jobId: b.jobId, epoch: b.epoch }))).toEqual([{ jobId: 2, epoch: 1 }]);

    pool.post({ type: "addModels", models: [] }); // modelEpochA → 2 AFTER b2 was enqueued
    // The later epoch-2 addModels must NOT retroactively raise b2's frozen epoch (it doesn't need those models).
    expect(pending[0].epoch).toBe(1);

    workers[0].flushOne(); // b1 done → b2 is now dispatched to the worker, carrying its CAPTURED epoch 1 (not 2)
    expect(log.buildEpochs).toEqual([1, 1]);
  });

  it("flag-OFF (round-robin): builds carry NO epoch — byte-identical dispatch", () => {
    const logs = [emptyModelLog(), emptyModelLog()];
    let i = 0;
    const pool = new WorkerPool({ mode: "worker", init: {}, size: 2, createWorker: () => makeModelAwareWorker(logs[i++]) as unknown as Worker });
    // workStealing left false → exact rr++ path, no epoch scalar bumped.
    pool.post({ type: "addModels", models: [] }); // Model A flag-off: addModels carries NO epoch
    expect(logs[0].addModels).toEqual([-1]); // -1 sentinel = epoch field was undefined
    expect(logs[1].addModels).toEqual([-1]);
    for (let j = 0; j < 4; j++) pool.post({ type: "build", jobId: j, snapshot: snapshot(1) });
    expect((pool as unknown as { pendingBuilds: unknown[] }).pendingBuilds.length).toBe(0); // no deque used
    expect(logs[0].buildEpochs).toEqual([undefined, undefined]); // jobs 0,2 → w0, NO epoch attached
    expect(logs[1].buildEpochs).toEqual([undefined, undefined]); // jobs 1,3 → w1, NO epoch attached
  });

  it("dispose resets modelEpochA and clears the cancelled set (a re-created pool starts clean)", () => {
    const log = emptyModelLog();
    const pool = new WorkerPool({ mode: "worker", init: {}, size: 1, workStealing: true, createWorker: () => makeModelAwareWorker(log) as unknown as Worker });
    pool.post({ type: "addModels", models: [] }); // modelEpochA → 1
    pool.post({ type: "addModels", models: [] }); // modelEpochA → 2
    pool.cancel(123); // leaves a jobId in the main-side cancelled set
    expect((pool as unknown as { modelEpochA: number }).modelEpochA).toBe(2);
    expect((pool as unknown as { cancelled: Set<number> }).cancelled.size).toBe(1);
    pool.dispose();
    expect((pool as unknown as { modelEpochA: number }).modelEpochA).toBe(0);
    expect((pool as unknown as { cancelled: Set<number> }).cancelled.size).toBe(0);
  });

  it("a worker constructed with workStealing:true uses the deque (option wires the public field)", () => {
    const logs = [emptyModelLog(), emptyModelLog()];
    let i = 0;
    const pool = new WorkerPool({ mode: "worker", init: {}, size: 2, workStealing: true, createWorker: () => makeModelAwareWorker(logs[i++]) as unknown as Worker });
    expect((pool as unknown as { workStealing: boolean }).workStealing).toBe(true);
    // 3 builds, 2 workers → the 3rd sits in the deque (proving the deque, not rr, is active).
    for (let j = 0; j < 3; j++) pool.post({ type: "build", jobId: j, snapshot: snapshot(1) });
    expect((pool as unknown as { pendingBuilds: unknown[] }).pendingBuilds.length).toBe(1);
  });
});

/** A fake worker for Model B that only records init (incl. whether it got the SAB handles) — it does NOT
 *  drain the ring, so the pool's enqueue/fallback behaviour is observable without a real worker thread. */
function makeSharedFakeWorker(log: { inits: number; gotShared: number }): Worker {
  return {
    onmessage: null,
    postMessage(req: WorkerRequest & { shared?: unknown }) {
      if (req.type === "init") { log.inits++; if (req.shared) log.gotShared++; }
    },
    terminate() {},
  } as unknown as Worker;
}

describe("WorkerPool (shared / Model B)", () => {
  it("hands each worker the SAB handles, enqueues builds into the ring, and falls back to inline ONLY when the pool is full", async () => {
    const queue = SharedMeshQueue.create({ apron: 1, slotCap: 8 });
    const log = { inits: 0, gotShared: 0 };
    const pool = new WorkerPool({ mode: "shared", init: {}, size: 2, queue, fallbackMesh: fakeMesh, createWorker: () => makeSharedFakeWorker(log) });
    expect(log.inits).toBe(2);
    expect(log.gotShared).toBe(2); // both workers received the shared ring handles at init

    const got: WorkerResponse[] = [];
    pool.onMessage((r) => got.push(r));
    // Fill all 8 slots — these go into the SAB ring (a real worker would drain them); nothing is delivered.
    for (let i = 0; i < 8; i++) pool.post({ type: "build", jobId: i, snapshot: snapshot(1) });
    await flush();
    expect(got).toHaveLength(0);

    // 9th build: the slot pool is full → mesh on the main thread (never dropped) → delivered as buildDone.
    pool.post({ type: "build", jobId: 99, snapshot: snapshot(1) });
    await flush();
    expect(got).toHaveLength(1);
    expect(got[0].type).toBe("buildDone");
    if (got[0].type === "buildDone") expect(got[0].jobId).toBe(99);

    // Drain one job (simulate a worker) → a slot frees → the next build goes into the ring, NOT the fallback.
    const j = queue.dequeue();
    queue.release(j!.slot, j!.estimateBytes);
    pool.post({ type: "build", jobId: 100, snapshot: snapshot(1) });
    await flush();
    expect(got).toHaveLength(1); // still 1 — job 100 enqueued into the SAB, not inline-fallen-back
  });

  it("addModels bumps the shared model epoch so later builds gate on the new models (Model B ordering)", () => {
    const queue = SharedMeshQueue.create({ apron: 1, slotCap: 8 });
    const pool = new WorkerPool({ mode: "shared", init: {}, size: 1, queue, fallbackMesh: fakeMesh, createWorker: () => makeSharedFakeWorker({ inits: 0, gotShared: 0 }) });
    expect(queue.modelEpoch()).toBe(0);
    pool.post({ type: "addModels", models: [] });
    expect(queue.modelEpoch()).toBe(1); // bumped BEFORE any later build is enqueued
    pool.post({ type: "build", jobId: 1, snapshot: snapshot(1) });
    expect(queue.dequeue()!.enqueueEpoch).toBe(1); // the build will wait for epoch-1 models before meshing
  });
});
