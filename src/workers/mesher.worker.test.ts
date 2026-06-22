// WRK-2: worker-side addModels model-epoch GATE for the Model-A `build` path.
//
// Drives the REAL `mesher.worker.ts` module (its `onmessage`) with a mocked `self`, so the assertion is on
// the shipped gate, not a re-implementation. A Model-A build stamped with `epoch > appliedModelEpoch` must
// NOT post `buildDone` until the matching `addModels` (carrying that epoch) is delivered — mirroring the SAB
// drain loop's `while (job.enqueueEpoch > appliedModelEpoch) await nextMacrotask()`. A build with NO epoch
// (rr / flag-off) must mesh immediately (byte-identical, no gate). Bounded macrotask-yield (NOT a blocking
// spin — TRAP WRK-2.B): the pending addModels postMessage is delivered between yields.

import { describe, it, expect, beforeEach, vi } from "vitest";
import type { WorkerRequest, WorkerResponse } from "./protocol";
import { sectionKey } from "../world/SectionKey";
import type { Generation } from "../types";
import type { SectionSnapshot } from "../world/SnapshotSource";

// ── Mock `self` BEFORE importing the worker module (the module body sets self.onmessage at import). ──
type Handler = (e: MessageEvent<WorkerRequest>) => void | Promise<void>;
const posted: WorkerResponse[] = [];
const selfMock: { onmessage: Handler | null; postMessage: (m: unknown, t: Transferable[]) => void } = {
  onmessage: null,
  postMessage: (m) => { posted.push(m as WorkerResponse); },
};
vi.stubGlobal("self", selfMock);

// A minimal valid init: empty palette + no fluids ⇒ an empty snapshot meshes to an empty (but valid) output
// and posts `buildDone`. Model A (`shared` absent, `cancelFlags` absent) ⇒ the plain synchronous build path
// guarded only by the WRK-2 epoch gate.
const INIT: WorkerRequest = { type: "init", config: { models: [] } };

function emptySnapshot(): SectionSnapshot {
  return { sectionKey: sectionKey(0, 0, 0), generation: 1 as Generation, origin: [0, 0, 0], size: 16, apron: 1, blocks: new Uint32Array(0), changedReason: 0 };
}

/** Deliver a message to the worker and await any async build gate it may enter. */
async function deliver(req: WorkerRequest): Promise<void> {
  await selfMock.onmessage?.({ data: req } as MessageEvent<WorkerRequest>);
}

/** Yield enough macrotasks that a pending `nextMacrotask()` (setTimeout 0) in the worker can resolve. */
function macrotask(): Promise<void> {
  return new Promise((r) => setTimeout(r, 0));
}

describe("mesher.worker — WRK-2 Model-A addModels epoch gate", () => {
  beforeEach(async () => {
    posted.length = 0;
    // The worker module is imported once (module-level singleton). Re-`init` resets its per-run state that
    // matters here: appliedModelEpoch is monotonic, so each test uses STRICTLY HIGHER epochs to stay above it.
    await import("./mesher.worker");
    await deliver(INIT);
  });

  it("a build with epoch > appliedModelEpoch does NOT mesh until the matching addModels lands", async () => {
    // Pick an epoch comfortably above any epoch a prior test applied (appliedModelEpoch is module-global +
    // monotonic). Use a large gap so this test is order-independent.
    const EPOCH = 1000;
    // Build arrives BEFORE its addModels (as if a steal routed it to a worker that hasn't merged the models).
    const buildDone = deliver({ type: "build", jobId: 42, snapshot: emptySnapshot(), epoch: EPOCH });

    // Give the gate several macrotask yields: it must STILL be waiting (no buildDone) — the addModels hasn't come.
    await macrotask();
    await macrotask();
    expect(posted).toHaveLength(0); // GATED: no buildDone while epoch > appliedModelEpoch

    // Now the satisfying addModels (carrying the epoch) is delivered → it bumps appliedModelEpoch.
    await deliver({ type: "addModels", models: [], epoch: EPOCH });

    // The next macrotask lets the gated build resume and post exactly one buildDone.
    await buildDone;
    expect(posted).toHaveLength(1);
    expect(posted[0].type).toBe("buildDone");
    if (posted[0].type === "buildDone") expect(posted[0].jobId).toBe(42);
  });

  it("a build with NO epoch (rr / flag-off) meshes immediately — the gate is skipped (byte-identical)", async () => {
    await deliver({ type: "build", jobId: 7, snapshot: emptySnapshot() }); // epoch undefined
    expect(posted).toHaveLength(1); // posted synchronously within the same onmessage turn (no await)
    expect(posted[0].type).toBe("buildDone");
    if (posted[0].type === "buildDone") expect(posted[0].jobId).toBe(7);
  });

  it("a build whose epoch is already satisfied meshes without waiting", async () => {
    const EPOCH = 2000;
    await deliver({ type: "addModels", models: [], epoch: EPOCH }); // apply the models FIRST
    await deliver({ type: "build", jobId: 9, snapshot: emptySnapshot(), epoch: EPOCH });
    // epoch (2000) is NOT > appliedModelEpoch (2000) ⇒ the loop never runs ⇒ buildDone posted right away.
    expect(posted).toHaveLength(1);
    expect(posted[0].type).toBe("buildDone");
    if (posted[0].type === "buildDone") expect(posted[0].jobId).toBe(9);
  });
});
