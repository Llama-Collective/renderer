// Priority + aging task queue feeding the (currently inline) mesher. RENDERER_PLAN.md §16; AUDIT N1/SCHED.
//
// COALESCED by section (AUDIT N1): a section has AT MOST ONE queued task — re-dirtying it before it
// dispatches upgrades the existing task in place (newest generation, strongest kind, highest priority,
// earliest enqueue frame) instead of queuing a redundant doomed build. This is the load-bearing dynamic
// fix: under a redstone clock the same section re-dirties faster than it meshes, and without coalescing
// the budget would run stale builds that `acceptBuild` then discards (TRAP 1.A).
//
// Priority (high->low): near/visible edits > visible rebuild > offscreen rebuild > distant sort.
// TRAP 16.B: AGE waiting tasks (eff = base + age) so a busy near-camera region can't starve offscreen
//            work forever. TRAP 16.A: nothing is silently dropped — pop returns until the queue is empty.

import { parseSectionKey, type SectionKey } from "../world/SectionKey";
import type { DirtyReason, Generation, Vec3i } from "../types";

export interface MeshTask {
  sectionKey: SectionKey;
  generation: Generation;
  kind: "build" | "sort";
  reason: DirtyReason;
  basePriority: number;
  enqueuedFrame: number;
}

/** Priority gained per frame a task waits, so deferred work can't starve (TRAP 16.B). */
const AGE_WEIGHT = 1;
/** SCHED-4 near-first weight: penalty per section of Chebyshev distance from the camera. Chebyshev is
 *  bounded by the render distance (~tens of sections), so this only TIE-BREAKS among equal base+age tasks
 *  (e.g. one edit burst, all at EDIT_PRIORITY) — it never flips a high-priority edit below a low-priority
 *  backfill (the gap there is ≫ the max penalty). The result: a burst meshes near sections first. */
const NEAR_WEIGHT = 1;

/** Internal: a queued task with its section coords parsed ONCE (P1.2). The near-camera tie-break in `pop`
 *  reads these cached ints instead of re-splitting the string key for every queued task on every pop —
 *  the O(n·k) string-split churn the audit flagged on bursts (H5). */
interface StoredTask extends MeshTask {
  sx: number;
  sy: number;
  sz: number;
}

export class TaskQueue {
  private readonly tasks = new Map<SectionKey, StoredTask>();

  /** Enqueue, COALESCING with any existing task for the same section (N1 — idempotent OR-join). */
  enqueue(task: MeshTask): void {
    const existing = this.tasks.get(task.sectionKey);
    if (!existing) {
      const [sx, sy, sz] = parseSectionKey(task.sectionKey); // parse ONCE on first enqueue, never per pop
      this.tasks.set(task.sectionKey, { ...task, sx, sy, sz });
      return;
    }
    existing.generation = task.generation; // always dispatch the NEWEST generation (supersede in place)
    if (task.kind === "build") existing.kind = "build"; // build outranks a pending sort-only
    if (task.basePriority > existing.basePriority) existing.basePriority = task.basePriority;
    if (task.enqueuedFrame < existing.enqueuedFrame) existing.enqueuedFrame = task.enqueuedFrame; // age from first dirty
  }

  /** Pop the highest effective-priority task (base + age boost − near-camera tie-break), or undefined if
   *  empty. `cameraSection` (SCHED-4) prefers sections nearer the camera among equal base+age tasks. */
  pop(currentFrame: number, cameraSection?: Vec3i): MeshTask | undefined {
    let bestKey: SectionKey | undefined;
    let best = -Infinity;
    for (const [key, t] of this.tasks) {
      let eff = t.basePriority + (currentFrame - t.enqueuedFrame) * AGE_WEIGHT;
      // SCHED-4 near-first tie-break — Chebyshev distance from the task's CACHED coords (P1.2: no per-pop split).
      if (cameraSection) eff -= NEAR_WEIGHT * Math.max(Math.abs(t.sx - cameraSection[0]), Math.abs(t.sy - cameraSection[1]), Math.abs(t.sz - cameraSection[2]));
      if (eff > best) {
        best = eff;
        bestKey = key;
      }
    }
    if (bestKey === undefined) return undefined;
    const t = this.tasks.get(bestKey)!;
    this.tasks.delete(bestKey);
    return t;
  }

  /** Drop a queued task (e.g. the section was disposed). */
  remove(key: SectionKey): void {
    this.tasks.delete(key);
  }

  has(key: SectionKey): boolean {
    return this.tasks.has(key);
  }

  get size(): number {
    return this.tasks.size;
  }
}
