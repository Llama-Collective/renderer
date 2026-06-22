import { describe, it, expect } from "vitest";
import { TaskQueue } from "./TaskQueue";
import { sectionKey } from "../world/SectionKey";
import { DirtyReason } from "../types";

const task = (kx: number, gen: number, basePriority = 0, enqueuedFrame = 0, kind: "build" | "sort" = "build") => ({
  sectionKey: sectionKey(kx, 0, 0),
  generation: gen,
  kind,
  reason: DirtyReason.Simulation,
  basePriority,
  enqueuedFrame,
});

describe("TaskQueue", () => {
  it("coalesces re-enqueues of the same section into ONE task (N1)", () => {
    const q = new TaskQueue();
    q.enqueue(task(0, 1));
    q.enqueue(task(0, 2));
    q.enqueue(task(0, 3));
    expect(q.size).toBe(1);
    const popped = q.pop(0)!;
    expect(popped.generation).toBe(3); // newest generation wins
    expect(q.size).toBe(0);
  });

  it("coalesce keeps the strongest kind, highest priority, earliest enqueue frame", () => {
    const q = new TaskQueue();
    q.enqueue(task(0, 1, 5, 10, "sort"));
    q.enqueue(task(0, 2, 9, 3, "build"));
    const t = q.pop(20)!;
    expect(t.kind).toBe("build"); // build outranks sort
    expect(t.basePriority).toBe(9); // higher priority kept
    expect(t.enqueuedFrame).toBe(3); // earliest frame kept (age from first dirty)
  });

  it("a re-enqueue RAISES priority but never lowers it (no inversion) while bumping the generation", () => {
    const q = new TaskQueue();
    q.enqueue(task(0, 1, 100)); // first dirty at priority 100
    q.enqueue(task(0, 2, 1000)); // re-dirty at a HIGHER priority → must raise (the inversion fix)
    q.enqueue(task(0, 3, 50)); // re-dirty at a LOWER priority → must NOT lower
    expect(q.size).toBe(1);
    const t = q.pop(0)!;
    expect(t.basePriority).toBe(1000); // highest priority kept across re-dirties
    expect(t.generation).toBe(3); // newest generation still wins
  });

  it("pop returns the highest effective priority (base + age)", () => {
    const q = new TaskQueue();
    q.enqueue(task(0, 1, 0, 0)); // base 0, but enqueued at frame 0 → ages
    q.enqueue(task(1, 1, 5, 99)); // base 5, enqueued at frame 99 → no age yet
    // At frame 100: section0 eff = 0 + 100 = 100; section1 eff = 5 + 1 = 6 → section0 wins.
    expect(q.pop(100)!.sectionKey).toBe(sectionKey(0, 0, 0));
  });

  it("SCHED-4: among equal-priority tasks, the section nearer the camera pops first", () => {
    const q = new TaskQueue();
    q.enqueue(task(10, 1, 1000, 0)); // section (10,0,0), edit priority
    q.enqueue(task(1, 1, 1000, 0)); // section (1,0,0), edit priority — nearer to a camera at section 0
    expect(q.pop(0, [0, 0, 0])!.sectionKey).toBe(sectionKey(1, 0, 0)); // nearest first
    expect(q.pop(0, [0, 0, 0])!.sectionKey).toBe(sectionKey(10, 0, 0));
  });

  it("SCHED-4: the near-camera tie-break never demotes a higher-priority edit below a far backfill", () => {
    const q = new TaskQueue();
    q.enqueue(task(50, 1, 1000, 0)); // FAR edit (base 1000) → eff 1000 − 50 = 950
    q.enqueue(task(0, 1, 0, 0)); // NEAR backfill (base 0) → eff 0 − 0 = 0
    expect(q.pop(0, [0, 0, 0])!.sectionKey).toBe(sectionKey(50, 0, 0)); // priority class wins over distance
  });
});
