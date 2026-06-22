// Per-frame upload policy: drain completed worker outputs, then commit pending sections
// within a byte AND time budget. RENDERER_PLAN.md §5, §11, §16.
//
// This is the frame-level enforcement of the Stable Presentation Invariant:
//  - draining calls SectionStore.acceptBuild/acceptSort, which discards stale-generation
//    outputs BEFORE upload (Invariant 2);
//  - commits are ordered by priority with aging so no section starves (TRAP 16.B);
//  - the byte/time budget caps work per frame — anything not committed stays pending and
//    is retried next frame, keeping old data drawable (Invariant 4);
//  - deferrals are reported via onDeferred / the returned stats — never silently dropped
//    (TRAP 16.A).
//
// It owns the budget as the single authority and drives the GPU only through SectionStore,
// so it needs no device reference. Pair it with a GpuSectionUploader constructed with the
// default (unbounded) budget — the scheduler, not the uploader, gates throughput.

import type { RenderSection } from "../world/RenderSection";
import type { SectionStore } from "../world/SectionStore";
import type { SectionBuildOutput } from "../workers/BuildOutput";
import type { SortOutput } from "../workers/SortOutput";

export interface UploadBudget {
  maxBytesPerFrame: number;
  maxMillisPerFrame: number;
}

export const DEFAULT_UPLOAD_BUDGET: UploadBudget = {
  maxBytesPerFrame: 4 * 1024 * 1024, // 4 MB — TUNE (§11)
  maxMillisPerFrame: 2,
};

export interface UploadStats {
  committed: number;
  bytesUploaded: number;
  discardedStale: number;
  deferredBudget: number;
  deferredTime: number;
}

export interface UploadSchedulerOptions {
  budget?: UploadBudget;
  /** Injectable clock for deterministic tests. Defaults to performance.now(). */
  now?: () => number;
  /** Higher commits first. Plug in distance-to-camera here (§16). Default: 0 (FIFO+aging). */
  priorityOf?: (section: RenderSection) => number;
  /** Called when work was deferred this frame, for logging/HUD (no silent caps). */
  onDeferred?: (stats: UploadStats) => void;
}

/** Per-frame priority gained per frame spent waiting, so deferred work can't starve. */
const AGE_WEIGHT = 1;

const defaultNow = (): number =>
  typeof performance !== "undefined" ? performance.now() : Date.now();

type InboxItem =
  | { kind: "build"; out: SectionBuildOutput }
  | { kind: "sort"; out: SortOutput };

function estimatePendingBytes(s: RenderSection): number {
  let bytes = 0;
  if (s.pendingBuild) bytes += s.pendingBuild.approxBytes;
  if (s.pendingSort) bytes += s.pendingSort.indexData.byteLength;
  return bytes;
}

function hasPending(s: RenderSection): boolean {
  return s.pendingBuild !== null || s.pendingSort !== null;
}

export class UploadScheduler {
  private budget: UploadBudget;
  private readonly now: () => number;
  private readonly priorityOf?: (section: RenderSection) => number;
  private readonly onDeferred?: (stats: UploadStats) => void;

  private readonly inbox: InboxItem[] = [];
  private readonly pending = new Set<RenderSection>();
  private readonly firstPendingFrame = new Map<RenderSection, number>();

  constructor(
    private readonly store: SectionStore,
    opts: UploadSchedulerOptions = {},
  ) {
    // INFRA-5: the scheduler is the SINGLE upload-budget authority. A scheduler-driven uploader must be
    // unbounded (Infinity) so throughput isn't gated in two places — the uploader's reuse-adjusted byte
    // check silently interacting with this gross-byte frame budget. (A standalone screenshot/item-slot
    // uploader may legitimately be finite; it just isn't paired with a scheduler, so it never reaches here.)
    const uploaderBudget = store.uploaderBudgetBytes();
    if (uploaderBudget !== undefined && uploaderBudget !== Infinity) {
      throw new Error(
        `UploadScheduler is the single upload-budget authority, but its uploader has a finite per-frame ` +
          `budget (${uploaderBudget} bytes). Construct the GpuSectionUploader with the default (Infinity) ` +
          `budget — the scheduler, not the uploader, gates throughput (INFRA-5).`,
      );
    }
    this.budget = opts.budget ?? DEFAULT_UPLOAD_BUDGET;
    this.now = opts.now ?? defaultNow;
    this.priorityOf = opts.priorityOf;
    this.onDeferred = opts.onDeferred;
  }

  /** ABP-1: retune the per-frame upload budget at runtime (adaptive sizing from a frame-time EMA). Merges
   *  onto the current budget so a caller can adjust just `maxMillisPerFrame` or just `maxBytesPerFrame`.
   *  Takes effect on the NEXT `drainAndCommit` (the in-flight frame keeps the budget it started with). */
  setBudget(budget: Partial<UploadBudget>): void {
    this.budget = { ...this.budget, ...budget };
  }

  /** The current per-frame upload budget (after any ABP-1 retune). */
  get currentBudget(): UploadBudget {
    return this.budget;
  }

  /** Queue a completed build output for this frame's drain. */
  submitBuild(out: SectionBuildOutput): void {
    this.inbox.push({ kind: "build", out });
  }

  /** Queue a completed index-only sort output for this frame's drain. */
  submitSort(out: SortOutput): void {
    this.inbox.push({ kind: "sort", out });
  }

  /** Number of outputs waiting to commit: staged-but-undrained (`inbox`, e.g. an off-thread build/sort reply
   *  that landed since the last `drainAndCommit`) PLUS accepted-but-budget-deferred (`pending`, carried over
   *  from a prior frame). The render-on-demand loop reads `> 0` as a liveness signal so it keeps drawing
   *  until every staged/deferred upload has actually committed — without this, a worker reply that stages
   *  into `inbox` (which empties `inFlight`) would leave the loop idling with the build never drawn. */
  get pendingCount(): number {
    return this.inbox.length + this.pending.size;
  }

  /**
   * Drain completed outputs (accept/discard-stale), then commit pending sections in
   * priority order within the byte+time budget. Returns this frame's stats.
   */
  drainAndCommit(currentFrame: number): UploadStats {
    const stats: UploadStats = {
      committed: 0,
      bytesUploaded: 0,
      discardedStale: 0,
      deferredBudget: 0,
      deferredTime: 0,
    };

    // 1) Drain inbox: accept (stages pending) or discard stale (disposes output).
    for (const item of this.inbox) {
      const res =
        item.kind === "build"
          ? this.store.acceptBuild(item.out)
          : this.store.acceptSort(item.out);
      if (res === "discarded-stale") {
        stats.discardedStale++;
        continue;
      }
      const section = this.store.get(item.out.sectionKey);
      if (section && !this.pending.has(section)) {
        this.pending.add(section);
        this.firstPendingFrame.set(section, currentFrame);
      }
    }
    this.inbox.length = 0;

    // 2) Order by effective priority (base + age). Higher first.
    const ordered = [...this.pending].sort(
      (a, b) => this.effectivePriority(b, currentFrame) - this.effectivePriority(a, currentFrame),
    );

    // 3) Commit within byte + time budget.
    let remainingBytes = this.budget.maxBytesPerFrame;
    const frameStart = this.now();
    for (let i = 0; i < ordered.length; i++) {
      const section = ordered[i];

      if (this.now() - frameStart >= this.budget.maxMillisPerFrame) {
        stats.deferredTime += ordered.length - i; // the rest wait for next frame
        break;
      }

      const bytes = estimatePendingBytes(section);
      if (!hasPending(section)) {
        this.drop(section); // nothing actually pending (e.g. superseded during drain)
        continue;
      }
      // NOTE: a 0-byte pending build is NOT "nothing to do" — it's an EMPTIED section (last block
      // removed). It must still commit so `presented` is cleared; dropping it would leave stale geometry.

      // Always let the first commit through so a large section can't starve forever
      // ("emergency" allowance, §16); enforce strictly after that.
      if (bytes > remainingBytes && stats.committed > 0) {
        stats.deferredBudget++;
        continue;
      }

      const committed = this.store.commit(section);
      if (committed) {
        stats.committed++;
        stats.bytesUploaded += bytes;
        remainingBytes -= bytes;
      }
      if (!hasPending(section)) this.drop(section);
    }

    if (stats.deferredBudget + stats.deferredTime > 0) this.onDeferred?.(stats);
    return stats;
  }

  private effectivePriority(section: RenderSection, currentFrame: number): number {
    const base = this.priorityOf?.(section) ?? 0;
    const since = this.firstPendingFrame.get(section) ?? currentFrame;
    return base + (currentFrame - since) * AGE_WEIGHT;
  }

  private drop(section: RenderSection): void {
    this.pending.delete(section);
    this.firstPendingFrame.delete(section);
  }
}
