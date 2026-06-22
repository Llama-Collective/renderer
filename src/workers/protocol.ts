// Worker <-> main message protocol. RENDERER_PLAN.md §6.
//
// Snapshots go IN, build/sort outputs come OUT — both as transferables (no structured
// clone of big arrays). Every message carries the generation so the main thread can
// discard stale results (§4, §5).

import type { Generation } from "../types";
import type { SectionKey } from "../world/SectionKey";
import type { SectionSnapshot } from "../world/SnapshotSource";
import type { BuildPayload } from "./BuildOutput";
import type { SharedMeshQueueHandles } from "./SharedMeshQueue";

export type WorkerRequest =
  // `config` is the MeshInitPayload (transferred baked palette models + fluids — see MeshInit). `shared`,
  // when present (Model B), carries the SharedMeshQueue SABs — the worker then drains build jobs from the
  // shared ring (zero-copy snapshots) instead of receiving them as cloned `build` messages (Model A).
  // `cancelFlags` (WRK-3, Model A, default-off) is the cooperative-cancel SAB + this worker's slot index;
  // absent ⇒ the worker attaches a no-op shim and meshing is byte-identical to today.
  | { type: "init"; config: unknown; shared?: SharedMeshQueueHandles; cancelFlags?: SharedArrayBuffer; workerIndex?: number }
  // Incremental palette growth (Phase 5): baked models for block ids first seen AFTER init, so a newly
  // placed block meshes off-thread instead of rendering as air. `models` is a MeshModels array. `epoch`
  // (Model B) is the SharedMeshQueue model-epoch these models satisfy — the worker records it as applied so
  // it won't mesh a SAB job stamped with a higher epoch before the models land (Model A leaves it undefined).
  | { type: "addModels"; models: unknown; epoch?: number }
  // `epoch` (WRK-2, Model A + work-stealing, default-off) is the pool-side `modelEpochA` CAPTURED when this
  // build entered `post()` — the highest `addModels` it needs. The executing worker (which may have STOLEN
  // the build off another worker's deque) gates on it: it won't `meshSection` until its `appliedModelEpoch`
  // reaches it, so a newly-minted palette id can never mesh as AIR. Absent (rr path / flag-off) ⇒ no gate.
  | { type: "build"; jobId: number; snapshot: SectionSnapshot; epoch?: number }
  | { type: "sort"; jobId: number; sectionKey: SectionKey; generation: Generation; sortData: ArrayBuffer; cameraPos: readonly [number, number, number] }
  | { type: "cancel"; jobId: number };

/** Off-thread re-sort result (TR-1): a new translucent INDEX order for a section, generation-tagged so the
 *  main thread can drop it if the section advanced while the worker sorted (TRAP 5.B). */
export interface SortDonePayload {
  sectionKey: SectionKey;
  generation: Generation;
  indexData: ArrayBuffer;
}

export type WorkerResponse =
  | { type: "buildDone"; jobId: number; payload: BuildPayload }
  | { type: "sortDone"; jobId: number; payload: SortDonePayload }
  | { type: "cancelled"; jobId: number };
