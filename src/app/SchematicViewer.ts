// The single app-facing façade over the WebGPU renderer — the drop-in replacement for the old
// three.js `EditableWorldRenderer`. Owns the device + render loop, the orbit camera + input, the
// WorldModel (authoritative block state), the SectionStore, the EntityWorld, and the editor
// interaction surface (picking, placement, overlays, explosions). RENDERER_INTEGRATION_PLAN Steps 2–9.
//
// Lifecycle: `new SchematicViewer(canvas)` is synchronous (matches the old API); device creation +
// scene build are async and buffered behind `ready`/`sceneReady`, and all world mutations are
// serialized through `opChain` so a sim diff that arrives before the scene is ready still applies in
// order (the tick-0 invariant). Every mutation funnels through `applyBlockEdits` → the
// markDirty → meshSection → acceptBuild → commit loop, preserving the Stable Presentation Invariant.

import { createDevice, type CanvasDevice } from "../core/createDevice";
import { BackendKind } from "../core/gpu-enums";
import { instrument } from "../core/Instrument";
import { Camera } from "../camera/Camera";
import { DirtyTracking, type BlockEdit } from "../world/DirtyTracking";
import { SnapshotSource, APRON } from "../world/SnapshotSource";
import { meshSection, type MeshOptions } from "../mesh/SectionMesher";
import { SharedMeshQueue, sharedMemoryAvailable } from "../workers/SharedMeshQueue";
import type { SectionStore } from "../world/SectionStore";
import { sectionOfBlock, sectionKey, type SectionKey } from "../world/SectionKey";
import { SortType } from "../mesh/SortTypes";
import { serializeQuads, uniquePlanes, crossedAnyPlane, farAngleResortAllowed, GATE_QUAD_THRESHOLD, GATE_DIST_EPS, GATE_ANGLE_COS, GATE_CENTER_LOCAL } from "../mesh/SortTrigger";
import { Scheduler } from "../scheduling/Scheduler";
import { UploadScheduler, DEFAULT_UPLOAD_BUDGET, type UploadBudget } from "../render/UploadScheduler";
import { FrameTimeEma } from "../scheduling/FrameTimeEma";
import { isContinuous, shouldRenderFrame, pickFrameSchedule, idlePollDelayMs, type ContinuousState } from "../scheduling/RenderGate";
import { WorkerPool } from "../workers/WorkerPool";
import { extractMeshInit, extractPaletteModels, unknownAtlasedPaletteIds } from "../workers/MeshInit";
import type { FluidContext } from "../mesh/FluidMesher";
import type { SectionDraw } from "../render/TerrainRenderer";
import { buildLightLut, type LightLutParams } from "../render/LightLut";
import type { EntityWorld } from "../render/entities/EntityWorld";
import type { EntitySnapshot } from "../render/entities/EntityScene";
import { OverlayRenderer } from "../render/overlay/OverlayRenderer";
import { ExplosionEffects } from "../render/effects/ExplosionEffects";
import { PistonTransients, type MovingPistonInput } from "../render/transient/PistonTransients";
import { DirtyReason, TerrainPass, type Vec3 } from "../types";
import { WorldModel, loadWorld, AIR } from "./WorldModel";
import { buildScene, buildEntityWorld, DrawList, commitWorld, makeFluidContext, FLUID_SPRITES, ALWAYS_ATLASED_BLOCK_NAMES, occlusionVisibleSet, occlusionHomeSection, filterDrawsByOcclusion, type Scene, type RegionCullOptions } from "./scene";
import { Frustum } from "../camera/Frustum";
import { hasBlockEntityModel } from "../render/entities/blockentities";
import { computeBeaconBeam } from "./beaconBeam";
import { pickWorld, entityBoundsSize, type Ray, type OutlineBox } from "./pick";
import { getPlacementHint as computePlacementHint, regionFacingMap as computeRegionFacingMap } from "./placement";
import type { EditableSchematic, EditableBlock, EditableEntity, PickResult, PlacementHint, FaceRegion, Vec3Tuple } from "../../../schematic-io/types";

const MAX_DIM = 8192;
/** Base priority for an edit-driven remesh — high so a few edits mesh the same frame (TaskQueue). */
const EDIT_PRIORITY = 1000;
const FOV_Y = (55 * Math.PI) / 180; // match the old renderer's PerspectiveCamera(55°) for placement/pick parity
/** TR-6: max off-thread translucent re-sorts dispatched per frame (bounds a camera sweep that crosses many
 *  glass planes at once; the rest fire next frame, the old committed index showing meanwhile — no pop). */
const SORT_DISPATCH_CAP = 16;

// Camera keybinds (status bar: "WASD pan · R/F up·down"). WASD pans on the ground plane (forward/back
// relative to the camera's horizontal facing, A/D strafe); R/F move straight up/down; Shift sprints.
const PAN_KEYS = new Set(["w", "a", "s", "d", "r", "f", "shift"]);

/** True when the page/tab is not visible (background tab, minimized window). Guards `document` for non-DOM
 *  contexts (unit tests). Used by the dynamicFps loop to stop rendering a tab nobody is looking at. */
function isPageHidden(): boolean {
  return typeof document !== "undefined" && document.visibilityState === "hidden";
}

/** True when a keystroke is destined for a focused text field — don't steal it for camera panning. */
function isTextEntry(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "select" || tag === "textarea" || target.isContentEditable;
}

/** A block change the renderer applies (sim/editor diff). All property values are strings. */
export interface RendererBlockChange {
  x: number;
  y: number;
  z: number;
  name: string;
  properties?: Record<string, string>;
  /** Surface-display items for a block entity that renders its contents on the block (a shelf's 3 slots),
   *  slot-indexed with "" for empty slots. Surfaced to the BE renderer as its `items` prop. */
  items?: string[];
  /** Sign text — the renderer bakes the FRONT lines onto the board (surfaced as the `text` prop). */
  signText?: { front?: { messages?: readonly string[] } };
  /** Container lid OPEN state (chest / trapped / ender / shulker) from the sim's open-viewer count — the
   *  renderer animates the lid open/closed. Absent for non-lid blocks; barrels carry OPEN in `properties`. */
  open?: boolean;
}

/** An entity change (flat sim diff). `position`/`velocity` are 3-number arrays. */
export interface RendererEntityChange {
  id: string;
  type: string;
  position: number[];
  velocity?: number[];
  properties?: Record<string, string>;
}

/** The batched per-tick diff the simulator feeds the renderer. Superset-tolerant (extra fields ignored). */
export interface RendererSimulationDiff {
  setBlocks?: RendererBlockChange[];
  removedBlocks?: { x: number; y: number; z: number }[];
  entities?: RendererEntityChange[];
  removedEntityIds?: string[];
  /** Active moving_piston block entities this tick (empty when none animating). */
  movingPistons?: MovingPistonInput[];
  gameTime?: number;
}

export interface ApplyDiffOptions {
  /** Tick interpolation timing (drives entity/piston partial-tick). */
  timing?: { animate: boolean; interval: number; startedAtMs: number };
  /** Soft CPU budget per call (ms); large diffs yield to the next frame past this. */
  budgetMs?: number;
}

/**
 * Phase-P metrics surface (RENDERER_OPTIMIZATION_PLAN.md §Phase P), sampled per frame. Flat numbers so it
 * serializes straight to the committed baseline JSON + the live HUD. Counters are PER-FRAME deltas
 * (drawCalls/passes/allocs/jobs); `liveBuffers` is an absolute gauge; `resorts` is cumulative.
 */
export interface RenderStats {
  // ── frame time (CPU encode, ms), split by phase. GPU time via timestamp queries is deferred (the
  //    audit lists it "where available"; CPU encode is the bottleneck signal for the dynamic regime). ──
  frameMs: number;
  cullMs: number; // terrain cull + draw-list build (includes resortMs)
  encodeMs: number; // terrain pass encode
  resortMs: number; // translucent index-only re-sort
  entityMs: number; // entity + block-entity pass
  // ── GPU ──
  drawCalls: number;
  passes: number;
  liveBuffers: number;
  /** MEM-2: REAL gpu.createBuffer calls this frame (delta). With the buffer pool on under churn this drops as
   *  pool hits absorb grow/compact reallocations; off it equals the logical create count, as today. */
  createBufferCalls: number;
  /** MEM-2: buffer-pool hits this frame (delta) — recycled GPUBuffer objects (0 when the pool is off). */
  poolHits: number;
  // ── sections ──
  sectionsTotal: number;
  sectionsDrawn: number;
  sectionsCulled: number;
  resorts: number;
  // ── entities ──
  entities: number;
  entityDraws: number;
  beStatic: number;
  beAnimating: number;
  // ── per-frame allocation churn (audit hot paths) ──
  allocRawVertex: number;
  allocDrawList: number;
  allocFrustum: number;
  allocEntityEmit: number;
  // ── per-tick remesh jobs (delta since last frame) ──
  jobsDirtied: number;
  jobsMeshed: number;
  jobsDiscarded: number;
  // ── OCC-1 camera occlusion BFS (delta since last frame) ──
  occBfsRuns: number;
  occVisited: number;
  /** All metrics are flat numbers — also consumable as a generic map (app activity-graph / HUD / JSON). */
  [metric: string]: number;
}

/** OCC-1: the pass-through frustum predicate for a REACHABILITY query — always visible, so the cached BFS
 *  set is orientation-independent (the live frustum is reapplied per frame downstream). Module-level so it is
 *  a stable identity, not a per-frame closure alloc. */
function occAllVisible(): boolean {
  return true;
}

/** OCC-1: membership equality for the reachable-set change check (decides whether to allocate a fresh
 *  `occDraws`). Order-independent; O(size). */
function setsEqual(a: ReadonlySet<number>, b: ReadonlySet<number>): boolean {
  if (a.size !== b.size) return false;
  for (const v of a) if (!b.has(v)) return false;
  return true;
}

/** PREC-1 auto-gate (shared by the class static `CAMERA_RELATIVE_THRESHOLD` and the gate below). The magnitude
 *  where f32's ULP first grows past ~1/1024 block — below it the camera-relative path stays off (byte-identical). */
const CAMERA_RELATIVE_THRESHOLD_BLOCKS = 16384;

/** PREC-1: does any axis of this world-block coordinate sit beyond the auto-gate threshold? */
function farFromOrigin(p: readonly [number, number, number]): boolean {
  return Math.abs(p[0]) > CAMERA_RELATIVE_THRESHOLD_BLOCKS || Math.abs(p[1]) > CAMERA_RELATIVE_THRESHOLD_BLOCKS || Math.abs(p[2]) > CAMERA_RELATIVE_THRESHOLD_BLOCKS;
}

const isBeacon = (name: string): boolean => name === "minecraft:beacon" || name === "beacon";

/** Could an edit at `e` change the beam of the beacon at `b`? — true if it's in the beacon's vertical column
 *  (the beacon cell or anything above it) or in its 4-layer base pyramid below. */
function editAffectsBeacon(e: { x: number; y: number; z: number }, b: { x: number; y: number; z: number }): boolean {
  if (e.x === b.x && e.z === b.z && e.y >= b.y) return true; // the beacon cell or its column above it
  const dy = b.y - e.y;
  return dy >= 1 && dy <= 4 && Math.abs(e.x - b.x) <= dy && Math.abs(e.z - b.z) <= dy; // the base pyramid
}

function emptyStats(): RenderStats {
  return {
    frameMs: 0, cullMs: 0, encodeMs: 0, resortMs: 0, entityMs: 0,
    drawCalls: 0, passes: 0, liveBuffers: 0, createBufferCalls: 0, poolHits: 0,
    sectionsTotal: 0, sectionsDrawn: 0, sectionsCulled: 0, resorts: 0,
    entities: 0, entityDraws: 0, beStatic: 0, beAnimating: 0,
    allocRawVertex: 0, allocDrawList: 0, allocFrustum: 0, allocEntityEmit: 0,
    jobsDirtied: 0, jobsMeshed: 0, jobsDiscarded: 0,
    occBfsRuns: 0, occVisited: 0,
  };
}

export class SchematicViewer {
  // ── Public hooks (set by the app) ───────────────────────────────────────────
  /** Per-block collision shape for picking + cursor fit. */
  outlineProvider: ((x: number, y: number, z: number) => OutlineBox[] | null) | null = null;
  /** FPS callback (≥500ms cadence). */
  onFpsUpdate: ((fps: number) => void) | null = null;
  /** Fatal-device callback: fired when NEITHER WebGPU NOR WebGL2 could be created, so the render loop
   *  never starts (W7). The app surfaces this to the user instead of leaving a silent black canvas. */
  onDeviceError: ((message: string) => void) | null = null;
  /** Fired once with the backend that actually came up (WebGPU or the forced/fallback WebGL2), so a debug
   *  UI can confirm which backend is live. Set before construction completes is impossible (the device is
   *  created in the ctor); set it right after `new SchematicViewer(...)` — it fires on the next microtask. */
  onBackendReady: ((backend: BackendKind) => void) | null = null;
  /** Item ids to build dropped-item geometry for (the app's inventory). Without these, dropped items fall
   *  back to a block model — wrong for "generated" items (a popped lever showing a 3D block, not its icon).
   *  Set before the resource pack loads; the pack-load rebuild picks it up. */
  itemIds: readonly string[] | undefined = undefined;

  // ── Device / loop ────────────────────────────────────────────────────────────
  private readonly canvas: HTMLCanvasElement;
  private device: CanvasDevice | null = null;
  private readonly ready: Promise<void>;
  private disposed = false;
  private rafId = 0;
  /** Render-on-demand idle-poll handle (`dynamicFps` only). At most one of (`rafId`, `pollId`) is pending. */
  private pollId = 0;
  private frames = 0;
  private fpsLast = 0;

  // ── Render-on-demand ("dynamic FPS") ─────────────────────────────────────────
  /** MASTER FLAG (default OFF). Off ⇒ the continuous rAF loop draws every frame, byte-for-byte the behavior
   *  the integration + harness baselines were recorded against (do NOT default this on). On ⇒ frames are
   *  drawn only when something changed — the camera moved, an edit/diff/overlay/brightness change arrived
   *  (via `invalidate()`), or something is animating/meshing — and the loop slow-polls while idle so the
   *  browser frame pipeline goes quiet. Safe to flip at runtime (the next tick re-evaluates). */
  dynamicFps = false;
  /** Animation master switch (default ON ⇒ today's behavior). Off ⇒ DECORATIVE animation freezes — animated
   *  textures stop advancing, entities snap to their target instead of tweening, block-entity motions +
   *  explosions stop. PISTONS are exempt (they convey redstone state); they slide whenever the sim is running
   *  (`simRunning`), regardless of this switch. World/state changes still render. Independent of `dynamicFps`. */
  animationsEnabled = true;
  /** Whether the simulation is currently RUNNING (set by the host). Gates piston slides under render-on-demand:
   *  a piston keeps the loop drawing only while the sim runs, so PAUSING the sim freezes an in-flight piston
   *  (and lets the loop idle) instead of holding it animating forever. Default true so a host that never sets
   *  it (harnesses) keeps the prior behavior (active pistons always animate). */
  simRunning = true;
  /** Stop rendering while the page/tab is hidden (default OFF). Only takes effect with `dynamicFps` on (the
   *  rAF path already halts when hidden; this additionally cancels the dynamic idle-poll so a backgrounded
   *  tab does zero render work). Defaulted off so headless/automated runs are unaffected. */
  pauseWhenHidden = false;
  /** Idle backstop poll rate (Hz) when `dynamicFps` is on and nothing is animating. A safety net that catches
   *  any direct camera/state mutation that forgot to call `invalidate()`; real changes wake instantly. */
  private idlePollFps = 5;
  /** Dirty flag: set by `invalidate()`, cleared once a frame is drawn (`dynamicFps` gate only). Starts true so
   *  the first frame always draws. */
  private needsRender = true;
  /** Active "keep drawing every frame" reasons (e.g. "simulation", "orbit-drag", "pan"). See `setContinuous`. */
  private readonly continuousReasons = new Set<string>();
  /** Hard pause (`pauseRendering`): the loop stops entirely (no rAF, no poll) until `resumeRendering`. Works
   *  in BOTH loop modes — the explicit "lock". */
  private renderingPaused = false;
  /** Camera change-version (`Camera.changeVersion`) at the last DRAWN frame, to detect movement while idle. */
  private lastDrawnCameraVersion = -1;
  /** Set when the dynamicFps loop drops to the idle poll; the next rendered frame reseeds the FPS window so
   *  the perf readout doesn't report a near-zero spike spanning the idle gap. Never set in the default loop. */
  private renderGapPending = false;
  /** Wall-clock ms of the last frame's render (CPU encode), for the perf harness. */
  private lastFrameMs = 0;
  /** CPU ms of the last frame's entity + block-entity pass (Phase-P split). */
  private lastEntityMs = 0;
  /** The latest per-frame metrics snapshot (Phase P). */
  private lastStats: RenderStats = emptyStats();
  /** Cumulative-counter snapshots from the previous frame, to compute per-frame deltas. */
  private readonly prevCounters = { drawCalls: 0, passes: 0, createBufferCalls: 0, poolHits: 0, aRaw: 0, aDraw: 0, aFrustum: 0, aEntity: 0, jDirtied: 0, jMeshed: 0, jDiscarded: 0, jOccBfsRuns: 0, jOccVisited: 0 };

  // ── Camera ───────────────────────────────────────────────────────────────────
  private readonly camera = new Camera();

  // ── World ──────────────────────────────────────────────────────────────────
  private model: WorldModel | null = null;
  private scene: Scene | null = null;
  /** P6/LM-1: current light-LUT params (global brightness / day-night). Persisted here so a scene rebuild —
   *  which creates a fresh renderer with the default daylight LUT — re-applies the user's setting. */
  private lightLutParams: LightLutParams = {};
  private store: SectionStore | null = null;
  private snapshots: SnapshotSource | null = null;
  /** Phase-1 dispatcher: edits enqueue here; the render loop ticks it (coalesced N1 + budgeted SCHED-3),
   *  routing commits through the UploadScheduler (U4). Re-created per scene. */
  private scheduler: Scheduler | null = null;
  /** The upload budget authority the scheduler commits through (re-created per scene). Held so ABP-1 can
   *  retune its budget each frame from the frame-time EMA when `adaptiveBudget` is on. */
  private uploads: UploadScheduler | null = null;
  /** ABP-1: smoothed inter-frame time driving the adaptive mesh/upload budget. */
  private readonly frameEma = new FrameTimeEma();
  /** ABP-1 toggle (default OFF ⇒ the fixed budget literals + the exact `meshBudgetMs` branch). Flipping it
   *  takes effect on the next scene load (the scheduler reads the provider it was constructed with). */
  adaptiveBudget = false;
  /** ABP-2 toggle (default OFF ⇒ all edits stay uniformly async). When on, a near edit meshes inline for
   *  same-frame presentation. Bytes are identical either way — only the present latency changes. */
  immediatePresent = false;
  /** WRK-1 toggle (default OFF ⇒ the Scheduler gets no `admissionGate` ⇒ the EXACT current dispatch path,
   *  byte-identical, mirroring `adaptiveBudget`). When on, the Scheduler caps per-frame POOL dispatch by a
   *  queued-duration estimate (`workers * frameMs * bytesPerMs`) so an explosion-size burst doesn't
   *  `buildSnapshot` + structured-clone every ~23 KB snapshot in one frame — the rest defer into the aging
   *  TaskQueue (never dropped). Takes effect on the next scene load (the scheduler reads the gate it was
   *  constructed with). */
  workerBackPressure = false;
  /** WRK-1: snapshot bytes one worker meshes per ms — a CONSERVATIVE measured constant (a ~23 KB snapshot
   *  meshes in well under 1 ms on a worker), NOT a learned regression. At 60 Hz with N workers the per-frame
   *  budget is `N * frameMs * bytesPerMs`, i.e. ~5–6 snapshots per worker-ms — enough to keep workers fed
   *  while bounding a single-frame burst's clone spike to roughly the budget. Tunable on simstorm. */
  private static readonly WRK1_BYTES_PER_MS = 4096;
  /** PREC-1: AUTO-GATE threshold (blocks). The DEFAULT-OFF camera-relative origin path is enabled for a loaded
   *  world ONLY when any section bound's |coordinate| exceeds this — the magnitude where f32's ULP first grows
   *  past ~1/1024 block (a perceptible shimmer). Below it the flag stays off and the render is byte-identical
   *  to the pre-PREC-1 build (near-origin schematics pay nothing); above it the precision fix turns on
   *  automatically so large-coordinate worlds stop jittering. Exposed as a constant for tuning/tests. */
  static readonly CAMERA_RELATIVE_THRESHOLD = CAMERA_RELATIVE_THRESHOLD_BLOCKS;
  /** FS-5 combined toggle (default OFF). When on, opaque/cutout geometry is partitioned by facing at MESH
   *  time (so it changes baked bytes ⇒ takes effect on the next scene build, like a full remesh) AND the
   *  renderer culls camera-away facing slices at DRAW time. Both move together — slicing needs the runs.
   *  Default-off ⇒ the byte-identical baseline; flip only after the 0-pixel-diff proof (FS-5 harness). */
  faceSliceCull = false;
  /** WRK-3 toggle (default OFF). When on, the Model-A worker pool wires a cooperative-cancel SAB so a
   *  superseded build (a section re-dirtied faster than it meshes — redstone clock / piston / fluid front)
   *  STOPS meshing instead of fully meshing + cloning a result the generation stale-drop would discard.
   *  Inert when SAB is absent. Off ⇒ no SAB, no extra init field ⇒ byte-identical. Read at pool construction
   *  (the SAB must ship in the init message), so flip it BEFORE the scene build that creates the pool. */
  cooperativeCancelA = false;
  /** WRK-2 toggle (default OFF). When on, the Model-A worker pool routes builds through the work-stealing
   *  deque (an idle worker pulls the next build, so a skewed streaming burst doesn't head-of-line-block on
   *  one round-robin victim) AND arms the ported addModels model-epoch gate so a STOLEN build never meshes a
   *  newly-minted palette id as AIR. Off ⇒ the exact rr++ dispatch with NO epoch attached and the worker's
   *  build gate skipped ⇒ byte-identical (FNV quadHash unchanged). Read at pool construction, so flip it
   *  BEFORE the scene build that creates the pool (mirrors `cooperativeCancelA`). Model B already steals via
   *  the SAB semaphore + gates via MODEL_EPOCH, so this is passed ONLY to the Model-A `worker` branch. */
  workStealEnabled = false;
  /** Multithreading enable (default ON). When on, live-edit meshing runs OFF-THREAD on the worker pool (the
   *  Phase-5 large-world win); when off, `makeWorkerPool` returns null so meshing + translucent re-sorts run
   *  INLINE on the main thread (the always-supported fallback — identical output, just no parallelism). Read at
   *  scene build (the pool is created there), so flip via `setMultithreading` to rebuild the current scene. */
  multithreading = true;
  /** SL-4 toggle (default OFF). When on, opaque/cutout aligned faces bake per-vertex smooth lighting + AO
   *  (the snapshot widens to apron 2 for corner reads). Changes baked vertex bytes ⇒ a remesh-required
   *  visual feature — takes effect on the next scene build (NOT a free LUT reupload like `setLightLut`).
   *  AO lives in byte18 (remesh-required); the LUT light LEVEL stays a free reupload — two separate knobs. */
  smoothLighting = false;
  /** OCC-2 toggle (DEFAULT OFF). When on, sections mesh the per-perspective DIRECTION_SETS into
   *  `visibilityData` (a 4-element array; `[0]` stays the symmetric union ⇒ off-readers byte-identical, outside
   *  the FNV vertex hash ⇒ quadHash unchanged) AND the camera occlusion BFS joins those sets by camera quadrant
   *  into a strict SUBSET of the symmetric mask (fewer terrain sections submitted — conservative, never a hole).
   *  Changes baked `visibilityData` (NOT vertex bytes) ⇒ takes effect on the next scene build, like a remesh.
   *  Off ⇒ the 1-element symmetric word + the unchanged symmetric BFS (byte-identical render + visible set). */
  perPerspectiveVisibility = false;
  /** OCC-2 sub-flag (DEFAULT OFF; requires `perPerspectiveVisibility`). When on, the BFS also ANDs the
   *  outer-bound angle/slope cone into each propagation step. Outer-bound (widen-only) so it drops only
   *  directions the camera cannot see down, never a grazing-but-visible one — shipped separately so a cone bug
   *  cannot regress the DIRECTION_SETS win. Off ⇒ DIRECTION_SETS only. */
  occlusionAngleMask = false;
  /** Phase 5: off-thread mesh pool the scheduler dispatches to (null ⇒ inline meshing). Re-created per
   *  scene (a new block name triggers a full rebuild, so each pool's init carries the full palette). */
  private workerPool: WorkerPool | null = null;
  /** WRK-1: the Model-B shared ring (when running Model B), so the admission gate can read its in-flight
   *  byte estimate (`queuedBytes()`) as the single source of truth. Null on Model A / inline. */
  private sharedMeshQueue: SharedMeshQueue | null = null;
  /** Palette ids the current worker pool's frozen model map knows (seeded with the pool's init set at
   *  scene build). A runtime edit that mints a new id beyond this set must push the baked model via
   *  `addModels` (`syncWorkerModels`), or off-thread meshing renders that block as AIR — audit Bug 1. */
  private workerKnownIds = new Set<number>();
  /** Monotonic frame counter driving the scheduler/upload aging + budgets. */
  private schedulerFrame = 0;
  private fluids: FluidContext | null = null;
  private entityWorld: EntityWorld | null = null;
  private overlay: OverlayRenderer | null = null;
  private explosions: ExplosionEffects | null = null;
  /** Moving-piston animation (transient terrain + local-sorted glass). Created at device init. */
  private pistons: PistonTransients | null = null;
  private draws: SectionDraw[] = [];
  /** F3: incremental draw-list cache. Re-resolves only committed/disposed sections per tick (O(changed)),
   *  full-rebuilds on arena relocation. Re-created per scene build (it caches the current store's sections). */
  private drawList = new DrawList();
  private pack: import("../resources/ResourcePack").ResourcePack | null = null;
  private readonly dirtyTracking = new DirtyTracking();
  private readonly renderDistance = Infinity; // editor worlds are bounded; cull by frustum only
  /** Latest entities by id (for picking). */
  private readonly entitiesById = new Map<string, EditableEntity>();
  /** Block-entity-bearing blocks by "x,y,z" (sign text / NBT) so picking returns the full block. */
  private readonly beBlocks = new Map<string, EditableBlock>();
  /** Block names the current atlas covers; a new name in an edit marks the palette stale. */
  private atlasedNames = new Set<string>();
  private paletteStale = false;

  /** Resolves once the current schematic's scene is built + committed. */
  private sceneReady: Promise<void> | null = null;
  /** Monotonic build id — a newer buildScene supersedes an older one still resolving (boot/reload race). */
  private buildToken = 0;
  /** Serializes world mutations (and defers them until the scene is ready). */
  private opChain: Promise<void> = Promise.resolve();

  // ── Tick interpolation ───────────────────────────────────────────────────────
  private tickInterval = 100; // ms between sim ticks (updated from diff timing)
  private tickStartMs = 0;
  private tickAnimate = false;

  /** MEM-2: device-level freed-GPUBuffer pool (default OFF). Captured at construction because the device is
   *  created in the ctor (before the caller can set a field) — pass `{ bufferPool: true }` to enable it. */
  private readonly bufferPool: boolean;
  /** Force the WebGL2 fallback backend even when WebGPU is available (debug toggle). Captured at
   *  construction like `bufferPool` because the backend is chosen when the device is created. */
  private readonly preferWebGL2: boolean;

  constructor(canvas: HTMLCanvasElement, opts: { bufferPool?: boolean; preferWebGL2?: boolean } = {}) {
    this.canvas = canvas;
    this.bufferPool = opts.bufferPool ?? false;
    this.preferWebGL2 = opts.preferWebGL2 ?? false;
    this.camera.fovY = FOV_Y;
    this.attachInput();
    this.ready = this.initDevice();
  }

  /** The backend actually in use, once the device is created (`await ready` first). Null before init or
   *  if no GPU backend was available. Lets a debug UI confirm which backend is live (e.g. forced WebGL2). */
  get activeBackend(): BackendKind | null {
    return this.device?.backend ?? null;
  }

  private async initDevice(): Promise<void> {
    let device: CanvasDevice;
    try {
      device = await createDevice(this.canvas, { bufferPool: this.bufferPool, preferWebGL2: this.preferWebGL2 });
    } catch (e) {
      // Both WebGPU and WebGL2 failed to create — the render loop will never start. Surface a
      // user-visible message (W7) instead of an unhandled rejection + a silent black canvas.
      console.error("[renderer] no GPU backend available (WebGPU and WebGL2 both failed)", e);
      this.onDeviceError?.(
        "3D rendering is unavailable — your browser or GPU does not support WebGPU or WebGL2.",
      );
      return; // leaves this.device null, loop unstarted; awaiters of `ready` resolve (no throw)
    }
    if (this.disposed) {
      device.destroy?.();
      return;
    }
    this.device = device;
    this.onBackendReady?.(device.backend); // debug: report the live backend (WebGPU vs forced/fallback WebGL2)
    // The piston driver bakes against whatever scene is live; guard the bake so a bad/missing blockstate
    // (e.g. an unatlased moved block) yields "nothing drawn" rather than breaking the render loop.
    this.pistons = new PistonTransients(device, (name, props) => {
      try {
        return this.scene?.baker.bake({ name, props }) ?? null;
      } catch {
        return null;
      }
    });
    this.syncCanvasSize();
    window.addEventListener("resize", this.onResize);
    document.addEventListener("visibilitychange", this.onVisibilityChange);
    this.rafId = requestAnimationFrame(this.loop);
  }

  /** dynamicFps + pauseWhenHidden: halt the loop when the tab goes hidden, redraw + resume when it returns.
   *  Inert otherwise (the default loop relies on the browser pausing rAF for hidden tabs). */
  private readonly onVisibilityChange = (): void => {
    if (this.disposed) return;
    if (isPageHidden()) {
      if (this.dynamicFps && this.pauseWhenHidden) this.stopLoop();
    } else {
      this.invalidate(); // wake + redraw whatever changed while we were away
    }
  };

  // ── Resource pack ────────────────────────────────────────────────────────────

  /** Load the resource pack from already-fetched zip bytes; rebuild the scene if a world is loaded. */
  async loadResourcePack(buffer: ArrayBuffer): Promise<number> {
    await this.ready;
    const { ResourcePack } = await import("../resources/ResourcePack");
    this.pack = await ResourcePack.fromBytes(buffer);
    if (this.model) {
      this.sceneReady = this.buildScene(true);
      await this.sceneReady;
    }
    // Report the pack's texture count (a real "pack loaded" figure), not the current world's palette size —
    // an empty starter world has a 0-entry palette, which previously logged a misleading "Loaded 0 textures".
    return this.pack.textureCount();
  }

  // ── Schematic load ──────────────────────────────────────────────────────────

  /** Replace the world with a parsed schematic. Synchronous model build; async scene build (buffered). */
  loadSchematic(schematic: EditableSchematic, frame = true): void {
    const loaded = loadWorld(schematic);
    this.model = loaded.model;
    // Seed the live tracking maps (also the source re-ingested on any scene rebuild).
    this.entitiesById.clear();
    for (const e of loaded.entities) this.entitiesById.set(e.id, e);
    this.beBlocks.clear();
    for (const be of loaded.blockEntities) this.beBlocks.set(`${be.x},${be.y},${be.z}`, be.block);
    this.sceneReady = this.buildScene(frame);
  }

  /** (Re)build the GPU scene + entity world for the current model + pack.
   *
   * BUILD-THEN-SWAP: the entire new scene is assembled into locals, and only then is the old scene torn
   * down and the new one published — in ONE synchronous step. The render loop therefore keeps drawing the
   * current scene until the swap, so a rebuild (e.g. the re-atlas when the first new block type appears)
   * never shows an empty frame (the "screen flickers on first placement" bug). A monotonic build token
   * discards a stale build whose results arrive after a newer rebuild started (the load→loadResourcePack race). */
  private async buildScene(frame: boolean): Promise<void> {
    await this.ready;
    const device = this.device;
    const model = this.model;
    if (!device || !model || this.disposed) return;
    const token = ++this.buildToken;

    if (!this.pack) {
      const { ResourcePack } = await import("../resources/ResourcePack");
      this.pack = await ResourcePack.load("pack.zip");
    }

    const scene = await buildScene(device, model.palette, undefined, FLUID_SPRITES, this.renderDistance, this.pack);
    if (this.disposed || token !== this.buildToken) { scene.renderer.dispose(); return; }
    const fluids = makeFluidContext(model.palette, scene.atlas);
    const snapshots = new SnapshotSource(model.source, this.smoothLighting ? 2 : undefined); // SL-3/SL-4 apron
    // Commit exactly the occupied chunks (any coordinate, incl. negative) — no [0,dims) scan.
    const store = commitWorld(device, model.source, scene.provider, [0, 0, 0], fluids, model.occupiedSections(), this.meshOptions());
    scene.renderer.faceSliceCull = this.faceSliceCull; // FS-5: draw-time slice cull moves with the mesh partition
    // F3: a fresh incremental cache for this store; the first flatten is a full build (drains the initial
    // commitWorld changes), every later tick re-resolves only the sections that committed.
    this.drawList = new DrawList();
    const draws = this.drawList.flatten(store);
    // Phase-1 dispatcher: per-tick edits enqueue into `scheduler`, which meshes within a CPU budget and
    // commits through `uploads` within its byte/time budget (N1/SCHED-3/U4). Initial load stays synchronous
    // (commitWorld above) so the first frame is complete; only live edits flow through the scheduler.
    // SCHED-4: commit near-the-camera sections first under the upload budget (wires the priorityOf hook).
    const uploads = new UploadScheduler(store, { priorityOf: (s) => this.uploadPriority(s) });
    this.uploads = uploads;
    // Phase 5: mesh live edits OFF-THREAD. The pool is fed the TRANSFERRED baked palette models + fluids
    // (byte-identical to inline — proven in harness/workers); the scheduler dispatches to it + drains
    // async with generation stale-drop. Falls back to inline meshing if workers can't spawn.
    const workerPool = this.makeWorkerPool(scene.provider, fluids, model.palette);
    // WRK-1: capture the Model-B ring (if any) so the gate reads its in-flight byte estimate as the single
    // source of truth (no double-count with the Scheduler's Model-A accumulator). Null on Model A / inline.
    const sharedQueue = this.sharedMeshQueue;
    const scheduler = new Scheduler({
      store, snapshots, provider: scene.provider, fluids, uploads, pool: workerPool ?? undefined,
      // ABP-1: provide an adaptive mesh budget ONLY when the toggle is on — off ⇒ undefined ⇒ the Scheduler
      // keeps its exact fixed `meshBudgetMs` branch (the flag-off golden path).
      meshBudgetOf: this.adaptiveBudget ? () => this.adaptiveMeshBudgetMs() : undefined,
      // WRK-1: the queued-duration admission gate ONLY when the toggle is on — off ⇒ undefined ⇒ the
      // Scheduler runs the EXACT current dispatch path (byte-identical). `frameMs` reads the clamped frame-EMA
      // (≥1ms, FrameTimeEma); `workers` from the pool (inline ⇒ 1); `bytesPerMs` a measured constant (NOT a
      // learned regression). On Model B `queuedBytesOf` makes the SAB ring the single source of truth.
      admissionGate:
        this.workerBackPressure && workerPool
          ? {
              frameMsOf: () => this.frameEma.average,
              workers: workerPool.workerCount,
              bytesPerMs: SchematicViewer.WRK1_BYTES_PER_MS,
              queuedBytesOf: sharedQueue ? () => sharedQueue.queuedBytes() : undefined,
            }
          : undefined,
      meshOptions: this.meshOptions(), // FS-5: inline edits partition exactly like the initial commit + worker
      // Render-on-demand: an off-thread mesh/sort reply stages a commit with no other wake path, so nudge the
      // loop to run the next drainAndCommit + draw. No-op visually in the continuous (default) loop.
      onWork: () => this.invalidate(),
    });
    const entityWorld = await buildEntityWorld(device, scene, this.renderDistance, this.itemIds);
    if (this.disposed || token !== this.buildToken) {
      workerPool?.dispose();
      entityWorld.dispose();
      for (const s of [...store.all()]) store.dispose(s.key);
      scene.renderer.dispose();
      return;
    }
    const overlay = new OverlayRenderer(device, device.colorFormat, device.depthFormat);
    overlay.setGrid(model.bounds()); // ground-reference floor grid centered on the build (null → origin default)
    const explosions = new ExplosionEffects(device, device.colorFormat, device.depthFormat);

    // Re-ingest the CURRENT entity + block-entity state into the new world (preserves live sim state).
    entityWorld.ingestEntities([...this.entitiesById.values()].map((e) => this.toSnapshot(e)), []);
    for (const b of this.beBlocks.values()) {
      // A loaded sign carries `signText` (a flat field) — flatten its FRONT lines into `props.text` so the
      // board shows text immediately (without waiting for a sim diff). Diff-updated BEs already have it in
      // `properties`. (Shelf/campfire items aren't a flat field — they arrive via the diff.)
      const msgs = b.signText?.front?.messages;
      const props = msgs ? { ...b.properties, text: msgs.join("|") } : b.properties;
      entityWorld.setBlockEntity({ x: b.x, y: b.y, z: b.z, type: b.name, props, animating: false });
    }

    // Atomic swap (synchronous — the render loop can't interleave): drop the old scene, publish the new.
    this.disposeScene();
    this.scene = scene;
    this.overlay = overlay;
    this.explosions = explosions;
    this.fluids = fluids;
    this.snapshots = snapshots;
    this.store = store;
    this.scheduler = scheduler;
    this.workerPool = workerPool;
    // TR-1: with a worker pool, DYNAMIC translucent re-sorts run off-thread (dispatchTranslucentSorts);
    // the renderer skips its inline re-sort. Without a pool it re-sorts inline (the cheap main-thread path).
    scene.renderer.offThreadSort = !!workerPool;
    // TR-1: propagate the huge-section re-sort gate to the (possibly fresh) renderer so the inline path and
    // this viewer's off-thread dispatch read the SAME flag (single source of truth — default-off).
    scene.renderer.hugeSortGate = this._hugeSortGate;
    // P6/LM-1: a fresh renderer starts on the default daylight LUT — re-apply any user brightness/day-night.
    if (Object.keys(this.lightLutParams).length > 0) scene.renderer.setLightLut(buildLightLut(this.lightLutParams));
    // The pool was init'd (makeWorkerPool) with exactly these ids — seed the known set so later edits push
    // only the genuinely-new ids via addModels (audit Bug 1). Mirrors makeWorkerPool's `ids`.
    this.workerKnownIds = new Set<number>([AIR, ...Object.keys(model.palette).map(Number)]);
    this.draws = draws;
    this.entityWorld = entityWorld;
    // Beacons were ingested above with their loaded props; now resolve each one's beam from the world (base
    // pyramid + clear column) so an un-activated/obstructed beacon shows no beam (vanilla getBeamSections).
    this.recomputeBeacons(() => true);
    // PREC-1 AUTO-GATE: enable the DEFAULT-OFF camera-relative origin path (terrain AND entities in lockstep —
    // TRAP PREC-1.D) ONLY when this world's block bounds reach far from the origin, where f32 vertex precision
    // starts to jitter. Near-origin schematics keep the flag off → byte-identical to the pre-PREC-1 build.
    const bb = model.bounds();
    const farWorld = !!bb && (farFromOrigin(bb.min) || farFromOrigin(bb.max));
    scene.renderer.cameraRelative = farWorld;
    entityWorld.cameraRelative = farWorld;
    // The atlas now covers the current palette's names PLUS the always-atlased set (cart contents +
    // piston-transient blocks like piston_head/moving_piston). A later edit introducing a name OUTSIDE
    // this set marks the palette stale → a rebuild re-atlases. Seeding the always-atlased names here is
    // what stops a piston's first fire from forcing a full rebuild (freeze) for piston_head/moving_piston.
    this.atlasedNames = new Set([...Object.values(model.palette).map((e) => e.name), ...ALWAYS_ATLASED_BLOCK_NAMES]);
    this.paletteStale = false;
    // The atlas (UVs) changed; drop the piston driver's cached baked geometry so it re-bakes against the
    // new atlas on next frame. Active piston entries are kept — only the GPU geometry cache is dropped.
    this.pistons?.invalidate();

    if (frame) {
      const b = model.bounds();
      // Empty world → frame a default chunk around the origin (where the user will start placing).
      if (b) this.camera.frameBounds(b.min, b.max);
      else this.camera.frameBounds([0, 0, 0], [16, 16, 16]);
    }
    this.invalidate(); // render-on-demand: the new scene was just published — draw it.
  }

  // ── Mutation: the one path all block edits funnel through ─────────────────────

  /** Apply a batch of block edits (set or remove), remesh affected sections, refresh the draw list. */
  private applyBlockEdits(edits: { x: number; y: number; z: number; name: string | null; properties?: Record<string, string>; items?: string[]; text?: string; open?: boolean }[]): void {
    const model = this.model;
    const scheduler = this.scheduler;
    if (!model || !scheduler) return;
    // Render-on-demand: any edit (terrain remesh OR a block-entity-only items/text change) should become
    // visible. Mark dirty up front so even a BE-only update — which returns before the dirty-section path —
    // wakes the loop. Mesh commits land over the next frames (kept live by scheduler.hasPendingWork).
    this.invalidate();

    const dirty = new Set<SectionKey>();
    for (const e of edits) {
      // No box to grow — chunk storage handles any coordinate (incl. negative) directly.
      const res = e.name === null || e.name === "minecraft:air" || e.name === "air"
        ? model.removeBlock(e.x, e.y, e.z)
        : model.setBlock(e.x, e.y, e.z, e.name, e.properties ?? {});
      // Route block-entity-rendered types (chest/shulker/bell/sign/shelf/campfire/…) to the BE renderer;
      // clear one when the cell becomes a non-BE block or air. Terrain meshing skips these (scene.ts
      // provider), so the BE renderer owns them. Done BEFORE the no-change skip below: a shelf/campfire's
      // display ITEMS or a sign's TEXT change without a blockstate change (they're block-entity data), so
      // the BE must update even when the terrain block id is identical. `items`/`text` are merged into the
      // BE props (NOT the terrain palette key, which would fragment it) as the props the BE renderer reads.
      if (this.entityWorld) {
        const newIsBE = e.name !== null && e.name !== "minecraft:air" && e.name !== "air" && hasBlockEntityModel(e.name);
        const key = `${e.x},${e.y},${e.z}`;
        if (newIsBE) {
          const props: Record<string, string> = { ...(e.properties ?? {}) };
          if (e.items) props.items = e.items.join(",");
          if (e.text !== undefined) props.text = e.text;
          this.entityWorld.setBlockEntity({ x: e.x, y: e.y, z: e.z, type: e.name!, props, animating: false });
          // Container lid OPEN (chest/shulker) from the sim's open-viewer count — drives the lid animation.
          // Set AFTER setBlockEntity (which replaces the record); a no-op when `open` is absent/unchanged.
          if (e.open !== undefined) this.entityWorld.setBlockEntityOpen(e.x, e.y, e.z, e.open);
          this.beBlocks.set(key, { x: e.x, y: e.y, z: e.z, name: e.name!, properties: props });
        } else {
          const oldName = res.oldId !== AIR ? model.palette[res.oldId]?.name : undefined;
          if (oldName !== undefined && hasBlockEntityModel(oldName)) {
            this.entityWorld.removeBlockEntity(e.x, e.y, e.z);
            this.beBlocks.delete(key);
          }
        }
      }
      if (!res.changed) continue; // terrain unchanged (e.g. items-only shelf update) → no remesh
      this.snapshots?.setBlock(e.x, e.y, e.z, model.getBlock(e.x, e.y, e.z)); // N3: patch the source cache
      const edit: BlockEdit = { x: e.x, y: e.y, z: e.z, reason: DirtyReason.Simulation };
      for (const key of this.dirtyTracking.sectionsAffectedBy(edit)) dirty.add(key);
    }
    // A beacon's beam depends on the world (its base pyramid + the column above it), not just its own cell —
    // recompute the beam visibility/color/height for any beacon an edit could have affected (a block placed in
    // its column or base). Cheap: beacons are rare and this only re-sets the (special-only) BE props.
    this.recomputeBeacons((b) => edits.some((e) => editAffectsBeacon(e, b)));
    if (dirty.size === 0) return;

    instrument.bumpJob("dirtied", dirty.size); // Phase-P jobs/tick: sections this edit re-dirtied
    // Enqueue (coalesced — N1). The render loop's `scheduler.tick` meshes within the CPU budget (SCHED-3),
    // commits through the UploadScheduler (U4), and refreshes `this.draws`. Edits get a high priority so a
    // handful mesh the same frame (no perceptible lag); a big burst spreads instead of stalling the thread.
    // ABP-2: a near in-frustum edit (distance-only first) meshes inline for same-frame presentation; the
    // far/bulk remainder stays on the budgeted async path. Off by default ⇒ every edit is uniformly async.
    const wantImmediate = this.immediatePresent;
    for (const key of dirty) scheduler.markDirty(key, DirtyReason.Simulation, EDIT_PRIORITY, wantImmediate && this.isNearImmediate(key));
    // Did an edit introduce a block name the current atlas doesn't cover? (drives ensureRenderablePalette)
    if (!this.paletteStale) {
      for (const id in model.palette) {
        if (!this.atlasedNames.has(model.palette[id].name)) {
          this.paletteStale = true;
          break;
        }
      }
    }
    // Ship any new property-variant palette ids to the mesh workers (audit Bug 1). MUST run after the
    // paletteStale check above: if a rebuild is coming (unatlased name), syncWorkerModels skips (the new
    // pool re-inits the full palette); otherwise it pushes the new atlased ids so they don't mesh as AIR.
    this.syncWorkerModels();
    // `this.draws` is refreshed in the render loop after `scheduler.tick` commits the enqueued sections.
  }

  /** Re-resolve the beam (activation + clear column → visibility/color/height) for every tracked beacon that
   *  `affected` selects, updating its (special-only) BE props. The renderer then draws the beam only when the
   *  beacon is actually activated + unobstructed (vanilla). No-op without an entity world. */
  private recomputeBeacons(affected: (b: { x: number; y: number; z: number }) => boolean): void {
    const ew = this.entityWorld;
    if (!ew) return;
    for (const b of this.beBlocks.values()) {
      if (!isBeacon(b.name) || !affected(b)) continue;
      const props = { ...b.properties, ...this.beaconBeamProps(b.x, b.y, b.z) };
      ew.setBlockEntity({ x: b.x, y: b.y, z: b.z, type: b.name, props, animating: false });
      this.beBlocks.set(`${b.x},${b.y},${b.z}`, { ...b, properties: props });
    }
  }

  /** Compute a beacon's beam props (`active`/`beamColor`/`height`) from the current world model. */
  private beaconBeamProps(x: number, y: number, z: number): Record<string, string> {
    const model = this.model;
    if (!model) return { active: "false" };
    const bounds = model.bounds();
    const scanTop = bounds ? bounds.max[1] : y;
    const beam = computeBeaconBeam(
      (sx, sy, sz) => {
        const id = model.getBlock(sx, sy, sz);
        return id === AIR ? null : (model.palette[id]?.name ?? null);
      },
      x, y, z, scanTop,
    );
    return { active: String(beam.active), beamColor: beam.color.toString(16).padStart(6, "0"), height: String(beam.height) };
  }

  /**
   * P6/LM-1 — global brightness / day-night / night-vision. Rewrites the light LUT: ONE `writeTexture`
   * with ZERO remesh jobs, the only genuinely free light win (TRAP 6.A — per-block light is NOT free).
   * `brightness` 1 = full daylight, 0 = black. Takes effect immediately (even mid static-camera F1 replay,
   * which samples the LUT by reference) and persists across scene rebuilds. Local light changes go through
   * the light-only invalidation path instead (a sim-supplied LightSource + DirtyReason.Light).
   */
  setBrightness(brightness: number): void {
    this.setLightLut({ brightness });
  }

  /** Rewrite the light LUT from explicit params (brightness / ambient floor / sky scale). See setBrightness. */
  setLightLut(params: LightLutParams): void {
    this.lightLutParams = { ...this.lightLutParams, ...params };
    this.scene?.renderer.setLightLut(buildLightLut(this.lightLutParams));
    this.invalidate(); // render-on-demand: the LUT changed the look without any remesh — redraw once.
  }

  /** Set a single block (editor convenience; sim edits flow through applySimulationDiff). */
  setBlock(x: number, y: number, z: number, name: string, properties: Record<string, string> = {}): void {
    this.opChain = this.opChain.then(async () => {
      await this.sceneReady;
      if (this.disposed) return;
      this.applyBlockEdits([{ x, y, z, name, properties }]);
      await this.ensureRenderablePalette();
    });
  }

  /**
   * Toggle a block entity's deliberate-preview animation (chest lid opening, bell swing, shulker box
   * opening, decorated-pot wobble) at a world position — the editor/UI seam for "animate this BE".
   * Idle-loop BEs (banners, conduits) wave/spin continuously regardless of this flag; this is only for the
   * event-driven types. Flipping it re-meshes exactly that BE's section (it leaves the static bake and
   * draws per-frame, or vice-versa) and never double-draws. No-op if no BE is at that position.
   */
  setBlockEntityAnimating(x: number, y: number, z: number, animating: boolean): void {
    this.entityWorld?.setBlockEntityAnimating(x, y, z, animating);
    this.invalidate(); // ensure a frame draws (dynamicFps render-on-demand)
  }

  // ── Simulation diff ──────────────────────────────────────────────────────────

  /** Apply a batched per-tick simulation diff: blocks → entities → block entities → pistons. */
  applySimulationDiff(diff: RendererSimulationDiff, opts: ApplyDiffOptions = {}): Promise<void> {
    this.opChain = this.opChain.then(async () => {
      await this.sceneReady;
      if (this.disposed || !this.model) return;

      if (opts.timing) {
        this.tickAnimate = opts.timing.animate;
        this.tickInterval = opts.timing.interval;
        this.tickStartMs = opts.timing.startedAtMs;
      }

      // Blocks: removals then placements, one remesh pass over the union of dirtied sections.
      const edits: { x: number; y: number; z: number; name: string | null; properties?: Record<string, string>; items?: string[]; text?: string; open?: boolean }[] = [];
      for (const r of diff.removedBlocks ?? []) edits.push({ x: r.x, y: r.y, z: r.z, name: null });
      for (const b of diff.setBlocks ?? []) {
        // Flatten a sign's FRONT lines into the `text` prop the BE renderer bakes (joined with "|", which
        // it splits back into ≤4 lines); back side + colour/glow aren't rendered yet. `open` rides through to
        // the BE renderer's lid animation (chest/shulker); barrels carry OPEN in `properties` instead.
        const msgs = b.signText?.front?.messages;
        edits.push({ x: b.x, y: b.y, z: b.z, name: b.name, properties: b.properties, items: b.items, text: msgs ? msgs.join("|") : undefined, open: b.open });
      }
      if (edits.length) this.applyBlockEdits(edits);
      await this.ensureRenderablePalette(); // re-atlas if a new block type appeared (rare)

      // Entities: ingest the sparse changed set + removals (interpolated by EntityScene; tracked for picking).
      const changed: EditableEntity[] = (diff.entities ?? []).map((e) => ({
        id: e.id,
        type: e.type,
        position: [e.position[0], e.position[1], e.position[2]] as Vec3Tuple,
        velocity: e.velocity ? ([e.velocity[0], e.velocity[1], e.velocity[2]] as Vec3Tuple) : undefined,
        properties: e.properties,
      }));
      this.trackAndIngest(changed, diff.removedEntityIds ?? []);

      // Moving pistons: replace the active set (one entry per moving_piston BE). The slide is animated in
      // the render loop by interpolating progressO→progress with the tick `partial`. The completion handoff
      // is implicit: when a push finishes the sim drops the piston AND writes the resting block in THIS same
      // diff, so set([]) clears the transient the same tick applyBlockEdits re-meshes the resting block —
      // no blank, no double-draw (PISTON_PLAN §5 D7). The static moving_piston block meshes nothing (D4).
      const pistonsWereActive = !!this.pistons?.active();
      this.pistons?.set(diff.movingPistons ?? []);
      // Render-on-demand: invalidate ONLY when this diff actually changed something the loop must redraw —
      // entities, or the moving-piston set (new/ongoing OR a clear that removes a transient). Block edits
      // already invalidated via applyBlockEdits. Crucially we do NOT invalidate on an EMPTY diff: the worker
      // broadcasts a diff EVERY tick even when quiescent (sim.worker collect()), so an unconditional wake
      // would re-render an unchanged frame ~20×/s and defeat dynamic-FPS idling for a settled-but-running sim.
      const entityChange = (diff.entities?.length ?? 0) > 0 || (diff.removedEntityIds?.length ?? 0) > 0;
      const pistonChange = (diff.movingPistons?.length ?? 0) > 0 || pistonsWereActive;
      if (entityChange || pistonChange) this.invalidate();
    });
    return this.opChain;
  }

  /** Spawn a one-shot explosion effect at world coords (driven by the app from `diff.events`). */
  spawnExplosion(x: number, y: number, z: number, radius: number): void {
    if (!this.animationsEnabled) return; // animations off ⇒ no visual burst (the block damage still applies via the diff)
    this.explosions?.spawn(x, y, z, radius);
    this.invalidate(); // wake; explosions.hasActive() then keeps the loop live until the burst finishes.
  }

  // ── Entities / block entities ────────────────────────────────────────────────

  private toSnapshot(e: EditableEntity): EntitySnapshot {
    return {
      id: e.id,
      type: e.type,
      position: [e.position[0], e.position[1], e.position[2]] as Vec3,
      velocity: e.velocity ? ([e.velocity[0], e.velocity[1], e.velocity[2]] as Vec3) : undefined,
      properties: e.properties,
    };
  }

  /** Update the pick index + EntityWorld from a changed/removed entity set. */
  private trackAndIngest(changed: readonly EditableEntity[], removed: readonly string[]): void {
    for (const id of removed) this.entitiesById.delete(id);
    for (const e of changed) this.entitiesById.set(e.id, e);
    this.entityWorld?.ingestEntities(changed.map((e) => this.toSnapshot(e)), removed);
  }

  /** If an edit introduced a block name the atlas doesn't cover, make it renderable. ATL-1: first TRY an
   * INCREMENTAL atlas grow (append the new sprites into the reserved headroom — existing UVs invariant, so
   * existing sections keep their geometry and only the edit's already-dirtied sections re-mesh). Only if the
   * grow can't be done (unknown block / animated sprite / headroom full) fall back to the full re-atlas +
   * world re-mesh (the original behaviour — correct, but a one-time hitch). Preserves live entity/BE state. */
  private async ensureRenderablePalette(): Promise<void> {
    if (!this.paletteStale) return;
    const model = this.model;
    const scene = this.scene;
    if (model && scene) {
      const newStates: { id: number; name: string; props: Record<string, string> }[] = [];
      for (const idStr in model.palette) {
        const id = Number(idStr);
        const e = model.palette[id];
        if (!this.atlasedNames.has(e.name)) newStates.push({ id, name: e.name, props: e.props });
      }
      if (newStates.length > 0 && (await scene.growPalette(newStates))) {
        // Incremental success — existing geometry untouched; the edit's dirtied sections mesh next tick with
        // the new sprites. Mark the names atlased + push the new ids to the off-thread workers.
        for (const st of newStates) this.atlasedNames.add(st.name);
        this.paletteStale = false;
        this.syncWorkerModels();
        return;
      }
    }
    // Fallback: full re-atlas + world re-mesh.
    this.sceneReady = this.buildScene(false);
    await this.sceneReady;
  }

  // ── Picking / placement (Steps 5/6) ──────────────────────────────────────────

  /** Build a world-space pick ray from a screen-space pointer (pinhole camera; no matrix inverse). */
  private rayFromScreen(clientX: number, clientY: number): Ray {
    const rect = this.canvas.getBoundingClientRect();
    const ndcX = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    const ndcY = 1 - ((clientY - rect.top) / Math.max(1, rect.height)) * 2;
    const { right, up, forward } = this.camera.viewBasis();
    const tanHalf = Math.tan(this.camera.fovY / 2);
    const aspect = this.camera.aspect;
    const dx = forward[0] + right[0] * ndcX * aspect * tanHalf + up[0] * ndcY * tanHalf;
    const dy = forward[1] + right[1] * ndcX * aspect * tanHalf + up[1] * ndcY * tanHalf;
    const dz = forward[2] + right[2] * ndcX * aspect * tanHalf + up[2] * ndcY * tanHalf;
    const len = Math.hypot(dx, dy, dz) || 1;
    return { origin: this.camera.position, dir: [dx / len, dy / len, dz / len] };
  }

  /** The full EditableBlock at a cell (name/props from the palette + any block-entity data). */
  private blockAt(x: number, y: number, z: number): EditableBlock | null {
    const model = this.model;
    if (!model) return null;
    const id = model.getBlock(x, y, z);
    if (id === AIR) return null;
    const entry = model.palette[id];
    if (!entry) return null;
    const be = this.beBlocks.get(`${x},${y},${z}`);
    return { x, y, z, name: entry.name, properties: { ...entry.props }, signText: be?.signText, blockEntity: be?.blockEntity };
  }

  pick(clientX: number, clientY: number): PickResult | null {
    if (!this.model) return null;
    return pickWorld({
      ray: this.rayFromScreen(clientX, clientY),
      getBlock: (x, y, z) => this.model!.getBlock(x, y, z),
      blockAt: (x, y, z) => this.blockAt(x, y, z),
      outlineProvider: this.outlineProvider,
      entities: [...this.entitiesById.values()],
      boundsOf: entityBoundsSize,
    });
  }

  getPlacementHint(pick: PickResult | null): PlacementHint | null {
    return computePlacementHint(pick, this.cameraHorizontalFacing());
  }

  regionFacingMap(pick: PickResult | null): Partial<Record<FaceRegion, string>> {
    return computeRegionFacingMap(pick);
  }

  /** Cardinal direction from the camera toward its target (for ground/entity placement). */
  private cameraHorizontalFacing(): "north" | "south" | "east" | "west" {
    const p = this.camera.position;
    const t = this.camera.target;
    const dx = t[0] - p[0];
    const dz = t[2] - p[2];
    if (Math.abs(dx) > Math.abs(dz)) return dx > 0 ? "east" : "west";
    return dz > 0 ? "south" : "north";
  }

  /** Deterministic throw spawn: 2 blocks ahead of the camera, speed 0.4 along its forward. */
  getThrowVector(): { position: Vec3Tuple; velocity: Vec3Tuple } {
    const { forward } = this.camera.viewBasis();
    const p = this.camera.position;
    return {
      position: [p[0] + forward[0] * 2, p[1] + forward[1] * 2, p[2] + forward[2] * 2],
      velocity: [forward[0] * 0.4, forward[1] * 0.4, forward[2] * 0.4],
    };
  }

  // ── Overlays (interim — Step 7) ──────────────────────────────────────────────

  // The overlay setters all invalidate(): in dynamicFps mode the cursor/region highlight follows the pointer,
  // so a hover/move that changes the overlay must wake the loop even when the world is otherwise static.
  showFaceRegions(pick: PickResult | null, regionStates?: Partial<Record<FaceRegion, string | null>>): void {
    this.overlay?.setFaceRegions(pick, regionStates);
    this.invalidate();
  }
  showRailRegions(pick: PickResult | null, shape?: string, canCurve?: boolean): void {
    this.overlay?.setRail(pick, shape ?? "", canCurve ?? false);
    this.invalidate();
  }
  clearFaceRegions(): void {
    this.overlay?.clearRegions();
    this.invalidate();
  }
  showCursorForPick(pick: PickResult | null, boxes?: ReadonlyArray<OutlineBox> | null): void {
    this.overlay?.setCursor(pick, boxes ?? null);
    this.invalidate();
  }
  clearCursorOutline(): void {
    this.overlay?.clearCursor();
    this.invalidate();
  }

  // ── Misc API ─────────────────────────────────────────────────────────────────

  /** Camera right/up/forward basis (for the on-screen compass). */
  getViewBasis(): { right: Vec3Tuple; up: Vec3Tuple; forward: Vec3Tuple } {
    const b = this.camera.viewBasis();
    return { right: [...b.right] as Vec3Tuple, up: [...b.up] as Vec3Tuple, forward: [...b.forward] as Vec3Tuple };
  }

  /** Resolves once the device is up and the current schematic's scene is committed (test/harness hook). */
  async whenReady(): Promise<void> {
    await this.ready;
    if (this.sceneReady) await this.sceneReady;
  }

  /** Read back the current swapchain pixels (headless verification / screenshot export). */
  async readback(): Promise<{ width: number; height: number; bytesPerRow: number; data: Uint8Array }> {
    await this.ready;
    if (!this.device) throw new Error("SchematicViewer: device not initialized");
    return this.device.readCanvasPixels();
  }

  getStats(): { renderer: RenderStats } {
    return { renderer: this.lastStats };
  }

  /** Sample the full Phase-P RenderStats for this frame: gauges + per-frame deltas off the cumulative
   *  device/instrument counters. Called once per rendered frame after `lastFrameMs` is set. */
  private sampleStats(): RenderStats {
    const dev = this.device;
    const r = this.scene?.renderer;
    const e = this.entityWorld?.stats;
    const a = instrument.alloc;
    const j = instrument.job;
    const p = this.prevCounters;
    const s: RenderStats = {
      frameMs: this.lastFrameMs,
      cullMs: r?.timing.cullMs ?? 0,
      encodeMs: r?.timing.encodeMs ?? 0,
      resortMs: r?.timing.resortMs ?? 0,
      entityMs: this.lastEntityMs,
      drawCalls: dev ? dev.drawCalls - p.drawCalls : 0,
      passes: dev ? dev.passCount - p.passes : 0,
      liveBuffers: dev?.liveBuffers ?? 0,
      createBufferCalls: dev ? dev.createBufferCalls - p.createBufferCalls : 0,
      poolHits: dev ? dev.poolHits - p.poolHits : 0,
      sectionsTotal: r?.stats.total ?? 0,
      sectionsDrawn: r?.stats.drawn ?? 0,
      sectionsCulled: r?.stats.culled ?? 0,
      resorts: (r?.resortCount ?? 0) + this.offThreadResorts, // inline (no pool) + off-thread (TR-1) re-sorts
      entities: e?.entities ?? 0,
      entityDraws: e?.drawn ?? 0,
      beStatic: e?.beStatic ?? 0,
      beAnimating: e?.beAnimating ?? 0,
      allocRawVertex: a.rawVertex - p.aRaw,
      allocDrawList: a.drawList - p.aDraw,
      allocFrustum: a.frustum - p.aFrustum,
      allocEntityEmit: a.entityEmit - p.aEntity,
      jobsDirtied: j.dirtied - p.jDirtied,
      jobsMeshed: j.meshed - p.jMeshed,
      jobsDiscarded: j.discarded - p.jDiscarded,
      occBfsRuns: j.occBfsRuns - p.jOccBfsRuns,
      occVisited: j.occVisited - p.jOccVisited,
    };
    if (dev) { p.drawCalls = dev.drawCalls; p.passes = dev.passCount; p.createBufferCalls = dev.createBufferCalls; p.poolHits = dev.poolHits; }
    p.aRaw = a.rawVertex; p.aDraw = a.drawList; p.aFrustum = a.frustum; p.aEntity = a.entityEmit;
    p.jDirtied = j.dirtied; p.jMeshed = j.meshed; p.jDiscarded = j.discarded;
    p.jOccBfsRuns = j.occBfsRuns; p.jOccVisited = j.occVisited;
    return s;
  }

  // ── Render loop ───────────────────────────────────────────────────────────────

  private readonly loop = (ts: number): void => {
    this.rafId = 0; // the rAF that invoked this tick has fired — clear the spent handle so wake() can re-arm.
    if (this.disposed) return;

    // Hard pause ("lock"): stop the loop entirely; resumeRendering() re-arms. Works in BOTH loop modes.
    // (Harnesses never pause, so this is inert on the default path — behavior-identity holds.)
    if (this.renderingPaused) return;

    // DEFAULT (continuous) mode — byte-for-byte the pre-feature loop: re-arm FIRST (before the device check,
    // so a device===null tick self-heals once the device arrives, exactly like the old loop), then draw every
    // frame. Kept a distinct branch so the integration/harness baselines (recorded against this exact path)
    // are untouched until a caller opts into dynamicFps.
    if (!this.dynamicFps) {
      this.rafId = requestAnimationFrame(this.loop);
      if (!this.device) return;
      this.renderFrame(ts);
      this.accountFps(ts, true); // every continuous tick draws → counts every tick (identical to the old loop)
      return;
    }

    // DYNAMIC-FPS mode (render-on-demand).
    const device = this.device;
    if (!device) return;
    // Don't draw while the tab is hidden if asked — the visibility handler re-arms on return (the rAF path is
    // already paused by the browser while hidden; this also stops the idle poll, so a backgrounded tab does
    // zero render work).
    if (this.pauseWhenHidden && isPageHidden()) return;

    // Sync the canvas first so a resize is reflected in the camera projection (hence changeVersion) before we
    // decide whether the camera moved.
    this.syncCanvasSize();
    this.camera.aspect = this.canvas.width / Math.max(1, this.canvas.height);

    const continuous = isContinuous(this.continuousState());
    const cameraMoved = this.camera.changeVersion !== this.lastDrawnCameraVersion;
    const drew = shouldRenderFrame({ needsRender: this.needsRender, cameraMoved, continuous });
    if (drew) {
      this.needsRender = false;
      this.renderFrame(ts);
      this.lastDrawnCameraVersion = this.camera.changeVersion; // post-render (updatePan may have moved it)
    }
    // FPS counts DRAWN frames only — so an idle (poll-only) stretch truthfully reports ~0 fps instead of
    // freezing the readout at the last busy value, which is how the user confirms dynamic-FPS is idling.
    this.accountFps(ts, drew);
    // Re-arm: stay on vsync while anything is still animating/pending; otherwise drop to the idle backstop.
    // Re-evaluate `continuous` AFTER rendering (this frame's tick may have drained the last mesh job).
    if (pickFrameSchedule({ continuous: isContinuous(this.continuousState()) }) === "raf") {
      this.rafId = requestAnimationFrame(this.loop);
    } else {
      this.renderGapPending = true; // dropping to idle — reseed the dt/timing baseline on the next drawn frame
      this.pollId = window.setTimeout(this.pollWake, idlePollDelayMs(this.idlePollFps));
    }
  };

  /** Idle-poll backstop fired between rendered frames (dynamicFps only) → re-arm one vsync frame so a direct
   *  mutation that bypassed `invalidate()` is still caught. Real changes wake instantly via `wake()`. */
  private readonly pollWake = (): void => {
    this.pollId = 0;
    if (this.disposed || this.renderingPaused || !this.device) return;
    // Don't arm a frame on a hidden tab — mirror wake()/loop(). A poll fired in the macrotask queue can land
    // just after the tab hid (before onVisibilityChange's stopLoop runs); without this it would waste a rAF.
    if (this.pauseWhenHidden && isPageHidden()) return;
    if (this.rafId === 0) this.rafId = requestAnimationFrame(this.loop);
  };

  /** Snapshot of the reasons the loop should keep drawing — fed to the render-on-demand gate. */
  private continuousState(): ContinuousState {
    return {
      reasonCount: this.continuousReasons.size,
      // Pending mesh work spans THREE stages, all of which must keep the loop drawing until they commit:
      // queued/in-flight jobs (hasPendingWork), staged-but-undrained replies, and budget-deferred commits
      // (both folded into UploadScheduler.pendingCount). Missing the upload stages would strand an off-thread
      // build/sort un-drawn the instant inFlight emptied (the loop would idle before drainAndCommit ran).
      meshPending: !!this.scheduler?.hasPendingWork() || (this.uploads?.pendingCount ?? 0) > 0,
      // The master-switch gating lives INSIDE isAnimating() now — pistons are exempt, so an active piston
      // keeps the loop drawing even with animations OFF (it must, to slide smoothly).
      animating: this.isAnimating(),
    };
  }

  /** True if any animation the viewer OWNS should keep the loop drawing this frame. A sliding PISTON and an
   *  opening/closing CONTAINER LID (chest/shulker) always count: they convey mechanism/interaction STATE, not
   *  decoration, so they keep animating even with the animations master switch off (just like pistons). Animated
   *  textures, explosions, and decorative block-entity motions (banner wave, conduit spin, spinning spawner/
   *  vault, end portal) only count while animations are enabled; under render-on-demand (dynamicFps) the master
   *  switch freezes those once the camera settles. Entity/sim motion rides the app's "simulation" reason. */
  private isAnimating(): boolean {
    // Pistons animate while the SIM RUNS, regardless of the animations master switch — but pausing the sim
    // freezes them (so a paused scene can idle), hence the simRunning gate rather than an unconditional pin.
    if (this.simRunning && this.pistons?.active()) return true;
    // Container lids (chest/shulker opening/closing) are an interaction state change, not decoration — like a
    // piston slide they keep drawing until the lid settles, regardless of the master switch (or pause: a lid
    // opened in the paused editor still animates). Self-terminating (~0.5s), so it never pins render-on-demand.
    if (this.entityWorld?.hasContainerLidTransitions()) return true;
    if (!this.animationsEnabled) return false;
    return (
      !!this.scene?.atlas.hasAnimated() ||
      !!this.explosions?.hasActive() ||
      !!this.entityWorld?.hasActiveAnimations()
    );
  }

  /** Mark the scene dirty so the next scheduled frame draws, and wake the loop if it is idle-polling. A no-op
   *  visually in the default (continuous) loop; the editor/simulator call this after any change that should
   *  become visible (block edits, sim diffs, overlay/cursor changes, brightness, scene rebuilds, camera
   *  input). Cheap + idempotent. */
  invalidate(): void {
    this.needsRender = true;
    this.wake();
  }

  /** Keep drawing every frame while `reason` is active (e.g. "simulation" while the sim runs, an orbit drag,
   *  held pan keys). Call with `active=false` to release. Drives continuous (smooth) rendering for things that
   *  animate every frame; only meaningful in dynamicFps mode (the continuous loop already draws every frame). */
  setContinuous(reason: string, active: boolean): void {
    if (active) {
      if (!this.continuousReasons.has(reason)) { this.continuousReasons.add(reason); this.invalidate(); }
    } else {
      this.continuousReasons.delete(reason);
    }
  }

  /** Ensure a frame is queued: cancel a pending idle poll and arm a rAF. No-op if disposed, hard-paused, the
   *  device isn't ready, or the tab is hidden (with pauseWhenHidden) — and in the default loop a frame is
   *  always already armed, so this just clears any stale poll. */
  private wake(): void {
    if (this.disposed || this.renderingPaused || !this.device) return;
    if (this.dynamicFps && this.pauseWhenHidden && isPageHidden()) return;
    if (this.pollId) { clearTimeout(this.pollId); this.pollId = 0; }
    if (this.rafId === 0) this.rafId = requestAnimationFrame(this.loop);
  }

  /** Cancel any pending frame/poll without tearing anything down. */
  private stopLoop(): void {
    if (this.rafId) { cancelAnimationFrame(this.rafId); this.rafId = 0; }
    if (this.pollId) { clearTimeout(this.pollId); this.pollId = 0; }
  }

  /** Hard-pause ("lock") rendering: the loop stops entirely (no rAF, no idle poll) until `resumeRendering`.
   *  The scene, GPU resources, workers, and pending edits all stay live; resume redraws. Works regardless of
   *  the dynamicFps flag — use it to freeze the view (e.g. while a modal owns the screen, or to lock the
   *  frame during a batch of edits). */
  pauseRendering(): void {
    if (this.renderingPaused) return;
    this.renderingPaused = true;
    this.stopLoop();
  }

  /** Resume after `pauseRendering` and request a redraw. */
  resumeRendering(): void {
    if (!this.renderingPaused) return;
    this.renderingPaused = false;
    this.invalidate();
  }

  /** Whether rendering is currently hard-paused (see `pauseRendering`). */
  get renderPaused(): boolean {
    return this.renderingPaused;
  }

  private renderFrame(ts: number): void {
    const device = this.device;
    if (!device) return;
    // Coming out of a dynamicFps idle gap: reseed the timing baselines BEFORE stepDt/the FPS window so the
    // first post-idle frame doesn't report a near-zero FPS spanning the idle interval, nor feed a giant dt
    // into the EMA. No-op in the default loop (renderGapPending is only ever set on the idle-poll path).
    if (this.renderGapPending) { this.renderGapPending = false; this.lastTs = ts; this.fpsLast = ts; this.frames = 0; }
    this.syncCanvasSize();
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.camera.aspect = w / Math.max(1, h);

    const dt = this.stepDt(ts);
    this.updatePan(dt);

    if (this.scene && this.store) {
      // ABP-1: fold this frame's inter-frame dt (stepDt — already spike-clamped to ≤100 ms) into the EMA, and
      // when the adaptive toggle is on, retune the upload budget BEFORE the tick consumes it. Off ⇒ the EMA
      // still tracks (cheap) but the upload budget keeps its fixed literals (no setBudget call). NOTE (P3): `dt`
      // is the rAF delta, which spans the WHOLE previous frame INCLUDING last frame's scheduler tick — so the
      // EMA intentionally measures full frame duration (mesh + render), matching `now − lastFrameAt`.
      // This is the faithful closed-loop pacing signal, NOT a render-only timer (distinct from `lastFrameMs`).
      this.frameEma.push(dt * 1000);
      if (this.adaptiveBudget) this.uploads?.setBudget(this.adaptiveUploadBudget());
      // Phase-1 dispatch (BEFORE the frame timer — meshing is pre-render work, as in the old diff-handler
      // path, so `frameMs` stays comparable render-encode time): mesh the sections enqueued by edits within
      // the CPU budget (SCHED-3), commit through the upload budget (U4). Refresh the draw list ONLY when a
      // section's GPU data changed this frame (a static/idle scene reuses the list — no O(world) rebuild).
      const tick = this.scheduler?.tick(this.schedulerFrame++, this.camera.position); // SCHED-4: near-first
      if (tick) {
        instrument.bumpJob("meshed", tick.meshed);
        instrument.bumpJob("discarded", tick.discarded); // Bug 3: was never bumped → jobsDiscarded pinned at 0
      }
      // F3: refresh the draw list whenever any section's `presented` changed — a COMMIT *or* a DISPOSE
      // (which commits nothing but still drops the section). Keying on `presentedChanges` (not just
      // `tick.committed`) closes the stale-draw gap: a dispose-without-commit would otherwise leave a freed
      // section in `this.draws` for a frame. O(changed), not O(world); a static/idle scene reuses the list.
      if (this.store.presentedChanges.size > 0) this.draws = this.drawList.flatten(this.store);

      const t0 = performance.now();
      // Animations master switch: when OFF, freeze DECORATIVE motion — animated textures stop advancing,
      // explosions stop updating, and entities SNAP to their target instead of tweening. PISTON slides are
      // EXEMPT (they convey redstone mechanism state, not decoration) and keep animating regardless. The
      // world/state still renders either way. Default ON ⇒ these guards are transparent (the pre-feature calls).
      if (this.animationsEnabled) {
        this.scene.atlas.tick(dt * 20);
        this.explosions?.update(dt);
      }
      // partial-tick: fraction (0..1) into the current sim tick, gated only by the diff cadence (tickAnimate).
      // MUST use performance.now() — tickStartMs comes from the diff's performance.now()-based arrival time
      // (Date.now() would mix epochs → garbage). Pistons always use it; entities snap (→1) when the master
      // switch is off.
      const tickFrac = this.tickAnimate ? Math.min(1, Math.max(0, (performance.now() - this.tickStartMs) / Math.max(1, this.tickInterval))) : 1;
      const entityPartial = this.animationsEnabled ? tickFrac : 1;
      // Moving pistons: build this frame's transient geometry and fold it into the terrain draw list —
      // opaque/cutout + the moving translucent draw APPENDED (no terrain section is ever removed) — BEFORE
      // the terrain render (it builds a fresh per-frame draw list). PISTON_PLAN §6.3. Always uses `tickFrac`.
      const draws = this.collectFrameDraws(tickFrac);
      // TR-1: dispatch off-thread DYNAMIC translucent re-sorts for terrain sections the camera moved past
      // (no-op without a worker pool; the worker result commits ≤1 frame later — old index shows until then).
      this.dispatchTranslucentSorts(this.draws, this.camera.position);
      // INFRA-6: pass the uploader's arena layout revision into the F1 bundle key — a relocation invalidates
      // the bundle even in the (TRAP-4.A) corner case where the draws array identity didn't change.
      // Vanilla order: opaque terrain → ENTITIES → translucent terrain. The entity geometry is drawn BETWEEN
      // the two terrain halves (renderInterleaved's callback) so entities depth-order against glass (an entity
      // behind a pane is tinted by it; in front, it occludes it) instead of painting over already-blended
      // translucent terrain. Specials (beams/wireframes/hitboxes) still draw LAST (renderSpecials, after pass 2).
      const layoutRev = this.store.uploadLayoutRevision();
      const ew = this.entityWorld;
      let entMs = 0;
      this.scene.renderer.renderInterleaved(draws, this.camera, w, h, layoutRev, () => {
        if (!ew) return;
        const tEnt = performance.now();
        ew.renderEntities(this.camera, entityPartial, ts / 1000, w, h);
        entMs += performance.now() - tEnt;
      });
      if (ew) {
        const tSp = performance.now();
        ew.renderSpecials();
        entMs += performance.now() - tSp;
      }
      this.lastEntityMs = entMs;
      // Gate the explosion DRAW by the master switch too (not just update() above): otherwise a burst frozen
      // by animationsEnabled=false would keep painting its un-aged flash every frame (update() never ages it
      // out). With the draw gated AND spawnExplosion() a no-op while off, animations-off shows no explosion.
      if (this.animationsEnabled && this.explosions?.hasActive()) {
        const b = this.camera.viewBasis();
        this.explosions.render(this.camera.viewProjection(), b.right, b.up, w, h);
      }
      this.overlay?.render(this.camera.viewProjection(), w, h);
      this.lastFrameMs = performance.now() - t0;
      this.lastStats = this.sampleStats(); // Phase-P: snapshot per-frame metrics (RENDERER_OPTIMIZATION_PLAN §P)
    } else if (device) {
      device.beginPass({ id: null, width: w, height: h }, { color: [0.0049, 0.0058, 0.0067, 1], depth: 1 }).end();
    }
  }

  /** Accumulate the rolling FPS gauge (≥500 ms cadence) and emit it. `drew` is whether this tick actually
   *  rendered — in dynamicFps mode an idle (poll-only) tick passes false so the readout reports the true low
   *  rate, instead of freezing at the last busy value. The continuous loop always passes true. */
  private accountFps(ts: number, drew: boolean): void {
    if (drew) this.frames++;
    if (ts - this.fpsLast >= 500) {
      const fps = (this.frames * 1000) / (ts - this.fpsLast);
      this.onFpsUpdate?.(fps);
      this.frames = 0;
      this.fpsLast = ts;
    }
  }

  private lastTs = 0;
  /** Seconds since the last frame (clamped). */
  private stepDt(ts: number): number {
    const dt = this.lastTs ? Math.min(0.1, (ts - this.lastTs) / 1000) : 0;
    this.lastTs = ts;
    return dt;
  }

  // ── ABP-1 adaptive budget sizing ─────────────────────────────────────────────────────────────────
  // Scale the streaming budgets off the frame-time EMA: cheap frames (headroom) earn a
  // bigger budget so meshing/upload catch up faster; expensive frames shrink it so streaming doesn't
  // deepen a stutter. Clamped both ways so one extreme reading can't starve OR flood a frame.
  private adaptiveScale(): number {
    const TARGET_MS = 1000 / 60; // aim for a 60 Hz frame; more headroom ⇒ scale > 1, over-budget ⇒ < 1
    return Math.min(2, Math.max(0.25, TARGET_MS / this.frameEma.average));
  }

  /** ABP-1: this frame's adaptive CPU mesh budget (ms), read by the Scheduler via `meshBudgetOf`. */
  private adaptiveMeshBudgetMs(): number {
    const BASE = 8; // the Scheduler's default meshBudgetMs
    return Math.min(16, Math.max(2, BASE * this.adaptiveScale()));
  }

  /** FS-5: the mesh-time options derived from the viewer toggles, shared by the initial commit, the inline
   *  scheduler, and the worker pool init — so every meshing path agrees on the byte layout. */
  private meshOptions(): MeshOptions {
    return {
      partitionFacing: this.faceSliceCull,
      smoothLighting: this.smoothLighting,
      perPerspectiveVisibility: this.perPerspectiveVisibility, // OCC-2: emit DIRECTION_SETS (4-element visibilityData)
      occlusionAngleMask: this.occlusionAngleMask,
    };
  }

  /** SL-4: enable/disable smooth lighting. Changes baked vertex bytes (AO byte18 + per-corner light), so it
   *  takes effect on the next scene build — flip BEFORE loading, or call `rebuild()`/reload to force a remesh.
   *  Distinct from `setLightLut` (the light LEVEL → LUT, a free remesh-free reupload). */
  setSmoothLighting(on: boolean): void {
    this.smoothLighting = on;
  }

  /** TR-1: enable/disable the DEFAULT-OFF huge-section distance/angle re-sort gate. Atomically updates the
   *  inline renderer field and this viewer's off-thread dispatch (single source of truth). Index-only — never
   *  remeshes, never changes the generation tag; off ⇒ byte-identical render. */
  setHugeSortGate(on: boolean): void {
    this._hugeSortGate = on;
    if (this.scene) this.scene.renderer.hugeSortGate = on;
  }

  /** Enable/disable multithreaded (off-thread worker-pool) meshing. The pool is created at scene build, so
   *  flipping this rebuilds the CURRENT scene to (re)create or drop the pool — `buildScene` re-meshes the live
   *  model, so block state is preserved (the sim runs on a separate worker, untouched). No rebuild before the
   *  first scene exists (the initial build reads the flag). Off ⇒ the inline meshing fallback (identical output). */
  setMultithreading(on: boolean): void {
    if (this.multithreading === on) return;
    this.multithreading = on;
    if (this.scene) this.sceneReady = this.buildScene(true);
  }

  /** ABP-1: this frame's adaptive upload budget (bytes + ms), merged onto the UploadScheduler. */
  private adaptiveUploadBudget(): Partial<UploadBudget> {
    const s = this.adaptiveScale();
    return {
      maxBytesPerFrame: Math.min(16 * 1024 * 1024, Math.max(1 * 1024 * 1024, DEFAULT_UPLOAD_BUDGET.maxBytesPerFrame * s)),
      maxMillisPerFrame: Math.min(8, Math.max(1, DEFAULT_UPLOAD_BUDGET.maxMillisPerFrame * s)),
    };
  }

  /**
   * Fold this frame's transient piston geometry into the committed terrain draw list. Opaque/cutout
   * moving blocks are appended as terrain draws (same shader → exact shading); a moving translucent block
   * (glass/slime) is appended as its OWN self-sorted, depth-writing translucent draw (vanilla's
   * `translucentMovingBlock`). No terrain section is ever removed — depth, not a geometry merge, orders
   * the moving glass against static glass (BUG_REPORT.md). Returns `this.draws` unchanged when no pistons
   * are animating.
   */
  private collectFrameDraws(partial: number): SectionDraw[] {
    const terrain = this.occludedTerrainDraws();
    const pistons = this.pistons;
    if (!pistons || !pistons.active()) return terrain;
    const { opaqueDraws, translucent } = pistons.frame(this.camera.position, partial);
    if (opaqueDraws.length === 0 && !translucent) return terrain;
    const out = terrain.slice();
    for (const d of opaqueDraws) out.push(d);
    if (translucent) out.push(translucent);
    return out;
  }

  // ── OCC-1 occlusion culling (camera BFS over the section visibility graph) ──────────────────────────
  /** Toggle the occlusion cull (verified on harness/stress; on by default — a strict perf win when correct). */
  occlusionEnabled = true;
  /** OCC-1 reachability cache (DEFAULT ON — P1.1). The BFS produces an orientation-INDEPENDENT reachable set
   *  cached on the clamped home section + graphRevision, so it re-runs only on a home-section crossing or a
   *  graph bump — NOT on every rotation/pan frame (closes H6). The live frustum is applied downstream (the
   *  renderer's own `testSection`), exactly like the now-orientation-independent legacy path (P0.3); the two
   *  draw identical pixels, the cache just amortizes the BFS across a home section. OFF ⇒ the per-frame legacy
   *  path (still correct, just un-amortized). Safe with `perPerspectiveVisibility` (OCC-2) on because that
   *  join now returns the quadrant-INDEPENDENT symmetric union (P2.2/M1); if the octant refinement is ever
   *  restored, the camera octant must be folded into the cache key first (see SectionVisibility). */
  occBfsCache = true;
  /** MEM-1 region cull PRE-PASS (DEFAULT OFF). When OFF, `occlusionVisibleSet` is called with no region
   *  options ⇒ the BFS bounds are the global populated-section box exactly as today (byte-identical visible
   *  set). When ON (and the store carries a RegionStore tier), the BFS bounds tighten to the union of
   *  in-frustum, populated 256-section region AABBs — rejecting whole out-of-frustum regions with one box
   *  test before the UNCHANGED per-section BFS. The visible SET is identical (a region box contains its
   *  section boxes, so the same conservative frustum never drops an in-frustum section — TRAP MEM-1.C);
   *  only the traversal domain shrinks. Pairs with the uploader's `regionArenas` flag (per-region arenas +
   *  contiguous reclaim); both off ⇒ the single-arena, section-granular path. */
  viewerRegionCullEnabled = false;
  private occVisible: Set<number> | null = null;
  private readonly occLastVP = new Float32Array(16);
  private occLastGraphRev = -1;
  /** OCC-1 cache key: the clamped home section the cached reachable set was computed for. NaN ⇒ no cache yet.
   *  The reachable BFS re-runs when this crosses a section boundary (or `occLastGraphRev` changes). Reused. */
  private readonly occLastHomeSec: [number, number, number] = [NaN, NaN, NaN];
  /** OCC-1 caller-owned reachable set (reused across runs — the zero-alloc path). Holds the orientation-
   *  independent reachable membership when `occBfsCache` is ON. Double-buffered with `occReachablePrev` so a
   *  re-run can detect whether membership ACTUALLY changed (→ fresh `occDraws`) vs is identical (→ keep the
   *  same array, stable identity for the F1 key — TRAP OCC-1.C). */
  private occReachable = new Set<number>();
  private occReachablePrev = new Set<number>();
  /** Occlusion-filtered terrain list. Re-allocated FRESH on each re-filter (not refilled in place) so its
   *  array IDENTITY versions the content — the F1 render-bundle key relies on a fresh array whenever the
   *  visible set / membership changes (a stale bundle must never replay). Stable ref on unchanged frames. */
  private occDraws: SectionDraw[] = [];
  private occDrawsSrc: SectionDraw[] | null = null;
  /** TR-1 off-thread re-sort trigger: per-section (keyed on the presented translucent vertex ArrayBuffer)
   *  last-sorted camera + last-sorted forward + unique planes. Mirrors the renderer's inline trigger, but
   *  dispatches to a worker. `lastFwd` is the camera forward the index was last sorted for (the huge-section
   *  gate's angle reference; advanced only on an ACTUAL dispatch — TRAP TR-1.C). */
  private readonly sortTrigger = new WeakMap<ArrayBuffer, { lastCam: Vec3; lastFwd: Vec3; planes: Float32Array }>();
  /** P1.4 / M4: reused scratch for the per-draw trigger TEST, so `dispatchTranslucentSorts` allocates nothing
   *  per dynamic translucent draw per frame (the inline path's A7 treatment — an owned copy is persisted into
   *  the trigger state ONLY on an actual dispatch, which is gated by `crossedAnyPlane` + the dispatch cap). */
  private readonly scratchSortCamLocal: [number, number, number] = [0, 0, 0];
  private sortJobId = 1;
  /** Off-thread re-sorts dispatched (TR-1) — folded into the cumulative `resorts` metric alongside the
   *  renderer's inline `resortCount`, so the gauge reflects total re-sort activity on either path. */
  private offThreadResorts = 0;
  /** TR-1 SINGLE SOURCE OF TRUTH for the huge-section distance/angle re-sort gate (DEFAULT-OFF). Drives BOTH
   *  the inline renderer (`renderer.hugeSortGate`, set wherever `offThreadSort` is) and this off-thread
   *  dispatch, so toggling is atomic across paths. When false, both sites take today's exact `crossedAnyPlane`
   *  path → byte-identical render. Flip via `setHugeSortGate`. */
  private _hugeSortGate = false;

  /**
   * The terrain draw list filtered to occlusion-visible sections.
   *
   * DEFAULT (occBfsCache ON) — the OCC-1 split: the BFS produces an orientation-INDEPENDENT reachable set
   * cached on the clamped home section + `graphRevision`, so it re-runs only on a home-section crossing or a
   * graph bump (≈0 across a pure rotation/orbit). The live frustum is NOT fused into traversal (that would
   * leave stale reachability holes — H1 / TRAP OCC-1.A); on-screen pruning is left to the renderer's own
   * per-frame `testSection`. A fresh `occDraws` is allocated only when reachable MEMBERSHIP actually changes
   * (TRAP OCC-1.C — its identity feeds the F1 bundle key); a rotation-only frame keeps the same array.
   *
   * occBfsCache OFF (or OCC-2 on — see the field note) — the legacy path: the SAME orientation-independent
   * BFS (P0.3) re-runs whenever the camera's view-projection or the occlusion graph (`store.graphRevision`,
   * N4) changes; the cheap per-draw filter re-runs only when that set or `this.draws` changed (a fully static
   * frame reuses the cached list).
   *
   * Empty/absent sections are traversed but never drawn (OCC-6); entities/transients are NOT occlusion-
   * culled (the invariant: entities never pop).
   */
  private occludedTerrainDraws(): SectionDraw[] {
    if (!this.store || !this.occlusionEnabled) return this.draws;
    // P2.2 / M2: OCC-2's `joinVisibilityData` now returns the symmetric union (quadrant-INDEPENDENT — M1), so
    // the home-section reachability cache is safe even with `perPerspectiveVisibility` on. (If the faithful
    // octant refinement is ever restored, the camera octant MUST be folded into the cache key before the cached
    // path may be used with OCC-2 — see SectionVisibility.joinVisibilityData.)
    return this.occBfsCache ? this.occludedTerrainDrawsCached() : this.occludedTerrainDrawsLegacy();
  }

  /** OCC-2: the per-perspective DIRECTION_SETS join knobs handed to `occlusionVisibleSet`. Both default OFF ⇒
   *  `undefined` per field ⇒ the BFS takes the exact symmetric path (byte-identical visible set). The join key
   *  is the camera QUADRANT (a function of camera position, not orientation), so it composes with BOTH the
   *  occBfsCache ON path (orientation-independent reachability — quadrant is position-keyed, re-run on a home
   *  crossing) and the OFF legacy path. The angle sub-flag only applies when perPerspective is also on. */
  private occ2Options(): { perPerspective?: boolean; angleMask?: boolean } | undefined {
    if (!this.perPerspectiveVisibility) return undefined;
    return { perPerspective: true, angleMask: this.occlusionAngleMask };
  }

  /** MEM-1: the region cull options handed to `occlusionVisibleSet`. Undefined (default) ⇒ no pre-pass ⇒
   *  the BFS bounds are the global box (byte-identical). Both live callers now pass `undefined` (P0.3 / H1:
   *  the frustum must never gate BFS traversal — at the section OR the region tier — or it drops geometry
   *  reachable only THROUGH an out-of-frustum region), so the tightening is the orientation-independent
   *  populated-region union. The `regionInFrustum` branch is retained for a future distance-bounded caller. */
  private regionCullOptions(frustum?: Frustum): RegionCullOptions | undefined {
    if (!this.viewerRegionCullEnabled || !this.store?.regionStore) return undefined;
    if (!frustum) return {}; // orientation-independent: tighten to the populated-region union only
    return { regionInFrustum: (lx, ly, lz, hx, hy, hz) => frustum.testAab(lx, ly, lz, hx, hy, hz) };
  }

  /** OFF-path (occBfsCache=false): the per-frame BFS keyed on any-viewProj-float change, fresh `occDraws`
   *  per re-filter. P0.3 (H1): the BFS is ORIENTATION-INDEPENDENT — the live frustum is NOT fused into
   *  traversal; it is applied downstream by the renderer's per-section `testSection`. (The cached path,
   *  P1.1, additionally amortizes this same BFS across a home section.) */
  private occludedTerrainDrawsLegacy(): SectionDraw[] {
    const vp = this.camera.viewProjection();
    let changed = this.occVisible === null || this.store!.graphRevision !== this.occLastGraphRev;
    for (let i = 0; i < 16 && !changed; i++) if (vp[i] !== this.occLastVP[i]) changed = true;
    if (changed) {
      // P0.3 (H1): traverse the reachable set with an ALWAYS-TRUE predicate (`occAllVisible`). Fusing the live
      // frustum into BFS traversal dead-ends the search at the frustum boundary and DROPS geometry reachable
      // only THROUGH an out-of-frustum section (a hole — e.g. a room behind a doorway section that sits just
      // past a side plane). Wide traversal tier is likewise distance-bounded, never frustum-gated; the
      // frustum is the per-section DRAW filter (TerrainRenderer.render `frustum.testSection`), applied AFTER
      // occlusion. Bounded by the populated-region AABB + outward culling. `regionCullOptions(undefined)` keeps
      // the region tightening orientation-independent too (matching the cached path; MEM-1 is default-off).
      this.occVisible = occlusionVisibleSet(this.store!, this.camera.position, occAllVisible, undefined,
        this.occ2Options(), this.regionCullOptions(undefined));
      this.occLastVP.set(vp);
      this.occLastGraphRev = this.store!.graphRevision;
    }
    if (changed || this.occDrawsSrc !== this.draws) {
      // Fresh array (not an in-place refill) so identity tracks content — see the field comment (F1 key).
      this.occDraws = filterDrawsByOcclusion(this.draws, this.occVisible!, []);
      this.occDrawsSrc = this.draws;
    }
    return this.occDraws;
  }

  /** ON-path: the OCC-1 home-section-keyed reachability cache. The BFS re-runs only on a home-section
   *  crossing or a graphRevision bump; the live frustum is applied downstream by the renderer (Option B). */
  private occludedTerrainDrawsCached(): SectionDraw[] {
    const store = this.store!;
    const graphRev = store.graphRevision;
    // Clamped home section (initOutsideWorld). null ⇒ empty world ⇒ keep an empty reachable set.
    const home = occlusionHomeSection(store, this.camera.position);
    const homeChanged =
      home === null
        ? !(Number.isNaN(this.occLastHomeSec[0])) // a non-empty→empty transition is a change
        : home[0] !== this.occLastHomeSec[0] || home[1] !== this.occLastHomeSec[1] || home[2] !== this.occLastHomeSec[2];
    const reachabilityStale = this.occVisible === null || graphRev !== this.occLastGraphRev || homeChanged;

    let membershipChanged = false;
    if (reachabilityStale) {
      // Swap buffers so the new BFS writes into a clean set while `occReachablePrev` holds the old membership
      // for the change check. occlusionVisibleSet clears the target itself.
      const next = this.occReachablePrev;
      this.occReachablePrev = this.occReachable;
      this.occReachable = next;
      // Reachability query: pass-through inFrustum (always true) so the cached set is orientation-independent.
      // The `bounds` AABB + outward culling still bound traversal (camera-section-stable); the live frustum is
      // reapplied downstream by the renderer's testSection (TRAP OCC-1.A/.B). NOTE: home is the clamped origin
      // for keying; occlusionVisibleSet re-derives the same clamped origin internally from camera.position.
      const wasNull = this.occVisible === null; // first run: no prior membership ⇒ force a fresh occDraws
      // MEM-1: orientation-INDEPENDENT region tightening (no frustum predicate) so the cached reachable set
      // stays keyed on the home section + graphRev only — the union-of-populated-regions bound doesn't
      // depend on camera orientation, preserving the OCC-1 cache (TRAP OCC-1.A).
      occlusionVisibleSet(store, this.camera.position, occAllVisible, this.occReachable, this.occ2Options(),
        this.regionCullOptions(undefined));
      this.occVisible = this.occReachable;
      membershipChanged = wasNull || !setsEqual(this.occReachable, this.occReachablePrev);
      if (home === null) { this.occLastHomeSec[0] = this.occLastHomeSec[1] = this.occLastHomeSec[2] = NaN; }
      else { this.occLastHomeSec[0] = home[0]; this.occLastHomeSec[1] = home[1]; this.occLastHomeSec[2] = home[2]; }
      this.occLastGraphRev = graphRev;
    }
    // Re-filter only when membership actually changed or the source draw list changed. On a rotation-only
    // frame both are stable ⇒ occDraws is the SAME array (stable identity, F1-coherent) and the renderer's
    // per-frame frustum cull drops off-screen sections.
    if (membershipChanged || this.occDrawsSrc !== this.draws) {
      this.occDraws = filterDrawsByOcclusion(this.draws, this.occVisible!, []);
      this.occDrawsSrc = this.draws;
    }
    return this.occDraws;
  }

  /**
   * TR-1/TR-6: dispatch off-thread DYNAMIC translucent re-sorts for terrain sections whose quad-plane the
   * camera crossed this frame (≤ SORT_DISPATCH_CAP/frame). The worker computes the new index off the render
   * thread; it commits generation-matched (TRAP 5.B), and the old committed index draws until then (no pop —
   * just ≤1 frame late). No-op without a worker pool (then the renderer re-sorts inline, offThreadSort=false).
   */
  private dispatchTranslucentSorts(draws: SectionDraw[], camPos: Vec3): void {
    const pool = this.workerPool;
    if (!pool || !this.store) return;
    // TR-1: the camera FORWARD is the huge-section gate's angle reference (memoized/zero-alloc on `Camera`).
    const curFwd = this.camera.viewBasis().forward;
    const gate = this._hugeSortGate;
    // M4: reused scratch for the per-draw TRIGGER TEST — an owned copy is persisted into the state only on an
    // actual dispatch (the inline path's A7 pattern), so a static-but-changed frame allocates per-DISPATCH, not
    // per dynamic draw.
    const camLocal = this.scratchSortCamLocal;
    let dispatched = 0;
    for (const d of draws) {
      if (d.pass !== TerrainPass.Translucent || d.sortType !== SortType.Dynamic) continue;
      if (!d.cpuVertex || !d.sortQuads || d.sortQuads.length === 0 || !d.index) continue;
      const o = d.originBlocks;
      camLocal[0] = camPos[0] - o[0]; camLocal[1] = camPos[1] - o[1]; camLocal[2] = camPos[2] - o[2];
      let st = this.sortTrigger.get(d.cpuVertex);
      let need = false;
      if (!st) {
        // First sight → an initial camera-relative sort (ALWAYS — never gated; the mesh-time index shows until
        // it lands). Snapshot OWNED copies (the scratch is overwritten next draw) so the gate's deadband is
        // anchored from here.
        st = { lastCam: [camLocal[0], camLocal[1], camLocal[2]], lastFwd: [curFwd[0], curFwd[1], curFwd[2]], planes: uniquePlanes(d.sortQuads) };
        this.sortTrigger.set(d.cpuVertex, st);
        need = true;
      } else if (
        crossedAnyPlane(st.planes, st.lastCam, camLocal) &&
        // TR-1: huge-section damper layered ON TOP of crossedAnyPlane (NEVER a replacement — TRAP TR-1.A);
        // below the threshold OR gate-off ⇒ today's exact behavior (TRAP TR-1.B). Short-circuits on !gate.
        (!gate || d.quadCount < GATE_QUAD_THRESHOLD ||
          farAngleResortAllowed(GATE_CENTER_LOCAL, st.lastCam, camLocal, st.lastFwd, curFwd, GATE_DIST_EPS, GATE_ANGLE_COS))
      ) {
        // Advance the deadband anchor ONLY here (the actual-dispatch branch — TRAP TR-1.C) with OWNED copies.
        st.lastCam = [camLocal[0], camLocal[1], camLocal[2]];
        st.lastFwd = [curFwd[0], curFwd[1], curFwd[2]];
        need = true;
      }
      if (!need) continue;
      if (dispatched >= SORT_DISPATCH_CAP) break; // TR-6
      const key = sectionKey(o[0] >> 4, o[1] >> 4, o[2] >> 4);
      const section = this.store.get(key);
      if (!section?.presented) continue;
      // Tag with the PRESENTED generation (the quads are presented geometry's): acceptSort drops the sort if
      // a rebuild has since advanced the section — that rebuild re-triggers a sort against the new geometry.
      // `st.lastCam` is the just-anchored section-local camera (an owned array, only ever reassigned — never
      // mutated in place — so it is safe to hand to the worker message even if read asynchronously).
      pool.post({ type: "sort", jobId: this.sortJobId++, sectionKey: key, generation: section.presented.generation, sortData: serializeQuads(d.sortQuads), cameraPos: st.lastCam });
      dispatched++;
    }
    this.offThreadResorts += dispatched;
  }

  /** SCHED-4 commit priority: nearer-to-camera sections commit FIRST (higher = sooner) under the upload
   *  budget. Negative Chebyshev section-distance from the camera — tie-breaks among same-age pending
   *  sections so a burst presents near edits before far ones. Aging (UploadScheduler) still prevents
   *  starvation. */
  private uploadPriority(section: { sx: number; sy: number; sz: number }): number {
    const p = this.camera.position;
    const cx = Math.floor(p[0]) >> 4, cy = Math.floor(p[1]) >> 4, cz = Math.floor(p[2]) >> 4;
    // P1.2: read the section's cached numeric coords (parsed once at construction) — no per-section string
    // split in this per-frame `UploadScheduler.priorityOf` hook.
    return -Math.max(Math.abs(section.sx - cx), Math.abs(section.sy - cy), Math.abs(section.sz - cz));
  }

  /** ABP-2: a section is "near" (eligible for same-frame inline meshing) when its Chebyshev distance from
   *  the camera's home section is within a couple of sections. Distance-only for now (frustum test is a
   *  later refinement); reuses `uploadPriority` (= −Chebyshev distance) so the metric can't drift. */
  private static readonly IMMEDIATE_NEAR_SECTIONS = 2;
  private isNearImmediate(key: SectionKey): boolean {
    // P1.2: read cached coords via the section (the following markDirty get-or-creates it regardless, so this
    // is idempotent) — no string parse. Only reached when `immediatePresent` is on (default-off).
    return -this.uploadPriority(this.store!.getOrCreate(key)) <= SchematicViewer.IMMEDIATE_NEAR_SECTIONS;
  }

  // ── Canvas / input ─────────────────────────────────────────────────────────

  private readonly onResize = (): void => {
    // While hard-paused we can't redraw, so don't resize the swapchain (it would present a stale/garbage
    // frame at the new size). renderFrame() calls syncCanvasSize() at its top, and resumeRendering() forces a
    // draw, so the new size is picked up automatically on resume.
    if (this.renderingPaused) return;
    this.syncCanvasSize();
    this.invalidate();
  };

  private syncCanvasSize(): void {
    if (!this.device) return;
    const scale = window.devicePixelRatio || 1;
    const w = Math.min(MAX_DIM, Math.max(1, Math.round(this.canvas.clientWidth * scale)));
    const h = Math.min(MAX_DIM, Math.max(1, Math.round(this.canvas.clientHeight * scale)));
    if (this.canvas.width === w && this.canvas.height === h) return;
    this.canvas.width = w;
    this.canvas.height = h;
    this.device.resize(w, h);
  }

  private dragging = false;
  private lastX = 0;
  private lastY = 0;

  private attachInput(): void {
    this.canvas.addEventListener("pointerdown", this.onPointerDown);
    window.addEventListener("pointerup", this.onPointerUp);
    window.addEventListener("pointermove", this.onPointerMove);
    this.canvas.addEventListener("wheel", this.onWheel, { passive: false });
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);
    window.addEventListener("blur", this.onWindowBlur);
  }
  private detachInput(): void {
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    window.removeEventListener("pointerup", this.onPointerUp);
    window.removeEventListener("pointermove", this.onPointerMove);
    this.canvas.removeEventListener("wheel", this.onWheel);
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
    window.removeEventListener("blur", this.onWindowBlur);
  }

  // ── Keyboard camera pan (WASD pan · R/F up·down) ───────────────────────────
  /** Currently-held pan keys (lowercased). Cleared on key-up and window blur. */
  private readonly pressedKeys = new Set<string>();

  private readonly onKeyDown = (e: KeyboardEvent): void => {
    if (isTextEntry(e.target)) return; // typing in a GUI field — leave the camera alone
    const k = e.key.toLowerCase();
    if (PAN_KEYS.has(k)) { this.pressedKeys.add(k); this.setContinuous("pan", true); }
  };
  private readonly onKeyUp = (e: KeyboardEvent): void => {
    this.pressedKeys.delete(e.key.toLowerCase());
    if (this.pressedKeys.size === 0) { this.setContinuous("pan", false); this.invalidate(); }
  };
  private readonly onWindowBlur = (): void => { this.pressedKeys.clear(); this.setContinuous("pan", false); };

  /** Per-frame camera pan from held keys. Moves the orbit target (the eye follows), so panning keeps
   *  the current orbit angle/zoom. Speed scales with orbit distance so it feels consistent at any zoom. */
  private updatePan(dt: number): void {
    if (dt <= 0 || this.pressedKeys.size === 0) return;
    const p = this.pressedKeys;
    const fwdAxis = (p.has("w") ? 1 : 0) - (p.has("s") ? 1 : 0); // forward / back
    const rightAxis = (p.has("d") ? 1 : 0) - (p.has("a") ? 1 : 0); // strafe
    const upAxis = (p.has("r") ? 1 : 0) - (p.has("f") ? 1 : 0); // world up / down
    if (fwdAxis === 0 && rightAxis === 0 && upAxis === 0) return;

    const b = this.camera.viewBasis();
    // Forward is the camera facing projected onto the ground plane; right is already horizontal.
    const fLen = Math.hypot(b.forward[0], b.forward[2]) || 1;
    const fx = b.forward[0] / fLen;
    const fz = b.forward[2] / fLen;
    let mx = fx * fwdAxis + b.right[0] * rightAxis;
    const my = upAxis;
    let mz = fz * fwdAxis + b.right[2] * rightAxis;
    const len = Math.hypot(mx, my, mz) || 1; // normalize so diagonals aren't faster
    const sprint = p.has("shift") ? 2.5 : 1;
    const speed = Math.max(6, this.camera.distance * 0.9) * sprint; // world units / second
    const step = (speed * dt) / len;
    const t = this.camera.target;
    this.camera.target = [t[0] + mx * step, t[1] + my * step, t[2] + mz * step];
  }

  private readonly onPointerDown = (e: PointerEvent): void => {
    // Left- or middle-button drag orbits the camera — no modifier key (shift/alt) required.
    if (e.button === 0 || e.button === 1) {
      this.dragging = true;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      e.preventDefault(); // suppress text selection / native drag while orbiting
    }
  };
  private readonly onPointerUp = (): void => {
    this.dragging = false;
  };
  private readonly onPointerMove = (e: PointerEvent): void => {
    if (!this.dragging) return;
    this.camera.yaw -= (e.clientX - this.lastX) * 0.01;
    this.camera.pitch = Math.max(-1.5, Math.min(1.5, this.camera.pitch + (e.clientY - this.lastY) * 0.01));
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.invalidate(); // wake from idle so the orbit drag is responsive (no damping → no continuous reason needed)
  };
  private readonly onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    this.camera.distance = Math.max(1.5, Math.min(2048, this.camera.distance * (e.deltaY > 0 ? 1.1 : 0.9)));
    this.invalidate(); // wake from idle so the zoom takes effect this frame
  };

  // ── Teardown ─────────────────────────────────────────────────────────────────

  /** Spawn the off-thread mesh pool for a scene, or null (→ inline meshing) if workers can't be created.
   *  The init carries the TRANSFERRED baked palette models + fluids (no re-baking ⇒ byte-identical). */
  /** Optional injectable mesher-worker factory. Default (undefined) uses the production pattern: a module Worker
   *  from `new URL("../workers/mesher.worker.ts", import.meta.url)`, which turbopack/webpack/vite bundle. The
   *  esbuild harness serve does NOT bundle that `new URL` (it leaves the raw `.ts`, which won't run), so harness
   *  pages inject a factory pointing at a prebuilt worker bundle (`dist/mesher.worker.entry.js`). Set before
   *  `loadSchematic`. Lets the renderer use REAL off-thread meshing wherever the default URL can't be bundled. */
  workerFactory?: (index: number) => Worker;

  private makeWorkerPool(provider: Scene["provider"], fluids: FluidContext, palette: WorldModel["palette"]): WorkerPool | null {
    if (!this.multithreading) return null; // multithreading disabled → inline meshing (the fallback path)
    try {
      const ids = [AIR, ...Object.keys(palette).map(Number)];
      const meshOpts = this.meshOptions(); // FS-5: ship the mesh toggles so off-thread output matches inline
      const init = extractMeshInit(provider, ids, fluids, meshOpts);
      const createWorker = this.workerFactory ?? ((index: number): Worker => {
        // Name each worker (renderer-mesher-0, -1, …) so it's identifiable in DevTools' thread list / profiler.
        const w = new Worker(new URL("../workers/mesher.worker.ts", import.meta.url), { type: "module", name: `renderer-mesher-${index}` });
        w.onerror = (ev) => console.warn("[viewer] mesher worker error", ev.message ?? ev);
        return w;
      });
      // Model B (Phase 5): when the page is cross-origin isolated, workers drain build jobs from a SHARED
      // ring + read snapshots zero-copy (no per-job structured clone — the large-world win). Otherwise
      // degrade to Model A (postMessage + clone). Feature-detected so it ships safely without COOP/COEP
      // headers — it simply stays on Model A until the app server sets them (TRAP 5.F).
      if (sharedMemoryAvailable()) {
        const queue = SharedMeshQueue.create({ apron: APRON });
        this.sharedMeshQueue = queue; // WRK-1: expose the ring's in-flight byte estimate to the admission gate
        return new WorkerPool({ mode: "shared", init, createWorker, queue, fallbackMesh: (snap) => meshSection(snap, provider, fluids, meshOpts) });
      }
      this.sharedMeshQueue = null; // Model A: the gate uses the Scheduler's own main-thread byte accumulator
      // WRK-3: pass the cooperative-cancel toggle (default-off). When on (and SAB present) the pool allocates a
      // cancel SAB + ships it in init so a superseded build stops meshing; off ⇒ exact current Model-A path.
      // WRK-2: pass the work-stealing toggle (default-off) ONLY here (Model A) — Model B already steals via the
      // SAB semaphore. Off ⇒ the exact rr++ dispatch with no epoch attached; the epoch gate stays disarmed.
      return new WorkerPool({ mode: "worker", init, createWorker, cooperativeCancelA: this.cooperativeCancelA, workStealing: this.workStealEnabled });
    } catch (e) {
      this.sharedMeshQueue = null;
      console.warn("[viewer] off-thread meshing unavailable — using inline meshing", e);
      return null;
    }
  }

  /**
   * Ship newly-interned palette ids to the mesh workers so off-thread meshing renders them instead of AIR
   * (audit Bug 1). A property-only edit (lamp `lit=true`, door `open=true`, redstone `power=N`) mints a NEW
   * palette id whose block NAME is already atlased — so it does NOT trip the name-keyed `paletteStale`
   * rebuild that would re-init the pool. The worker's model map is frozen at scene-build init, so without
   * pushing the new id's baked model `provider.get(newId)` returns null and the block meshes empty. Ids with
   * an UNATLASED name are skipped (a full rebuild re-inits the pool for those). No-op when meshing inline
   * (the inline mesher uses the live main-thread provider directly).
   */
  private syncWorkerModels(): void {
    const pool = this.workerPool;
    const model = this.model;
    const scene = this.scene;
    if (!pool || !model || !scene || this.paletteStale) return; // a rebuild (pool re-init) is already coming
    const newIds = unknownAtlasedPaletteIds(model.palette, this.workerKnownIds, this.atlasedNames);
    if (newIds.length === 0) return;
    // Bake each new id against the CURRENT atlas (the same provider the inline mesher uses) and broadcast to
    // every worker. Posted synchronously here — before the render loop's next scheduler.tick dispatches the
    // edit's build — so the target worker has the model before it meshes the section (per-worker FIFO order).
    pool.post({ type: "addModels", models: extractPaletteModels(scene.provider, newIds) });
    for (const id of newIds) this.workerKnownIds.add(id);
  }

  private disposeScene(): void {
    this.workerPool?.dispose();
    this.workerPool = null;
    this.entityWorld?.dispose();
    this.entityWorld = null;
    this.overlay?.dispose();
    this.overlay = null;
    this.explosions?.dispose();
    this.explosions = null;
    if (this.store) {
      for (const s of [...this.store.all()]) this.store.dispose(s.key);
    }
    this.scene?.renderer.dispose();
    this.scene?.atlas.dispose(); // O1: release the block atlas texture + CPU mirrors on scene teardown
    this.scene = null;
    this.store = null;
    this.snapshots?.clear(); // deterministic source/light-cube cache release on teardown (don't wait for GC)
    this.snapshots = null;
    this.scheduler = null;
    this.uploads = null;
    this.draws = [];
  }

  dispose(): void {
    this.disposed = true;
    this.stopLoop(); // cancels both the rAF and the dynamicFps idle poll
    this.detachInput();
    window.removeEventListener("resize", this.onResize);
    document.removeEventListener("visibilitychange", this.onVisibilityChange);
    this.disposeScene();
    // The piston driver owns device-tied GPU buffers (opaque cache + per-frame translucent); it lives
    // across scene swaps (invalidated, not disposed, on rebuild) so it's torn down here, before the device.
    this.pistons?.dispose();
    this.pistons = null;
    this.device?.destroy?.();
    this.device = null;
  }
}

export { AIR };
