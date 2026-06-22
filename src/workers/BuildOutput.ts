// Worker build output — a RESOURCE with a single owner, not a plain message.
// RENDERER_PLAN.md §6 (ownership), §11 (independent vertex/index replacement).
//
// Carries transferable typed arrays. Stale/cancelled outputs MUST be disposed so their
// ArrayBuffers are released (§22 test: "stale outputs release transferred buffers").

import type { Generation, TerrainPass } from "../types";
import type { SectionKey } from "../world/SectionKey";
import { SortType } from "../mesh/SortTypes";
import type { TQuad } from "../mesh/TranslucentCollector";

export interface MeshPart {
  vertexData: ArrayBuffer;
  quadCount: number;
  /** FS-2: per-facing VERTEX counts (Uint32Array(7), canonical facing order 0..6) when the section was
   *  meshed with `partitionFacing`. The concatenated vertex runs are in this same order, so the draw layer
   *  can emit one drawIndexed per visible facing run (FS-4). Absent ⇒ one undifferentiated run (legacy
   *  single draw). Vertex-relative ⇒ survives arena relocation. Structured-cloned (tiny, not transferred). */
  facingVertexCounts?: Uint32Array;
}

export interface TranslucentMeshPart extends MeshPart {
  sortType: SortType;
  /** Present only when order differs from the shared quad EBO. */
  indexData?: ArrayBuffer;
  /** Per-quad geometry for runtime topological re-sort (Dynamic sections); CPU-side, inline path only. */
  quads?: TQuad[];
  quadHash: number;
}

/** Compact per-section render info (flags, visibility, animated sprites, entities). */
export interface SectionRenderInfo {
  flags: number;
  visibilityData?: Uint32Array;
  animatedSpriteIds?: Uint32Array;
}

/**
 * Serialized, transferable build output — the on-the-wire shape between a worker and the main thread, and
 * the canonical BASE for the richer build types. No `dispose()` / `approxBytes` (the main thread re-wraps
 * via `makeBuildOutput`). The other two shapes extend this, so the 5 shared fields live in ONE place.
 */
export interface BuildPayload {
  sectionKey: SectionKey;
  generation: Generation;
  info: SectionRenderInfo;
  parts: Partial<Record<TerrainPass, MeshPart>>;
  translucent?: TranslucentMeshPart;
}

export interface SectionBuildOutput extends BuildOutputData {
  approxBytes: number;
  /** Release all transferred buffers. Call on stale/cancelled/failed/after-upload. */
  dispose(): void;
}

/** Flatten a build output into a `BuildPayload` + the list of ArrayBuffers to transfer. */
export function serializeBuildOutput(out: SectionBuildOutput): { payload: BuildPayload; transfer: ArrayBuffer[] } {
  const parts: Partial<Record<TerrainPass, MeshPart>> = {};
  const transfer: ArrayBuffer[] = [];
  for (const key of Object.keys(out.parts)) {
    const pass = Number(key) as TerrainPass;
    const part = out.parts[pass];
    if (!part) continue;
    // FS-2: carry the per-facing counts (structured-cloned Uint32Array — small, NOT in the transfer list, so
    // it survives the postMessage; transferring its buffer would detach it on the main side).
    parts[pass] = { vertexData: part.vertexData, quadCount: part.quadCount, facingVertexCounts: part.facingVertexCounts };
    transfer.push(part.vertexData);
  }
  let translucent: TranslucentMeshPart | undefined;
  if (out.translucent) {
    const t = out.translucent;
    // Carry the per-quad TQuads for DYNAMIC sections so the worker path supports camera-driven INDEX
    // re-sorting exactly like inline (TRAP 12.A — re-sort rewrites indices only). They're small CPU
    // objects (structured-cloned, NOT in the transfer list); static sort types bake a fixed index and
    // never re-sort, so they ship none. Without this, worker-meshed Dynamic glass would freeze its index
    // and mispaint as the camera moves — a gap the static-camera smoke can't catch.
    translucent = {
      vertexData: t.vertexData,
      quadCount: t.quadCount,
      sortType: t.sortType,
      quadHash: t.quadHash,
      indexData: t.indexData,
      quads: t.sortType === SortType.Dynamic ? t.quads : undefined,
    };
    transfer.push(t.vertexData);
    if (t.indexData) transfer.push(t.indexData);
  }
  return { payload: { sectionKey: out.sectionKey, generation: out.generation, info: out.info, parts, translucent }, transfer };
}

/** Fields a mesher fills in; the factory adds `approxBytes` + a `dispose()` that drops buffers.
 *  (P3: the former `pickBoxes` field was dead — never serialized across the worker boundary and picking is
 *  unwired — so it was dropped; re-add it to `BuildPayload` + `serializeBuildOutput`'s transfer list if/when
 *  CPU picking is wired, so the inline and off-thread mesh paths stay byte-identical.) */
export type BuildOutputData = BuildPayload;

/**
 * Wrap raw mesh data in a single-owner `SectionBuildOutput`. `dispose()` drops references to the
 * (transferable) buffers so they can be GC'd / their backing released; it is idempotent and MUST
 * be called on stale/cancelled/after-upload outputs (§22: "stale outputs release transferred
 * buffers"). Sums `approxBytes` from the geometry so the upload budget can account for it.
 */
export function makeBuildOutput(data: BuildOutputData): SectionBuildOutput {
  let approxBytes = 0;
  for (const part of Object.values(data.parts)) approxBytes += part?.vertexData.byteLength ?? 0;
  if (data.translucent) {
    const t = data.translucent;
    approxBytes += t.vertexData.byteLength + (t.indexData?.byteLength ?? 0);
  }

  const out: SectionBuildOutput = {
    ...data,
    approxBytes,
    dispose() {
      // Drop strong refs; for real transferables the ArrayBuffers are already detached post-post.
      out.parts = {};
      out.translucent = undefined;
    },
  };
  return out;
}
