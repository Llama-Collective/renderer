// Translucent sort classification. RENDERER_PLAN.md §12.
//
// Names mirror SortType enum
//   NoTranslucent  <- NO_TRANSLUCENT (+ EMPTY_SECTION folded in)
//   AnyOrder       <- NONE   (do NOT grep for "ANY_ORDER")
//   StaticNormal   <- STATIC_NORMAL_RELATIVE
//   StaticTopo     <- STATIC_TOPO   (one-time topological sort, valid for all views)
//   Dynamic        <- DYNAMIC       (re-sorted on a GFNI plane-crossing trigger)
//
// `directionMixing` mirrors `needsDirectionMixing` flag. It controls VERTEX BUCKETING, not
// cull state: when true (StaticTopo/Dynamic) forces a section's translucent quads into one
// vertex bucket (`forceUnassigned`) so a single GLOBAL back-to-front index can interleave quads of
// every facing. It does NOT disable back-face culling — culls all terrain unconditionally
// (`ShaderChunkRenderer` builds every pipeline `.withCull(true)`); GPU cull is per-triangle by winding
// and independent of index order. Invariant: a sort type never both mixes directions and is statically
// orderable. (We emit translucent quads in one stream already, so this flag is currently informational.)

export enum SortType {
  /** No translucent geometry in this section. */
  NoTranslucent = 0,
  /** Translucent quads present but order never matters — shared quad EBO. */
  AnyOrder = 1,
  /** Sort once by normal-relative distance; static index. */
  StaticNormal = 2,
  /** Sort once by topological order (valid for every view); static index. */
  StaticTopo = 3,
  /** Re-sort the INDEX order on a plane-crossing trigger (never remesh — TRAP 12.A). */
  Dynamic = 4,
}

/**
 * `needsDirectionMixing`: true ⇒ the section's translucent quads must share one global sorted
 * index (merges them into the UNASSIGNED vertex bucket). This is about vertex layout / sort
 * scope, NOT cull state — back-face culling stays on for every sort type. Currently informational
 * (our mesher already emits translucent quads in a single stream).
 */
export function needsDirectionMixing(t: SortType): boolean {
  return t === SortType.StaticTopo || t === SortType.Dynamic;
}
