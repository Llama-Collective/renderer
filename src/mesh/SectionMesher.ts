// The hot loop: SectionSnapshot → SectionBuildOutput, baking real block models. RENDERER_PLAN §6, §24.
//
// Per non-air cell: get its BakedBlockModel, cull each quad by its cullface against the neighbor's
// occlusion (+ skip-render groups), and pack surviving quads into per-render-pass buckets (SOLID /
// CUTOUT / TRANSLUCENT). Pure data (no GPU): models are baked by an injected provider, so this can
// run in a worker. (Phase-1's full-cube path is replaced by this model path.)
//
// SOLID/CUTOUT go into `parts` (greedy-merge-eligible later). TRANSLUCENT is emitted SEPARATELY into
// `out.translucent`: one quad per face (never merged — TRAP 6.D), with per-quad centroid+normal
// metadata so the section can be sorted back-to-front, classified into the cheapest correct sort
// (§12), and carry its own index order (TRAP 12.A — sorting is index-only).

import { Direction, DIRECTION_OFFSET, TerrainPass } from "../types";
import { snapshotIndex, snapshotStride, type SectionSnapshot } from "../world/SnapshotSource";
import { AO_NONE, FULLBRIGHT_LIGHT, packLight, type PackedLight } from "./Lighting";
import { NO_SHADE } from "./VertexFormat";
import { shouldDrawQuad, isFullOpaqueCube, occludesAo } from "./model/OcclusionShape";
import { computeSectionVisibility, computeSectionVisibilitySets } from "../world/SectionVisibility";
import type { BakedBlockModel } from "./model/BakedBlockModel";
import { PackedVertexSink, FacingPartitionedSink, quadIndices } from "./VertexFormat";
import { quadFacing } from "./ModelQuadFacing";
import { smoothFaceLight, isAlignedFullFace } from "./SmoothLight";
import { makeBuildOutput, type MeshPart, type SectionBuildOutput, type TranslucentMeshPart } from "../workers/BuildOutput";
import { TranslucentCollector, sortIndicesStaticNormal } from "./TranslucentCollector";
import { expandQuadOrder, topoSortOrder } from "./TranslucentTopoSort";
import { SortType } from "./SortTypes";
import { meshFluidCell, type FluidAppearance, type FluidContext, type FluidSampler, type FluidSink } from "./FluidMesher";

/** Resolves a snapshot cell value (block id) to its baked model (cached upstream). */
export interface BakedModelProvider {
  /** Baked model for a block id, or null for air / nothing to render. */
  get(blockId: number): BakedBlockModel | null;
}

/** Optional meshing toggles (default-off; absent ⇒ the exact byte-identical baseline). */
export interface MeshOptions {
  /** FS-2: partition opaque/cutout geometry into 7 facing runs so the draw layer can cull camera-away
   *  slices (FS-4). Reorders the opaque bytes (whole quads, winding preserved) — NOT byte-identical, but
   *  pixel-identical (the dropped slices are GPU back-face-culled). Off ⇒ the single undifferentiated run. */
  partitionFacing?: boolean;
  /** SL-2/SL-4: emit per-vertex smooth lighting + ambient occlusion for aligned full faces (
   *  `AoFaceData`). Changes baked vertex bytes ⇒ a remesh-required visual feature. Off ⇒ flat per-face
   *  light (byte-identical baseline). REQUIRES the snapshot to carry apron 2 (SL-3) for corner reads. */
  smoothLighting?: boolean;
  /** OCC-2: emit the per-perspective DIRECTION_SETS (`computeSectionVisibilitySets`) into `visibilityData`
   *  as a 4-element `Uint32Array` instead of the 1-element symmetric word. `visibilityData[0]` stays the
   *  symmetric union (so every off-path reader is byte-identical) and it lives OUTSIDE the FNV vertex hash, so
   *  geometry quadHash is PROVABLY unchanged on/off. Off ⇒ the 1-element baseline (byte-identical). The camera
   *  BFS joins the sets per-frame (`OcclusionCuller`) only when this flag is on AND the store carries the sets. */
  perPerspectiveVisibility?: boolean;
  /** OCC-2 sub-flag (depends on `perPerspectiveVisibility`): also AND the angle/slope cone into the BFS
   *  propagation. Mesh-side it changes NOTHING (the cone is a pure BFS-time refinement) — declared here so the
   *  viewer can thread both knobs through one `MeshOptions`. Off ⇒ DIRECTION_SETS only. */
  occlusionAngleMask?: boolean;
}

// P6/LM-1: per-cell propagated light, nibble-packed (Minecraft convention): `(sky << 4) | block`, each
// 0..15. `snapshot.light` is OPTIONAL — the editor has no light engine yet, so it's usually absent and
// every face bakes full-bright (sky 15 → white LUT ⇒ render unchanged). When a world DOES supply light,
// the renderer CONSUMES it (vanilla-style "flat" per-face lighting); it never PROPAGATES it (that's the
// sim's job, exactly as reads light from the level rather than computing it).
const SKY_SHIFT = 4;
const LEVEL_MASK = 0xf;

/**
 * Flat per-face light word for a face at cell (lx,ly,lz) facing `normal`. Vanilla non-smooth lighting:
 * a face takes the light of the neighbour cell it looks INTO (normal ∈ 0..5 → that neighbour; NO_SHADE →
 * the cell itself). All four of a quad's vertices share it (v1 flat); the per-vertex encoding leaves room
 * for smooth interpolation later. Returns FULLBRIGHT when no light array is supplied.
 */
function faceLight(light: Uint8Array | undefined, lx: number, ly: number, lz: number, normal: number, apron: number): PackedLight {
  if (!light) return FULLBRIGHT_LIGHT;
  let sx = lx, sy = ly, sz = lz;
  if (normal !== NO_SHADE && DIRECTION_OFFSET[normal as Direction]) {
    const off = DIRECTION_OFFSET[normal as Direction];
    sx += off[0]; sy += off[1]; sz += off[2];
  }
  // Clamp into the apron cube so a boundary face can never read off the sampled region.
  const lo = -apron, hi = 15 + apron;
  if (sx < lo) sx = lo; else if (sx > hi) sx = hi;
  if (sy < lo) sy = lo; else if (sy > hi) sy = hi;
  if (sz < lo) sz = lo; else if (sz > hi) sz = hi;
  const packed = light[snapshotIndex(sx, sy, sz, apron)];
  return packLight(packed & LEVEL_MASK, (packed >> SKY_SHIFT) & LEVEL_MASK, AO_NONE, false);
}

/** WRK-3: distinct sentinel returned by `meshSection` when a cooperative-cancel poll fired mid-mesh. The
 *  worker uses it to distinguish a real (empty) mesh from a cancelled one — on cancel it posts `cancelled`
 *  and returns WITHOUT serializing/transferring. `null` (not an empty SectionBuildOutput) so it's unambiguous. */
export const MESH_CANCELLED = null;

/**
 * Mesh a section snapshot into a `SectionBuildOutput`.
 *
 * `isCancelled` (WRK-3, optional) is polled ONCE PER Y-LAYER (≤16 `Atomics.load`/section — 
 * granularity, never per-quad): when it returns true the mesh bails and returns `MESH_CANCELLED` (null). It is
 * ONLY passed by the off-thread Model-A worker for a superseded build; the inline Scheduler call passes
 * `undefined`, so the inline path takes the exact byte-identical loop (the poll is `undefined?.()` ⇒ skipped).
 *
 * Overloaded so callers that omit `isCancelled` (inline Scheduler + Model-B drain) keep the non-null return
 * type unchanged — only the cancellable call site has to handle `MESH_CANCELLED`.
 */
export function meshSection(snapshot: SectionSnapshot, provider: BakedModelProvider, fluids: FluidContext | undefined, opts: MeshOptions | undefined, isCancelled: () => boolean): SectionBuildOutput | typeof MESH_CANCELLED;
export function meshSection(snapshot: SectionSnapshot, provider: BakedModelProvider, fluids?: FluidContext, opts?: MeshOptions): SectionBuildOutput;
export function meshSection(snapshot: SectionSnapshot, provider: BakedModelProvider, fluids?: FluidContext, opts?: MeshOptions, isCancelled?: () => boolean): SectionBuildOutput | typeof MESH_CANCELLED {
  const { blocks, apron, light } = snapshot;
  // FS-2: when partitioning, opaque/cutout vertices go into 7 facing runs; otherwise the plain single-run
  // sinks (byte-identical baseline). Exactly one of each pair is live, chosen by the flag.
  const partition = opts?.partitionFacing ?? false;
  // AUDIT #1: encode straight into growable buffers in the cull loop — no `RawVertex[]`, no 2nd pass.
  const solid = partition ? null : new PackedVertexSink();
  const cutout = partition ? null : new PackedVertexSink();
  const solidF = partition ? new FacingPartitionedSink() : null;
  const cutoutF = partition ? new FacingPartitionedSink() : null;
  const tSink = new PackedVertexSink();
  const collector = new TranslucentCollector();
  // OCC-1: full-opaque-cube grid for the section visibility graph, captured in the cull loop below
  // (near-zero cost — one predicate per non-air cell). 1 ⇒ the cell fully blocks sight on all faces.
  const opacity = new Uint8Array(16 * 16 * 16);

  // SL-1: smooth-lighting source grids over the FULL apron cube (needs apron 2 — SL-3) so a boundary face's
  // diagonal corner reads stay in-bounds. shadeGrid = per-cell directional-shade multiplier (air 1.0);
  // aoFlags = per-cell "occludes AO" (occludesAny, so slabs/stairs darken corners). Built ONLY when smooth
  // lighting is on (zero cost off); read by the per-corner AO emission (SL-2). Nothing reads them at flat.
  const smoothLighting = opts?.smoothLighting ?? false;
  const aoCells = smoothLighting ? snapshotStride(apron) ** 3 : 0;
  const shadeGrid = smoothLighting ? new Float32Array(aoCells).fill(1) : null;
  const aoFlags = smoothLighting ? new Uint8Array(aoCells) : null;
  if (smoothLighting) {
    for (let i = 0; i < aoCells; i++) {
      const id = blocks[i];
      if (id === 0) continue;
      const m = provider.get(id);
      if (!m) continue;
      if (m.shadeBrightness !== undefined) shadeGrid![i] = m.shadeBrightness;
      if (occludesAo(m.occlusion)) aoFlags![i] = 1;
    }
  }

  // Push a translucent quad (section-local positions) + record its sort metadata.
  const emitTranslucent = (positions: readonly [number, number, number][], uv: readonly [number, number][], normal: number, color: number, lightWord: PackedLight): void => {
    for (let v = 0; v < 4; v++) tSink.push(positions[v][0], positions[v][1], positions[v][2], uv[v][0], uv[v][1], normal, color, 0, lightWord);
    collector.add(positions[0], positions[1], positions[2], positions[3]);
  };

  const sampler: FluidSampler | null = fluids
    ? {
        idAt: (x, y, z) => blocks[snapshotIndex(x, y, z, apron)],
        fluidOf: fluids.fluidOf,
        // Cull fluid faces against full opaque occluders (DefaultFluidRenderer).
        occludesFace: (x, y, z, dir) => provider.get(blocks[snapshotIndex(x, y, z, apron)])?.occlusion.byFace[dir].full ?? false,
      }
    : null;

  // A2: ONE reused fluid sink (defined once — not a per-cell closure) reading mutable cell coords/appearance
  // set right before each meshFluidCell call; routes each emitted face to the right bucket with its light.
  let fLx = 0, fLy = 0, fLz = 0;
  let fApp: FluidAppearance | null = null;
  const fluidSink: FluidSink = (positions, atlasUV, normal, colorRGBA) => {
    const lightWord = faceLight(light, fLx, fLy, fLz, normal, apron);
    if (fApp!.translucent) { emitTranslucent(positions, atlasUV, normal, colorRGBA, lightWord); return; }
    // Lava (opaque) → the SOLID pass; bucket by facing when partitioning (FS-2).
    if (partition) {
      const facing = quadFacing(positions[0], positions[1], positions[2], positions[3]);
      for (let v = 0; v < 4; v++) solidF!.pushFacing(facing, positions[v][0], positions[v][1], positions[v][2], atlasUV[v][0], atlasUV[v][1], normal, colorRGBA, 0, lightWord);
    } else {
      for (let v = 0; v < 4; v++) solid!.push(positions[v][0], positions[v][1], positions[v][2], atlasUV[v][0], atlasUV[v][1], normal, colorRGBA, 0, lightWord);
    }
  };

  for (let ly = 0; ly < 16; ly++) {
    // WRK-3: cooperative-cancel poll, ONCE PER Y-LAYER (≤16 Atomics.load/section — never per-quad, TRAP
    // WRK-3.D). `isCancelled` is undefined on the inline path ⇒ this is a `undefined?.()` no-op ⇒ byte-
    // identical. A long superseded mesh bails within ~1/16th of its remaining work, emitting NO geometry.
    if (isCancelled?.()) return MESH_CANCELLED;
    for (let lz = 0; lz < 16; lz++) {
      for (let lx = 0; lx < 16; lx++) {
        const id = blocks[snapshotIndex(lx, ly, lz, apron)];

        // Fluid surface — SEPARATE from + additive to the block model (a waterlogged block emits
        // both its model and a water surface). Water → translucent; lava → solid.
        if (fluids && sampler) {
          const fl = fluids.fluidOf(id);
          if (fl) {
            fLx = lx; fLy = ly; fLz = lz; fApp = fluids.appearance(fl.type);
            meshFluidCell(fl, lx, ly, lz, fApp, sampler, fluidSink);
          }
        }

        const self = provider.get(id);
        // OCC-1: record solid-cube occluders before the empty-model skip (drives the visibility flood).
        if (self && isFullOpaqueCube(self.occlusion)) opacity[(ly * 16 + lz) * 16 + lx] = 1;
        if (!self || self.quads.length === 0) continue;

        for (const quad of self.quads) {
          if (quad.cullface !== null) {
            const off = DIRECTION_OFFSET[quad.cullface];
            const neighbor = provider.get(blocks[snapshotIndex(lx + off[0], ly + off[1], lz + off[2], apron)]);
            if (!shouldDrawQuad(quad, neighbor?.occlusion ?? null, self.skipGroup, neighbor?.skipGroup ?? null)) continue;
          }

          const lightWord = faceLight(light, lx, ly, lz, quad.normal, apron);
          if (quad.layer === TerrainPass.Translucent) {
            // A8: build the 4 retained corner tuples as locals (collector.add keeps them) — no `corners`
            // wrapper array per translucent quad.
            const qp = quad.positions, qu = quad.atlasUV;
            const c0: [number, number, number] = [lx + qp[0][0], ly + qp[0][1], lz + qp[0][2]];
            const c1: [number, number, number] = [lx + qp[1][0], ly + qp[1][1], lz + qp[1][2]];
            const c2: [number, number, number] = [lx + qp[2][0], ly + qp[2][1], lz + qp[2][2]];
            const c3: [number, number, number] = [lx + qp[3][0], ly + qp[3][1], lz + qp[3][2]];
            tSink.push(c0[0], c0[1], c0[2], qu[0][0], qu[0][1], quad.normal, quad.colorRGBA, quad.material, lightWord);
            tSink.push(c1[0], c1[1], c1[2], qu[1][0], qu[1][1], quad.normal, quad.colorRGBA, quad.material, lightWord);
            tSink.push(c2[0], c2[1], c2[2], qu[2][0], qu[2][1], quad.normal, quad.colorRGBA, quad.material, lightWord);
            tSink.push(c3[0], c3[1], c3[2], qu[3][0], qu[3][1], quad.normal, quad.colorRGBA, quad.material, lightWord);
            collector.add(c0, c1, c2, c3);
          } else {
            // SL-2: per-corner SMOOTH light for an aligned full face of a shaded, non-emissive block; else
            // the shared flat `lightWord` (byte-identical baseline). NOTE (documented simplification, NOT full
            // parity): emissive blocks bake flat full-bright with NO per-corner AO — keeps AO on
            // emissive blocks and applies the emissive lightmap per-corner after the min-non-zero normalization.
            // Porting that is a follow-up; translucent/fluid also stay flat for now. `cornerLights[v]` lines up
            // with vertex v.
            const cornerLights = (smoothLighting && self.emissive !== true && quad.normal === quad.cullface && isAlignedFullFace(quad.cullface, quad.positions))
              ? smoothFaceLight(quad.cullface!, lx, ly, lz, light, aoFlags!, apron, shadeGrid![snapshotIndex(lx, ly, lz, apron)])
              : null;
            if (partition) {
              // FS-2: bucket this opaque/cutout quad into its facing run (geometric normal, translation-
              // invariant ⇒ computed from the untranslated model corners). All 4 vertices share one facing.
              const fb = quad.layer === TerrainPass.Cutout ? cutoutF! : solidF!;
              const facing = quadFacing(quad.positions[0], quad.positions[1], quad.positions[2], quad.positions[3]);
              for (let v = 0; v < 4; v++) {
                const p = quad.positions[v];
                const uv = quad.atlasUV[v];
                fb.pushFacing(facing, lx + p[0], ly + p[1], lz + p[2], uv[0], uv[1], quad.normal, quad.colorRGBA, quad.material, cornerLights ? cornerLights[v] : lightWord);
              }
            } else {
              const bucket = quad.layer === TerrainPass.Cutout ? cutout! : solid!;
              for (let v = 0; v < 4; v++) {
                const p = quad.positions[v];
                const uv = quad.atlasUV[v];
                bucket.push(lx + p[0], ly + p[1], lz + p[2], uv[0], uv[1], quad.normal, quad.colorRGBA, quad.material, cornerLights ? cornerLights[v] : lightWord);
              }
            }
          }
        }
      }
    }
  }

  const parts: Partial<Record<TerrainPass, MeshPart>> = {};
  if (partition) {
    // FS-2: carry the per-facing vertex counts so the draw layer can cull camera-away slices (FS-4).
    if (!solidF!.isEmpty()) parts[TerrainPass.Solid] = { vertexData: solidF!.finish(), quadCount: solidF!.quadCount, facingVertexCounts: solidF!.facingVertexCounts() };
    if (!cutoutF!.isEmpty()) parts[TerrainPass.Cutout] = { vertexData: cutoutF!.finish(), quadCount: cutoutF!.quadCount, facingVertexCounts: cutoutF!.facingVertexCounts() };
  } else {
    if (!solid!.isEmpty()) parts[TerrainPass.Solid] = { vertexData: solid!.finish(), quadCount: solid!.quadCount };
    if (!cutout!.isEmpty()) parts[TerrainPass.Cutout] = { vertexData: cutout!.finish(), quadCount: cutout!.quadCount };
  }

  const translucent = buildTranslucent(tSink, collector);
  // OCC-1: encode the section's face-to-face visibility (DirectionalVisGraph). Produced now,
  // consumed by the camera occlusion BFS — see world/OcclusionCuller. OCC-2 (perPerspectiveVisibility,
  // default-off): emit the 4-element DIRECTION_SETS array instead — `[0]` is the SAME symmetric union word, so
  // every off-path reader (`visibilityData[0] ?? VIS_ALL`) is byte-identical and the geometry quadHash is
  // unaffected (visibilityData is outside the FNV vertex hash). Off ⇒ the 1-element baseline.
  const isOpaqueCell = (x: number, y: number, z: number) => opacity[(y * 16 + z) * 16 + x] === 1;
  const visibilityData = opts?.perPerspectiveVisibility
    ? computeSectionVisibilitySets(isOpaqueCell)
    : Uint32Array.of(computeSectionVisibility(isOpaqueCell));
  return makeBuildOutput({
    sectionKey: snapshot.sectionKey,
    generation: snapshot.generation,
    info: { flags: 0, visibilityData },
    parts,
    translucent,
  });
}

/** Build the translucent mesh part: classify (heuristic) + bake the per-sort-type index. */
function buildTranslucent(tSink: PackedVertexSink, collector: TranslucentCollector): TranslucentMeshPart | undefined {
  if (tSink.isEmpty()) return undefined;
  const vertexData = tSink.finish();
  const { sortType: classified, quads } = collector.classify();
  const quadCount = tSink.quadCount;
  let sortType = classified;
  const part: TranslucentMeshPart = { vertexData, quadCount, sortType, quadHash: fnv1a(vertexData) };

  // AnyOrder → shared quad EBO (no index). StaticNormal → one-time normal-relative order. StaticTopo
  // → one-time topological order (valid for all views; on a cycle, downgrade to Dynamic). Dynamic →
  // identity order + carry the TQuads so the renderer re-runs the topo sort on a plane-crossing
  // trigger — vertices are never re-meshed (TRAP 12.A).
  if (sortType === SortType.StaticNormal) {
    part.indexData = sortIndicesStaticNormal(quads).buffer as ArrayBuffer;
  } else if (sortType === SortType.StaticTopo) {
    const order = topoSortOrder(quads, null, true);
    if (order) part.indexData = expandQuadOrder(order).buffer as ArrayBuffer;
    else sortType = SortType.Dynamic;
  }
  if (sortType === SortType.Dynamic) {
    part.indexData = quadIndices(quadCount).buffer as ArrayBuffer;
    part.quads = quads;
  }
  part.sortType = sortType;
  return part;
}

/** 32-bit FNV-1a over the vertex bytes — geometry-identity for reuse detection (TRAP 12.D). */
function fnv1a(buf: ArrayBuffer): number {
  const b = new Uint8Array(buf);
  let h = 0x811c9dc5;
  for (let i = 0; i < b.length; i++) {
    h ^= b[i];
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}
