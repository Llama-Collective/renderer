// Builds immutable SectionSnapshots for worker meshing + a short-lived source cache.
// RENDERER_PLAN.md §6.
//
// TRAP 6.A: apron is 1 block for face-culling-only v1; widen to 2 only when smooth AO /
// biome-blend sampling lands. APRON is the default; the apron is a SnapshotSource constructor
// parameter, so smooth lighting (SL-3) constructs its source with apron 2 (corner reads for boundary
// faces stay in-bounds) while the flat default stays apron 1 — zero extra snapshot bytes when off.
// The flat geometry is FNV-identical at either apron (the apron only widens sampling REACH, not the
// emitted quads — proven in SnapshotSource.apron.test).
// TRAP 13.B (DECIDED 2026-06-18): fluid corner/flow sampling does NOT need a wider apron — every
// neighbor it reads (orthogonals, the diagonal (x±1,z±1), the cell above, the below-diagonal
// (x±1,y−1,z±1)) is Chebyshev distance ≤1, all inside this 1-block apron. So APRON stays 1.
// TRAP 6.B: the SOURCE cache and the PRESENTED-mesh cache are different things. Invalidating
// source data on an edit must NEVER touch presented geometry.

import type { Generation, Vec3i, DirtyReason } from "../types";
import { parseSectionKey, sectionKey, type SectionKey } from "./SectionKey";

/** Apron radius in blocks copied around the 16³ section. See TRAP 6.A. */
export const APRON = 1;

/** Section coords whose [core−apron, core+15+apron] cube COVERS world `coord` — the core section, plus a
 *  neighbour when `coord` sits within `apron` of a face (so an edit patches every snapshot that samples it).
 *  `& 15` gives the correct local index for negative coords too. */
function coveringSections(coord: number, apron: number): number[] {
  const c = coord >> 4;
  const local = coord & 15;
  const out = [c];
  if (local < apron) out.push(c - 1);
  if (local >= 16 - apron) out.push(c + 1);
  return out;
}

/** Read-only world block grid the snapshotter samples. World (not section-local) coords. */
export interface BlockSource {
  /** Packed block/state id at a world coordinate; 0 = air. Out-of-bounds reads return air. */
  getBlock(x: number, y: number, z: number): number;
}

/** Optional propagated-light grid (P6/LM-1). Returns a nibble-packed level `(sky << 4) | block`, each 0..15.
 *  The renderer CONSUMES this (flat per-face) but never PROPAGATES it — light comes from the world/sim, like
 *  reading `getLightLevel`. Absent ⇒ snapshots carry no light ⇒ the mesher bakes full-bright. */
export interface LightSource {
  getLight(x: number, y: number, z: number): number;
}

/** Side length of the sampled cube including the apron on both sides. */
export function snapshotStride(apron: number): number {
  return 16 + 2 * apron;
}

/**
 * Flat index into `SectionSnapshot.blocks` for SECTION-LOCAL coords `lx,ly,lz ∈ [-apron, 15+apron]`.
 * x is fastest, then z, then y. The mesher and the snapshotter MUST agree on this.
 */
export function snapshotIndex(lx: number, ly: number, lz: number, apron: number): number {
  const s = snapshotStride(apron);
  return lx + apron + s * (lz + apron + s * (ly + apron));
}

/** Immutable, transferable input to worker meshing. RENDERER_PLAN §6. */
export interface SectionSnapshot {
  sectionKey: SectionKey;
  generation: Generation;
  origin: Vec3i;
  size: 16;
  apron: number;
  /** Packed block/state ids over (16 + 2*apron)³. */
  blocks: Uint32Array;
  light?: Uint8Array;
  biomeTint?: Uint32Array;
  changedReason: DirtyReason;
}

/**
 * Is the section's 16³ CORE all air? (AUDIT #15 — `LevelSlice.prepare` early-reject.) Apron
 * blocks belong to neighbor sections and never emit geometry for THIS section, so an air core ⇒ nothing
 * to mesh (no block models, and a fluid is a non-air block id, so no fluid surface either). The dirty
 * path uses this to skip dispatching a mesh job for an empty section (e.g. a border edit dirtying an
 * all-air neighbor) when nothing is currently presented there.
 */
export function isAllAirCore(snapshot: SectionSnapshot): boolean {
  const { blocks, apron } = snapshot;
  for (let ly = 0; ly < 16; ly++)
    for (let lz = 0; lz < 16; lz++)
      for (let lx = 0; lx < 16; lx++) if (blocks[snapshotIndex(lx, ly, lz, apron)] !== 0) return false;
  return true;
}

export class SnapshotSource {
  /** N3 source cache: the canonical (16+2·apron)³ block cube per section, kept CURRENT incrementally by
   *  `setBlock` (the viewer patches it on every edit). buildSnapshot then returns a copy instead of
   *  re-reading 5832 cells through `source.getBlock` (each of which allocates a section-key string + a Map
   *  lookup). A 1-block edit thus re-clones the edited section once and PATCHES its neighbours' apron cell
   *  (RENDERER_PLAN §16 N3) rather than rebuilding them. Mirrors `source` exactly (TRAP 6.B: never touches
   *  presented geometry). Bounded by edited sections; fresh per scene (the viewer makes a new SnapshotSource). */
  private readonly cache = new Map<SectionKey, Uint32Array>();
  /** P6/LM-1 light-cube cache (1 byte/cell, nibble-packed) — mirrors `cache`, only when a LightSource is
   *  supplied. Patched incrementally by `setLight`; a snapshot carries `light` only when this is populated. */
  private readonly lightCache = new Map<SectionKey, Uint8Array>();

  constructor(
    private readonly source: BlockSource,
    private readonly apron: number = APRON,
    private readonly lightSource?: LightSource,
  ) {}

  /** Read the full section + apron cube out of the live block source (the cache-miss / cold path). */
  private readCube(origin: Vec3i): Uint32Array {
    const apron = this.apron;
    const s = snapshotStride(apron);
    const blocks = new Uint32Array(s * s * s);
    for (let ly = -apron; ly < 16 + apron; ly++) {
      for (let lz = -apron; lz < 16 + apron; lz++) {
        for (let lx = -apron; lx < 16 + apron; lx++) {
          blocks[snapshotIndex(lx, ly, lz, apron)] = this.source.getBlock(origin[0] + lx, origin[1] + ly, origin[2] + lz);
        }
      }
    }
    return blocks;
  }

  /** Read the section + apron light cube (nibble-packed) out of the LightSource — the cold path. */
  private readLightCube(origin: Vec3i): Uint8Array {
    const apron = this.apron;
    const s = snapshotStride(apron);
    const light = new Uint8Array(s * s * s);
    for (let ly = -apron; ly < 16 + apron; ly++) {
      for (let lz = -apron; lz < 16 + apron; lz++) {
        for (let lx = -apron; lx < 16 + apron; lx++) {
          light[snapshotIndex(lx, ly, lz, apron)] = this.lightSource!.getLight(origin[0] + lx, origin[1] + ly, origin[2] + lz) & 0xff;
        }
      }
    }
    return light;
  }

  /** Assemble a snapshot from the (cached, current) source cube — a copy so the consumer/worker owns it
   *  independently of later patches. Cold sections read once + cache; warm ones reuse the patched cube. */
  buildSnapshot(key: SectionKey, generation: Generation, reason: DirtyReason): SectionSnapshot {
    const [sx, sy, sz] = parseSectionKey(key);
    const origin: Vec3i = [sx * 16, sy * 16, sz * 16];
    let cube = this.cache.get(key);
    if (!cube) {
      cube = this.readCube(origin);
      this.cache.set(key, cube);
    }
    let light: Uint8Array | undefined;
    if (this.lightSource) {
      let lcube = this.lightCache.get(key);
      if (!lcube) {
        lcube = this.readLightCube(origin);
        this.lightCache.set(key, lcube);
      }
      light = lcube.slice();
    }
    return { sectionKey: key, generation, origin, size: 16, apron: this.apron, blocks: cube.slice(), light, changedReason: reason };
  }

  /** Patch the source cache for a changed block (N3): update the cell in EVERY cached section whose cube
   *  samples it (the core section + apron neighbours). `id` must match what `source.getBlock` now returns —
   *  the viewer passes `model.getBlock(x,y,z)` after the edit, keeping the cache a faithful source mirror. */
  setBlock(x: number, y: number, z: number, id: number): void {
    const apron = this.apron;
    for (const sx of coveringSections(x, apron)) {
      for (const sy of coveringSections(y, apron)) {
        for (const sz of coveringSections(z, apron)) {
          const cube = this.cache.get(sectionKey(sx, sy, sz));
          if (cube) cube[snapshotIndex(x - sx * 16, y - sy * 16, z - sz * 16, apron)] = id;
        }
      }
    }
  }

  /** P6/LM-2 — patch the cached LIGHT cube for a changed cell (analogue of `setBlock`): update the cell in
   *  every cached section whose cube samples it (core + apron neighbours). `packed` is `(sky << 4) | block`,
   *  matching what `lightSource.getLight` now returns. No-op when no LightSource is configured. */
  setLight(x: number, y: number, z: number, packed: number): void {
    if (!this.lightSource) return;
    const apron = this.apron;
    for (const sx of coveringSections(x, apron)) {
      for (const sy of coveringSections(y, apron)) {
        for (const sz of coveringSections(z, apron)) {
          const cube = this.lightCache.get(sectionKey(sx, sy, sz));
          if (cube) cube[snapshotIndex(x - sx * 16, y - sy * 16, z - sz * 16, apron)] = packed & 0xff;
        }
      }
    }
  }

  /** Drop the cached SOURCE cube for a section (force a cold re-read next build). Must NOT touch presented
   *  meshes (TRAP 6.B). The per-section eviction hook for FUTURE runtime section streaming/unload — when a
   *  section leaves the loaded set, evict its cube so the cache stays bounded (frees a `LevelSlice` once
   *  its chunk unloads). Not yet called at runtime: the editor only disposes sections in bulk at teardown
   *  (handled by `clear()`), never per-section while live, so the cache is bounded by the edited-section count. */
  invalidateSource(key: SectionKey): void {
    this.cache.delete(key);
    this.lightCache.delete(key);
  }

  /** Drop EVERY cached cube (source + light) — deterministic teardown release, so the caches don't linger to
   *  GC after the viewer drops this source on scene swap/dispose. Must NOT touch presented meshes (TRAP 6.B). */
  clear(): void {
    this.cache.clear();
    this.lightCache.clear();
  }
}
