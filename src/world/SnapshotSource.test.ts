// N3 source cache: a patched cube must stay a faithful mirror of the live source (RENDERER_PLAN §16).

import { describe, it, expect } from "vitest";
import { SnapshotSource, snapshotIndex, type BlockSource, type LightSource } from "./SnapshotSource";
import { sectionKey } from "./SectionKey";
import { DirtyReason } from "../types";

function gridSource() {
  const m = new Map<string, number>();
  const source: BlockSource = { getBlock: (x, y, z) => m.get(`${x},${y},${z}`) ?? 0 };
  return { set: (x: number, y: number, z: number, id: number) => m.set(`${x},${y},${z}`, id), source };
}

const A = sectionKey(0, 0, 0);
const B = sectionKey(1, 0, 0);

describe("SnapshotSource N3 cache", () => {
  it("cold buildSnapshot reads the section + apron cube from the source", () => {
    const g = gridSource();
    g.set(0, 0, 0, 5);
    g.set(15, 0, 0, 7);
    g.set(-1, 0, 0, 3); // apron cell (lx = -1)
    const snap = new SnapshotSource(g.source).buildSnapshot(A, 1, DirtyReason.Simulation);
    expect(snap.blocks[snapshotIndex(0, 0, 0, 1)]).toBe(5);
    expect(snap.blocks[snapshotIndex(15, 0, 0, 1)]).toBe(7);
    expect(snap.blocks[snapshotIndex(-1, 0, 0, 1)]).toBe(3);
  });

  it("setBlock patches every cached section that samples the cell (core + apron), matching a cold re-read", () => {
    const g = gridSource();
    const ss = new SnapshotSource(g.source);
    ss.buildSnapshot(A, 1, DirtyReason.Simulation); // warm A
    ss.buildSnapshot(B, 1, DirtyReason.Simulation); // warm B

    // x=16 is B's core (lx=0) AND A's +x apron (lx=16) — the boundary case the patch must cover in BOTH.
    g.set(16, 2, 3, 9);
    ss.setBlock(16, 2, 3, 9);

    const fresh = new SnapshotSource(g.source); // uncached reference
    for (const k of [A, B]) {
      const cached = ss.buildSnapshot(k, 2, DirtyReason.Simulation);
      const cold = fresh.buildSnapshot(k, 2, DirtyReason.Simulation);
      expect(Array.from(cached.blocks)).toEqual(Array.from(cold.blocks)); // bit-identical to a full re-read
    }
  });

  it("a long random edit sequence keeps the patched cube identical to a cold re-read", () => {
    const g = gridSource();
    const ss = new SnapshotSource(g.source);
    ss.buildSnapshot(A, 1, DirtyReason.Simulation);
    ss.buildSnapshot(B, 1, DirtyReason.Simulation);
    // Deterministic pseudo-edits across the A/B boundary region (cover apron + core on both sides).
    let seed = 12345;
    const rnd = (n: number) => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) % n;
    for (let i = 0; i < 200; i++) {
      const x = rnd(34) - 1; // -1..32 spans A core/apron + B core
      const y = rnd(18) - 1;
      const z = rnd(18) - 1;
      const id = rnd(5);
      g.set(x, y, z, id);
      ss.setBlock(x, y, z, id);
    }
    const fresh = new SnapshotSource(g.source);
    for (const k of [A, B]) {
      expect(Array.from(ss.buildSnapshot(k, 2, DirtyReason.Simulation).blocks))
        .toEqual(Array.from(fresh.buildSnapshot(k, 2, DirtyReason.Simulation).blocks));
    }
  });

  it("no LightSource ⇒ snapshots carry no light (mesher falls back to full-bright)", () => {
    const g = gridSource();
    expect(new SnapshotSource(g.source).buildSnapshot(A, 1, DirtyReason.Simulation).light).toBeUndefined();
  });

  it("with a LightSource (P6/LM-1): cold build reads the light cube; setLight patches it like setBlock", () => {
    const g = gridSource();
    const lm = new Map<string, number>();
    const lightSource: LightSource = { getLight: (x, y, z) => lm.get(`${x},${y},${z}`) ?? 0 };
    lm.set("0,0,0", (15 << 4) | 3); // sky 15, block 3
    const ss = new SnapshotSource(g.source, 1, lightSource);
    ss.buildSnapshot(A, 1, DirtyReason.Simulation); // warm A
    ss.buildSnapshot(B, 1, DirtyReason.Simulation); // warm B

    const snap = ss.buildSnapshot(A, 2, DirtyReason.Simulation);
    expect(snap.light).toBeDefined();
    expect(snap.light![snapshotIndex(0, 0, 0, 1)]).toBe((15 << 4) | 3);

    // x=16 is B core (lx=0) AND A +x apron (lx=16): the patch must reach both cached light cubes.
    lm.set("16,2,3", (4 << 4) | 12);
    ss.setLight(16, 2, 3, (4 << 4) | 12);
    const fresh = new SnapshotSource(g.source, 1, lightSource);
    for (const k of [A, B]) {
      expect(Array.from(ss.buildSnapshot(k, 3, DirtyReason.Simulation).light!))
        .toEqual(Array.from(fresh.buildSnapshot(k, 3, DirtyReason.Simulation).light!));
    }
  });

  it("invalidateSource forces a cold re-read (drops the cache)", () => {
    const g = gridSource();
    const ss = new SnapshotSource(g.source);
    ss.buildSnapshot(A, 1, DirtyReason.Simulation); // cache A
    g.set(1, 1, 1, 42); // mutate the source WITHOUT a setBlock patch (untracked change)
    ss.invalidateSource(A);
    expect(ss.buildSnapshot(A, 2, DirtyReason.Simulation).blocks[snapshotIndex(1, 1, 1, 1)]).toBe(42);
  });

  it("clear() drops EVERY cached cube (deterministic teardown release) → all sections re-read cold", () => {
    let reads = 0;
    const g = gridSource();
    const counting: BlockSource = { getBlock: (x, y, z) => { reads++; return g.source.getBlock(x, y, z); } };
    const ss = new SnapshotSource(counting);
    ss.buildSnapshot(A, 1, DirtyReason.Simulation); // cache A
    ss.buildSnapshot(B, 1, DirtyReason.Simulation); // cache B
    const cached = reads; // a warm re-build reads nothing more…
    ss.buildSnapshot(A, 2, DirtyReason.Simulation);
    expect(reads).toBe(cached);
    ss.clear(); // …until clear() evicts both
    ss.buildSnapshot(A, 3, DirtyReason.Simulation);
    ss.buildSnapshot(B, 3, DirtyReason.Simulation);
    expect(reads).toBe(cached * 2); // both A and B were re-read cold (a full extra pass of cube reads)
  });
});
