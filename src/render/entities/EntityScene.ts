// Entity scene state + tick interpolation. RENDERER_PLAN.md §18, Phase 4.5a.
//
// The renderer consumes the FLAT sim diff (`EntitySnapshot` = `EditableEntityLike`: id / type /
// position / velocity? / properties?(strings)), NOT live SimEntity objects. The diff has no
// `xo/yo/zo` previous-tick position, so — exactly as vanilla interpolates `lerp(xo, x, partialTick)` —
// this class RETAINS the previous position per id and interpolates toward the latest on each frame.
//
// Lifecycle per id:
//   - first seen        → prev = pos (no interpolation jump on spawn).
//   - position changed  → prev = the last committed pos, pos = the new pos (one tick of travel).
//   - unchanged a tick  → prev advances to pos (settles at rest). Vanilla does `xo = x` EVERY tick for
//                         EVERY entity, so an entity that stops moving — and thus drops out of the
//                         sparse `changed` diff — must settle, not keep interpolating behind itself.
//   - removed           → dropped entirely.
// `ingest()` is the per-tick boundary; `frame(partialTick)` is called every render frame with
// partialTick climbing 0→1 between ticks (reuse the piston partial-tick accumulator).
//
// Pure data, GPU-free, unit-tested. The mesh/draw side (EntityRenderer) consumes `frame()`'s output.

export type Vec3 = readonly [number, number, number];

/** The renderer-local mirror of the sim's `EditableEntityLike` (kept decoupled from `simulation/`). */
export interface EntitySnapshot {
  id: string;
  /** Namespaced type, e.g. "minecraft:falling_block". */
  type: string;
  position: Vec3;
  velocity?: Vec3;
  /** All values are strings (the sim serializes properties as `Record<string,string>`). */
  properties?: Readonly<Record<string, string>>;
}

/** An entity resolved for one render frame: position interpolated, metadata carried through. */
export interface RenderEntity {
  id: string;
  type: string;
  /** Interpolated render position (lerp(prev, pos, partialTick)). */
  position: Vec3;
  /** Latest tick velocity (drives velocity-derived walk/idle anim — TRAP 18.C). */
  velocity: Vec3;
  properties: Readonly<Record<string, string>>;
}

interface Tracked {
  type: string;
  prev: Vec3;
  pos: Vec3;
  velocity: Vec3;
  properties: Readonly<Record<string, string>>;
}

const ZERO: Vec3 = [0, 0, 0];
const NO_PROPS: Readonly<Record<string, string>> = Object.freeze({});

/** A pooled `RenderEntity` whose `position` array is mutated in place each frame (AUDIT F8/ENT-4). */
interface MutableRenderEntity {
  id: string;
  type: string;
  position: [number, number, number];
  velocity: Vec3;
  properties: Readonly<Record<string, string>>;
}

export class EntityScene {
  private readonly entities = new Map<string, Tracked>();
  // Reused output (AUDIT F8/ENT-4): a static/animated entity scene allocates nothing per frame here —
  // the result array, the per-entity objects, and their position arrays are all reused in place.
  private readonly pool: MutableRenderEntity[] = [];
  private readonly out: RenderEntity[] = [];

  /** Number of tracked entities (test/instrumentation). */
  get size(): number {
    return this.entities.size;
  }

  /**
   * Apply one sim tick's diff: `changed` carries every entity whose serialized state differs this
   * tick (new or moved); `removedIds` are despawns. Returns nothing — call `frame()` to read.
   */
  ingest(changed: readonly EntitySnapshot[], removedIds: readonly string[] = []): void {
    for (const id of removedIds) this.entities.delete(id);
    // Advance prev = pos for EVERY survivor first (vanilla's per-tick `xo = x`). An entity that stops
    // moving is omitted from `changed`, so without this it would keep lerping from its stale prev; the
    // moved entities below then overwrite prev with their pre-move pos (read from `existing.pos`).
    for (const e of this.entities.values()) e.prev = e.pos;
    for (const s of changed) {
      const pos = s.position;
      const existing = this.entities.get(s.id);
      this.entities.set(s.id, {
        type: s.type,
        // First sighting starts settled (prev = pos); a moved entity travels from where it was.
        prev: existing ? existing.pos : pos,
        pos,
        velocity: s.velocity ?? ZERO,
        properties: s.properties ?? NO_PROPS,
      });
    }
  }

  /** Drop everything (scene reset / reload). */
  clear(): void {
    this.entities.clear();
  }

  /**
   * Resolve every entity for this frame. `partialTick` in [0,1] interpolates prev→pos; values outside
   * are clamped (a long frame must never overshoot past the target tick position).
   */
  frame(partialTick: number): RenderEntity[] {
    const t = partialTick < 0 ? 0 : partialTick > 1 ? 1 : partialTick;
    const out = this.out;
    out.length = 0;
    let i = 0;
    for (const [id, e] of this.entities) {
      let slot = this.pool[i];
      if (!slot) {
        slot = { id, type: e.type, position: [0, 0, 0], velocity: ZERO, properties: NO_PROPS };
        this.pool[i] = slot;
      }
      slot.id = id;
      slot.type = e.type;
      const p = slot.position;
      p[0] = e.prev[0] + (e.pos[0] - e.prev[0]) * t;
      p[1] = e.prev[1] + (e.pos[1] - e.prev[1]) * t;
      p[2] = e.prev[2] + (e.pos[2] - e.prev[2]) * t;
      slot.velocity = e.velocity;
      slot.properties = e.properties;
      out.push(slot);
      i++;
    }
    return out;
  }
}
