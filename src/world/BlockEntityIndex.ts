// The spatial block-entity index. RENDERER_PLAN.md §18, Phase 4.5c — "the one real architectural gap".
//
// Block entities don't flow through `SectionSnapshot` (which carries only packed block ids), so the
// renderer collects them in a SEPARATE static structure, KEYED BY SECTION COORD so it doubles as the
// authoritative spatial BE index (BBE re-scans every block each rebuild; this gives O(1) "which BEs are
// in this dirtied section"). BEs don't move, so it's rebuilt on edit, not per frame.
//
// Each entry carries a per-BE `animating` flag (owned by the editor/preview layer). The hybrid bake
// (§18 / BBB_FINDINGS) reads it: an idle BE (`animating == false`) is baked into the section's static
// geometry (culls with the section); an animating one is excluded from the bake and drawn per-frame.
// Setting a block / toggling `animating` returns the affected section key(s) so the renderer re-meshes
// exactly those sections through the existing dirty-section path.

import { sectionOfBlock, type SectionKey } from "./SectionKey";

/** A renderer-side block entity: world position, vanilla BE id, render variant props, animating flag. */
export interface BlockEntityRecord {
  x: number;
  y: number;
  z: number;
  /** Vanilla BE type id, e.g. "minecraft:chest" / "minecraft:sign" / "minecraft:banner". */
  type: string;
  /** Render-relevant props (FACING, dye color, double-chest side, sign text key, …). Strings. */
  props: Readonly<Record<string, string>>;
  /** True → drawn per-frame (animating preview); false → baked into the section's static geometry. */
  animating: boolean;
  /** Container lid OPEN target for an `openable` BE (chest/shulker) — true while a viewer has it open (set
   *  from the sim's open-viewer count). The baker integrates the lid openness toward this; an opening/open/
   *  closing container draws per-frame, a fully-closed one bakes into the static section mesh. */
  open?: boolean;
}

const posKey = (x: number, y: number, z: number): string => `${x},${y},${z}`;

export class BlockEntityIndex {
  /** position-key → record. */
  private readonly byPos = new Map<string, BlockEntityRecord>();
  /** section-key → set of position-keys in that section. */
  private readonly bySection = new Map<SectionKey, Set<string>>();

  get size(): number {
    return this.byPos.size;
  }

  /** Add/replace a BE at its position. Returns the section key it landed in (to mark dirty). */
  set(rec: BlockEntityRecord): SectionKey {
    const pk = posKey(rec.x, rec.y, rec.z);
    const sk = sectionOfBlock(rec.x, rec.y, rec.z);
    // If it moved sections (shouldn't for a BE, but be safe), unlink the old section.
    const prev = this.byPos.get(pk);
    if (prev) {
      const prevSk = sectionOfBlock(prev.x, prev.y, prev.z);
      if (prevSk !== sk) this.bySection.get(prevSk)?.delete(pk);
    }
    this.byPos.set(pk, { ...rec, props: rec.props ?? {} });
    let set = this.bySection.get(sk);
    if (!set) this.bySection.set(sk, (set = new Set()));
    set.add(pk);
    return sk;
  }

  /** Remove the BE at a position. Returns its former section key, or null if none was there. */
  remove(x: number, y: number, z: number): SectionKey | null {
    const pk = posKey(x, y, z);
    if (!this.byPos.delete(pk)) return null;
    const sk = sectionOfBlock(x, y, z);
    const set = this.bySection.get(sk);
    set?.delete(pk);
    if (set && set.size === 0) this.bySection.delete(sk);
    return sk;
  }

  get(x: number, y: number, z: number): BlockEntityRecord | undefined {
    return this.byPos.get(posKey(x, y, z));
  }

  /** All BEs in a section (empty array if none). Order is insertion-stable per section. */
  inSection(key: SectionKey): BlockEntityRecord[] {
    const set = this.bySection.get(key);
    if (!set) return [];
    const out: BlockEntityRecord[] = [];
    for (const pk of set) {
      const r = this.byPos.get(pk);
      if (r) out.push(r);
    }
    return out;
  }

  /** Section keys that currently hold at least one BE (for an initial bake sweep). */
  occupiedSections(): SectionKey[] {
    return [...this.bySection.keys()];
  }

  /** Every BE record (insertion order). Used by the render gate to test for active animations. */
  values(): IterableIterator<BlockEntityRecord> {
    return this.byPos.values();
  }

  /**
   * Toggle a BE's animating flag. Returns the section key to re-mesh, or null if the position has no BE
   * or the flag is unchanged (no-op → no needless re-mesh).
   */
  setAnimating(x: number, y: number, z: number, animating: boolean): SectionKey | null {
    const rec = this.byPos.get(posKey(x, y, z));
    if (!rec || rec.animating === animating) return null;
    rec.animating = animating;
    return sectionOfBlock(x, y, z);
  }

  /**
   * Set a container's lid OPEN target (chest/shulker). Returns the section key to invalidate so the baker
   * promptly re-evaluates the static/per-frame split, or null if the position has no BE or the target is
   * unchanged (no-op). The baker then ramps the lid openness toward the new target.
   */
  setOpen(x: number, y: number, z: number, open: boolean): SectionKey | null {
    const rec = this.byPos.get(posKey(x, y, z));
    if (!rec || !!rec.open === open) return null;
    rec.open = open;
    return sectionOfBlock(x, y, z);
  }

  clear(): void {
    this.byPos.clear();
    this.bySection.clear();
  }
}
