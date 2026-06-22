// Section coordinate keying. RENDERER_PLAN.md §4, §22 (unit-tested, incl. borders).

import type { Vec3i } from "../types";

/** A section's world coordinates (in section units, i.e. block coords floor-divided by 16). */
export type SectionKey = string & { readonly __sectionKey: true };

export function sectionKey(sx: number, sy: number, sz: number): SectionKey {
  return `${sx},${sy},${sz}` as SectionKey;
}

export function parseSectionKey(k: SectionKey): Vec3i {
  const parts = (k as string).split(",");
  return [Number(parts[0]), Number(parts[1]), Number(parts[2])];
}

/**
 * Section containing a block coordinate. `>> 4` is arithmetic shift = floor-divide by 16,
 * which is correct for negative coordinates too (e.g. -1 >> 4 === -1, -17 >> 4 === -2).
 */
export function sectionOfBlock(bx: number, by: number, bz: number): SectionKey {
  return sectionKey(bx >> 4, by >> 4, bz >> 4);
}
