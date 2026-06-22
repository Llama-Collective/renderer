// Beacon beam visibility — a faithful port of BeaconBlockEntity's beam logic, computed from the world
// blocks (so it works whether the sim runs or the editor is paused). Vanilla:
//   reference_code/26.1.2/src/net/minecraft/world/level/block/entity/BeaconBlockEntity.java
//
// The beam shows IFF BOTH:
//   1. the beacon is ACTIVATED — a valid base pyramid below it (`updateBase` returns `levels > 0`:
//      1..4 complete (2·step+1)² layers of BEACON_BASE_BLOCKS at y-step), and
//   2. the column above it (beacon+1 up to the build top / sky) is unobstructed — an opaque block clears
//      the WHOLE beam (vanilla `checkingBeamSections.clear()`); stained glass tints it; clear glass / air pass.
// `getBeamSections()` returns nothing when `levels == 0`, so a beacon WITHOUT a base shows no beam — the bug
// this fixes (we previously drew the beam unconditionally).

import { DYE_DIFFUSE } from "../render/entities/blockentities/dyes";

/** BlockTags.BEACON_BASE_BLOCKS (data/minecraft/tags/block/beacon_base_blocks.json). */
const BEACON_BASE_BLOCKS = new Set(["iron_block", "gold_block", "diamond_block", "emerald_block", "netherite_block"]);

/** Extra beam height above the highest block, so an unobstructed beam visibly shoots into the sky. */
const SKY_MARGIN = 24;

const strip = (name: string): string => name.replace(/^minecraft:/, "");

/** Result: whether the beam is visible, its (glass-tinted) color 0xRRGGBB, and its height in blocks. */
export interface BeaconBeam {
  active: boolean;
  color: number;
  height: number;
}

/** What a block in the beam column does to the beam. */
type ColumnEffect = "pass" | "block" | { tint: number };

/** A stained-glass block/pane (a vanilla BeaconBeamBlock) tints the beam by its dye; clear glass passes;
 *  tinted glass (light-dampening 15) and any other solid block blocks it. Air is handled by the caller
 *  (a null sample). */
function columnEffect(name: string): ColumnEffect {
  const s = strip(name);
  const stained = /^([a-z_]+)_stained_glass(?:_pane)?$/.exec(s);
  if (stained && DYE_DIFFUSE[stained[1]] !== undefined) return { tint: DYE_DIFFUSE[stained[1]] };
  if (s === "glass" || s === "glass_pane") return "pass"; // clear glass: transparent, no tint
  return "block"; // tinted_glass + every solid block dampen the beam
}

/** Per-channel average of two 0xRRGGBB colors (vanilla ARGB.average for stacked glass sections). */
function averageRgb(a: number, b: number): number {
  const r = (((a >> 16) & 0xff) + ((b >> 16) & 0xff)) >> 1;
  const g = (((a >> 8) & 0xff) + ((b >> 8) & 0xff)) >> 1;
  const bl = ((a & 0xff) + (b & 0xff)) >> 1;
  return (r << 16) | (g << 8) | bl;
}

/**
 * Compute the beacon beam at (x,y,z). `blockAt` returns the block name at a position, or null for air/empty.
 * `scanTop` is the highest block y in the world (the column is scanned up to it; above is open sky). Returns
 * `active=false` (no beam) when there is no base pyramid OR the column is obstructed.
 */
export function computeBeaconBeam(
  blockAt: (x: number, y: number, z: number) => string | null,
  x: number,
  y: number,
  z: number,
  scanTop: number,
): BeaconBeam {
  const none: BeaconBeam = { active: false, color: 0xffffff, height: 0 };

  // updateBase: count complete pyramid layers (1..4) of beacon-base blocks below.
  let levels = 0;
  for (let step = 1; step <= 4; step++) {
    let complete = true;
    for (let lx = x - step; lx <= x + step && complete; lx++) {
      for (let lz = z - step; lz <= z + step; lz++) {
        const name = blockAt(lx, y - step, lz);
        if (!name || !BEACON_BASE_BLOCKS.has(strip(name))) {
          complete = false;
          break;
        }
      }
    }
    if (!complete) break;
    levels = step;
  }
  if (levels === 0) return none; // not activated → no beam (getBeamSections() returns empty when levels == 0)

  // Column scan: a white beam tinted by stained glass; any opaque block clears the whole beam.
  let color = 0xffffff;
  let glassCount = 0;
  for (let cy = y + 1; cy <= scanTop; cy++) {
    const name = blockAt(x, cy, z);
    if (name === null) continue; // air → pass
    const effect = columnEffect(name);
    if (effect === "block") return none;
    if (effect !== "pass") {
      color = glassCount === 0 ? effect.tint : averageRgb(color, effect.tint);
      glassCount++;
    }
  }
  // Unobstructed → the beam reaches the sky (above the highest block).
  return { active: true, color, height: Math.max(scanTop - y, 0) + SKY_MARGIN };
}
