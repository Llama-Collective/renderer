// Skip-render group derivation from block type. RENDERER_PLAN §8, §24.6 (TRAP 8.A/8.B). AUDIT H2.
//
// Vanilla culls the shared face between two blocks of the SAME translucent block via
// `BlockBehaviour.skipRendering`. Two families do this:
//   - HalfTransparentBlock (glass, stained glass, tinted glass, ice, frosted/blue ice, honey, slime):
//     `skipRendering` returns true iff `neighborState.is(this)` — the SAME block, every direction.
//   - IronBarsBlock (glass panes, iron bars, copper bars): same-block skip too, but on a HORIZONTAL
//     axis only when both panes are connected toward each other. This renderer's multipart bake only
//     emits a pane's connecting-arm geometry (the flush quad that would cull) WHEN it is connected, so
//     a plain block-identity group reproduces vanilla: the would-cull arm faces exist only when both
//     panes connect, and the post faces are never flush so they never cull.
//
// The cull mechanism (OcclusionShape.shouldDrawQuad) is symmetric string equality
// `selfSkipGroup === neighborSkipGroup`, so the group key is the block's identity (its name): the SAME
// block → SAME group → cull; a DIFFERENT block (e.g. a different stained-glass colour, or a glass block
// vs a same-colour pane) → DIFFERENT group → render the boundary (TRAP 8.B).
//
// Excluded on purpose: packed_ice and full leaves are OPAQUE — their shared faces are already culled by
// the opaque occlusion shape, no group needed. Same-fluid culling lives in the FluidMesher. Default
// (fancy) leaves do NOT skip-cull in vanilla, so leaves are intentionally not grouped. (Minor known gap:
// vanilla's BlockTags.BARS cross-type skip between iron bars and copper-bars variants is not modelled —
// each bars block only self-culls; an iron_bars↔copper_bars shared face renders.)

import { stripNamespace } from "./ModelTypes";

/** Blocks whose vanilla `skipRendering` self-culls, matched exactly (after stripping the namespace). */
const SELF_CULL_EXACT = new Set<string>([
  "glass",
  "tinted_glass",
  "ice",
  "frosted_ice",
  "blue_ice",
  "honey_block",
  "slime_block",
  "glass_pane",
  "iron_bars",
]);

/**
 * The skip-render group for a block, or `null` if it does not self-cull. Keyed by block identity, so
 * `deriveSkipGroup(a) === deriveSkipGroup(b)` (both non-null) iff `a` and `b` are the SAME self-culling
 * block — exactly vanilla's `neighborState.is(this)`.
 */
export function deriveSkipGroup(name: string): string | null {
  const n = stripNamespace(name);
  if (
    SELF_CULL_EXACT.has(n) ||
    n.endsWith("_stained_glass") || // 16 colours (StainedGlassBlock)
    n.endsWith("_stained_glass_pane") || // 16 colours (StainedGlassPaneBlock)
    n.endsWith("copper_bars") // copper_bars + (waxed_)(exposed|weathered|oxidized)_copper_bars (WeatheringCopperBarsBlock)
  ) {
    return n;
  }
  return null;
}
