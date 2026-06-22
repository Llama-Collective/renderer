// Block-display entities reuse the EXISTING block baker — no new geometry. RENDERER_PLAN.md §18, Phase
// 4.5a (the cheapest, highest-value slice). `falling_block` / `primed_tnt` / dropped block-items /
// minecart contents are just a baked blockstate drawn at an entity transform. This module converts a
// `BakedBlockModel` (block-local [0,1] quads) into per-pass entity `RawVertex` buckets (ALL faces — an
// entity has no neighbors to cull against) and parses the blockstate strings the sim hands us.

import type { BakedBlockModel } from "../../mesh/model/BakedBlockModel";
import type { BlockProps } from "../../mesh/model/BlockStateResolver";
import { TerrainPass, TERRAIN_PASSES } from "../../types";
import type { RawVertex } from "../../mesh/VertexFormat";

export type EntityVertsByPass = Partial<Record<TerrainPass, RawVertex[]>>;

/** Convert a baked block model's quads into entity verts per pass (no face culling — draw all faces). */
export function bakedModelToEntityVerts(model: BakedBlockModel): EntityVertsByPass {
  const out: EntityVertsByPass = {};
  for (const quad of model.quads) {
    let bucket = out[quad.layer];
    if (!bucket) out[quad.layer] = bucket = [];
    for (let v = 0; v < 4; v++) {
      const p = quad.positions[v];
      const uv = quad.atlasUV[v];
      bucket.push({ x: p[0], y: p[1], z: p[2], u: uv[0], v: uv[1], normal: quad.normal, colorRGBA: quad.colorRGBA, material: quad.material });
    }
  }
  return out;
}

/** True if any pass has geometry. */
export function hasGeometry(v: EntityVertsByPass): boolean {
  return TERRAIN_PASSES.some((p) => (v[p]?.length ?? 0) > 0);
}

/**
 * Parse a canonical blockstate string ("minecraft:oak_log[axis=y,...]") into name + props. Accepts a
 * bare name (no brackets). Whitespace-tolerant. Returns the bare name unchanged for the namespace.
 */
export function parseBlockState(state: string): { name: string; props: BlockProps } {
  const open = state.indexOf("[");
  if (open < 0) return { name: state.trim(), props: {} };
  const name = state.slice(0, open).trim();
  const inner = state.slice(open + 1, state.lastIndexOf("]"));
  const props: BlockProps = {};
  for (const pair of inner.split(",")) {
    const eq = pair.indexOf("=");
    if (eq < 0) continue;
    const k = pair.slice(0, eq).trim();
    const val = pair.slice(eq + 1).trim();
    if (k) props[k] = val;
  }
  return { name, props };
}
