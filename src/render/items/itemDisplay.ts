// Item display transforms (vanilla `models/item/generated.json` + `models/block/block.json` `display`
// blocks). RENDERER_PLAN.md §18. Each context (GUI inventory slot, GROUND dropped entity, FIXED item
// frame) is a translate(px/16) · rotateXYZ(deg) · scale, applied around the item's centre — item models
// are authored in a [0,16]px ([0,1] block) box, so the geometry is centred (−0.5) before the transform.

import { DEG, mul, rotationX, rotationY, rotationZ, scaling, translation, type Mat4 } from "../../mesh/entity/mat4";

// "entity" = a `builtin/entity` item (shulker box, chest, banner, …) — no flat sprite; renders as its
// block-entity box model. It shares the BLOCK display transforms (iconic isometric / ground scale).
export type ItemKind = "generated" | "block" | "entity";
export type ItemContext = "gui" | "ground" | "fixed";

interface Display {
  rot: [number, number, number]; // degrees, applied X→Y→Z
  trans: [number, number, number]; // model px (1/16 block)
  scale: number;
}

// item/generated: GUI is unspecified → flat head-on (identity). GROUND scale 0.5; FIXED flips 180°.
const GENERATED: Record<ItemContext, Display> = {
  gui: { rot: [0, 0, 0], trans: [0, 0, 0], scale: 1 },
  ground: { rot: [0, 0, 0], trans: [0, 2, 0], scale: 0.5 },
  fixed: { rot: [0, 180, 0], trans: [0, 0, 0], scale: 1 },
};
// block/block: GUI is the iconic isometric (30°,225°, scale 0.625); GROUND scale 0.25; FIXED scale 0.5.
const BLOCK: Record<ItemContext, Display> = {
  gui: { rot: [30, 225, 0], trans: [0, 0, 0], scale: 0.625 },
  ground: { rot: [0, 0, 0], trans: [0, 3, 0], scale: 0.25 },
  fixed: { rot: [0, 0, 0], trans: [0, 0, 0], scale: 0.5 },
};

/** Display matrix for a [0,1]-block item model in a given context (includes the −0.5 centre). */
// A3: the display matrix is CONSTANT per (kind, ctx) — only 6 possibilities — so build each once and cache
// it instead of recomposing 6 mat4s every frame for every dropped item / GUI slot. Consumers only `mul()`
// it (read-only), so a shared cached buffer is safe.
const _displayCache = new Map<string, Mat4>();
export function itemDisplayMatrix(kind: ItemKind, ctx: ItemContext): Mat4 {
  const key = kind + "|" + ctx;
  let m = _displayCache.get(key);
  if (!m) {
    const d = (kind === "generated" ? GENERATED : BLOCK)[ctx]; // block + entity items use the BLOCK transforms
    m = mul(
      translation(d.trans[0] / 16, d.trans[1] / 16, d.trans[2] / 16),
      rotationX(d.rot[0] * DEG),
      rotationY(d.rot[1] * DEG),
      rotationZ(d.rot[2] * DEG),
      scaling(d.scale, d.scale, d.scale),
      translation(-0.5, -0.5, -0.5),
    );
    _displayCache.set(key, m);
  }
  return m;
}
