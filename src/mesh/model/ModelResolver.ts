// Resolve a block model's parent chain + texture `#refs`. RENDERER_PLAN §24.3.
//
// Vanilla semantics (ResolvedModel/TextureSlots): walk `parent` to the root; child `textures`
// override parent; the FIRST `elements` found walking up (deepest child) wins; `ambientocclusion`
// is the top-most definition (default true). Texture refs `#a → #b → "block/x"` resolve to a
// fixpoint against the MERGED slot map.
//
// We fix cubane's bugs: NO artificial depth cap (cycle-detect instead), and the `{sprite,
// force_translucent}` object form is honored.

import {
  stripNamespace,
  type RawBlockModel,
  type RawTexture,
  type ResolvedModel,
  type ResolvedTexture,
} from "./ModelTypes";

/** Read-only source of raw model JSON by id (e.g. "block/oak_stairs"). */
export interface RawModelProvider {
  getModel(id: string): RawBlockModel | undefined;
}

const MISSING_SPRITE = "block/missing";

/** Resolve `rootId` (namespaced ok) to a merged, ref-resolved model ready for baking. */
export function resolveModel(rootId: string, provider: RawModelProvider): ResolvedModel {
  // Collect the chain child→root (chain[0] = root model, chain[last] = top ancestor).
  const chain: RawBlockModel[] = [];
  const seen = new Set<string>();
  let id: string | undefined = stripNamespace(rootId);
  while (id && !seen.has(id)) {
    seen.add(id);
    const model = provider.getModel(id);
    if (!model) break;
    chain.push(model);
    id = model.parent ? stripNamespace(model.parent) : undefined;
  }

  // Merge top→child so the child's entries win (processed last).
  const rawTextures: Record<string, RawTexture> = {};
  let elements: RawBlockModel["elements"];
  let ambientocclusion: boolean | undefined;
  for (let i = chain.length - 1; i >= 0; i--) {
    const m = chain[i];
    if (m.textures) Object.assign(rawTextures, m.textures);
    if (m.elements) elements = m.elements; // last (deepest child) with elements wins
    if (m.ambientocclusion !== undefined) ambientocclusion = m.ambientocclusion;
  }

  return {
    ambientocclusion: ambientocclusion ?? true,
    textures: resolveTextureSlots(rawTextures),
    elements: elements ?? [],
  };
}

/** Resolve every slot to a literal sprite, following `#refs` to a fixpoint. */
function resolveTextureSlots(raw: Record<string, RawTexture>): Record<string, ResolvedTexture> {
  const out: Record<string, ResolvedTexture> = {};
  for (const slot of Object.keys(raw)) out[slot] = resolveSlot(slot, raw);
  return out;
}

function resolveSlot(slot: string, raw: Record<string, RawTexture>): ResolvedTexture {
  let value: RawTexture | undefined = raw[slot];
  let forceTranslucent = false;
  const visited = new Set<string>([slot]);

  for (let guard = 0; guard < 64; guard++) {
    if (value === undefined) break;
    if (typeof value === "object") {
      forceTranslucent ||= value.force_translucent === true;
      value = value.sprite ?? "";
    }
    // value is a string: either "#ref" or a literal sprite id.
    if (value.startsWith("#")) {
      const ref = value.slice(1);
      if (visited.has(ref) || !(ref in raw)) return { sprite: MISSING_SPRITE, forceTranslucent };
      visited.add(ref);
      value = raw[ref];
      continue;
    }
    return { sprite: value ? stripNamespace(value) : MISSING_SPRITE, forceTranslucent };
  }
  return { sprite: MISSING_SPRITE, forceTranslucent };
}

/**
 * Resolve a face's `texture` (`#slot` or literal) against a resolved model's slot map.
 * Used by the bakery.
 */
export function resolveFaceTexture(ref: string, textures: Record<string, ResolvedTexture>): ResolvedTexture {
  if (ref.startsWith("#")) {
    return textures[ref.slice(1)] ?? { sprite: MISSING_SPRITE, forceTranslucent: false };
  }
  return { sprite: stripNamespace(ref), forceTranslucent: false };
}
