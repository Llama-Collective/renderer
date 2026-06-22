// Resolve a concrete BlockState (block + property map) → the list of model parts to bake.
// RENDERER_PLAN §24.4. Vanilla semantics (BlockStateModelDispatcher / MultiPartModel).
//
//  - `variants`: pick the entry whose comma-key predicate matches (AND over listed `prop=val`;
//    unlisted props are wildcards; `""` matches everything). The most specific match wins.
//  - `multipart`: every case whose `when` passes contributes its `apply` (results concatenated).
//  - `when` grammar: object of `key:value` = AND of keys; value `a|b` = OR; leading `!` = negate;
//    `{"OR":[...]}` / `{"AND":[...]}` = explicit combinators (recursive).
//  - Weighted arrays: pick index 0 (deterministic — §24.4; position-seeded is a later option).

import {
  stripNamespace,
  type RawBlockState,
  type RawMultipartCase,
  type RawVariant,
  type RawWhen,
  type VariantPart,
} from "./ModelTypes";

/** Block property map, e.g. { facing: "east", half: "bottom", shape: "straight" }. Values are strings. */
export type BlockProps = Record<string, string>;

export function resolveBlockState(state: RawBlockState, props: BlockProps): VariantPart[] {
  if (state.variants) {
    const v = matchVariant(state.variants, props);
    return v ? [normalizePart(pickVariant(v))] : [];
  }
  if (state.multipart) {
    return state.multipart
      .filter((c: RawMultipartCase) => !c.when || evalWhen(c.when, props))
      .map((c) => normalizePart(pickVariant(c.apply)));
  }
  return [];
}

// --- variants --------------------------------------------------------------------
function matchVariant(
  variants: Record<string, RawVariant | RawVariant[]>,
  props: BlockProps,
): RawVariant | RawVariant[] | undefined {
  let best: RawVariant | RawVariant[] | undefined;
  let bestSpecificity = -1;
  for (const key of Object.keys(variants)) {
    const pairs = parseVariantKey(key);
    if (pairs.every(([p, v]) => props[p] === v) && pairs.length > bestSpecificity) {
      bestSpecificity = pairs.length;
      best = variants[key];
    }
  }
  if (best) return best;
  // No variant's full key is satisfied — the caller gave UNDER-SPECIFIED props (e.g. `hopper` with no
  // `facing`/`enabled`, common when a block is atlased/baked from a bare name). Vanilla resolves against the
  // block's full default state; we don't have it, so fall back to the first variant the given props don't
  // CONTRADICT (a missing prop = wildcard/default) — else the first variant. This keeps an under-specified
  // block rendering a sensible model (and, crucially, lets sprite-discovery atlas its textures) instead of
  // vanishing. Exact matches above are unaffected, so fully-specified states are unchanged.
  for (const key of Object.keys(variants)) {
    if (parseVariantKey(key).every(([p, v]) => props[p] === undefined || props[p] === v)) return variants[key];
  }
  const first = Object.keys(variants)[0];
  return first !== undefined ? variants[first] : undefined;
}

/** "facing=east,half=bottom" → [["facing","east"],["half","bottom"]]; "" → []. */
function parseVariantKey(key: string): Array<[string, string]> {
  if (key === "") return [];
  return key.split(",").map((tok) => {
    const eq = tok.indexOf("=");
    return [tok.slice(0, eq), tok.slice(eq + 1)] as [string, string];
  });
}

// --- multipart `when` ------------------------------------------------------------
function evalWhen(when: RawWhen, props: BlockProps): boolean {
  if ("OR" in when) return (when.OR as RawWhen[]).some((w) => evalWhen(w, props));
  if ("AND" in when) return (when.AND as RawWhen[]).every((w) => evalWhen(w, props));
  return Object.entries(when as Record<string, string>).every(([prop, spec]) =>
    matchValueSpec(props[prop], spec),
  );
}

/** `"side|up"` = OR of values; `"!none"` = negation; combinations supported. Vanilla compares blockstate
 *  `when` values as strings, but the JSON may carry them as numbers/booleans (e.g. redstone_wire's
 *  `{"power": 0}` cases, or `{"powered": false}`) — coerce to string so those don't crash `.split`. */
function matchValueSpec(actual: string | undefined, spec: string | number | boolean): boolean {
  const value = actual ?? "";
  const positives: string[] = [];
  const negatives: string[] = [];
  for (const term of String(spec).split("|")) {
    if (term.startsWith("!")) negatives.push(term.slice(1));
    else positives.push(term);
  }
  if (positives.length > 0 && !positives.includes(value)) return false;
  if (negatives.includes(value)) return false;
  return true;
}

// --- shared ----------------------------------------------------------------------
function pickVariant(v: RawVariant | RawVariant[]): RawVariant {
  return Array.isArray(v) ? v[0] : v; // index 0 (deterministic) — §24.4
}

function normalizePart(v: RawVariant): VariantPart {
  return { model: stripNamespace(v.model), x: v.x ?? 0, y: v.y ?? 0, z: v.z ?? 0, uvlock: v.uvlock === true };
}
