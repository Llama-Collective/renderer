// Guards the "first piston activation" fix: the piston-transient blocks (piston_head / moving_piston)
// must stay in the always-preload/atlas set, else the first fire bakes an empty head AND forces a full
// rebuild (freeze). See the investigation in PISTON_PLAN. Pure-data — no GPU.

import { describe, it, expect } from "vitest";
import { PISTON_TRANSIENT_BLOCKS, ALWAYS_ATLASED_BLOCK_NAMES, ENTITY_CONTAINED_BLOCKS } from "./scene";

describe("always-atlased block set (first-piston-fire fix)", () => {
  it("PISTON_TRANSIENT_BLOCKS covers piston_head + moving_piston (preloaded + atlased up front)", () => {
    const names = PISTON_TRANSIENT_BLOCKS.map((b) => b.name);
    expect(names).toContain("minecraft:piston_head"); // its models aren't in the base piston's chain
    expect(names).toContain("minecraft:moving_piston");
  });

  it("ALWAYS_ATLASED_BLOCK_NAMES seeds atlasedNames with the piston transients (so they never trip a rebuild)", () => {
    expect(ALWAYS_ATLASED_BLOCK_NAMES).toContain("minecraft:piston_head");
    expect(ALWAYS_ATLASED_BLOCK_NAMES).toContain("minecraft:moving_piston");
    // and still includes the cart-contents it always covered
    for (const b of ENTITY_CONTAINED_BLOCKS) expect(ALWAYS_ATLASED_BLOCK_NAMES).toContain(b.name);
  });
});
