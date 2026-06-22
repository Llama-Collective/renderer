// Fixed-function state per terrain pass. RENDERER_PLAN.md §12.
//
// These three rows are load-bearing for transparency correctness:
//  - Cutout keeps depth WRITE on (alpha-tested, behaves like opaque).
//  - Translucent has depth TEST on but depth WRITE off + blend on, or glass occludes
//    glass behind it (TRAP 12.C).
// This is configuration, not logic, so it is defined for real — and it is the SINGLE source of truth:
// `TerrainRenderer.ensureResources` builds its solid/cutout/translucent pipelines by spreading these
// (layering on a renderer-specific `cull`), so the depth/blend rules can't drift between the two.

import { TerrainPass } from "../types";
import type { PassStateDesc } from "./GraphicsDevice";

export const PASS_STATE: Record<TerrainPass, PassStateDesc> = {
  [TerrainPass.Solid]: { depthTest: true, depthWrite: true, blend: false },
  [TerrainPass.Cutout]: { depthTest: true, depthWrite: true, blend: false },
  [TerrainPass.Translucent]: { depthTest: true, depthWrite: false, blend: true },
};
