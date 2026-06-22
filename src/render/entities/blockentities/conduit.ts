// Conduit. RENDERER_PLAN.md §18, Phase 4.5e. Vanilla ConduitRenderer:
//   - inactive → the shell (6³) spinning slowly about Y;
//   - active (`active=true`, i.e. an open prismarine frame) → a bobbing cage (8³) spinning about a TILTED
//     axis (vanilla `normalize(0.5,1,0.5)`, NOT pure Y) plus a CAMERA-FACING eye (open when hunting, else
//     closed) billboarded toward the camera (`ctx.cameraPos`). The animated `wind` cloud (a multi-frame
//     texture strip) is still omitted. Multi-texture, so this hand-builds bake groups instead of `boxBE`.

import { bakeModel, type ModelPartDef } from "../../../mesh/entity/ModelPart";
import type { BlockEntityRecord } from "../../../world/BlockEntityIndex";
import { TerrainPass, type Vec3 } from "../../../types";
import type { Mat4 } from "../../../mesh/entity/mat4";
import type { BEBakeContext, BlockEntityDef } from "./registry";
import { identity, mul, rotationAxis, rotationY, scaling, translation } from "./transforms";

const SHELL: ModelPartDef[] = [{ name: "shell", pivot: [0, 0, 0], cubes: [{ origin: [-3, -3, -3], size: [6, 6, 6], uv: [0, 0] }] }];
const CAGE: ModelPartDef[] = [{ name: "cage", pivot: [0, 0, 0], cubes: [{ origin: [-4, -4, -4], size: [8, 8, 8], uv: [0, 0] }] }];
// inflate 0.01 (vanilla CubeDeformation) makes the eye a thin slab, not a degenerate zero-Z plane — both
// faces then render (so it shows from either side of the billboard).
const EYE: ModelPartDef[] = [{ name: "eye", pivot: [0, 0, 0], cubes: [{ origin: [-4, -4, 0], size: [8, 8, 0], uv: [0, 0], inflate: 0.01 }] }];

const BASE_TEX = "entity/conduit/base";
const CAGE_TEX = "entity/conduit/cage";
const OPEN_EYE_TEX = "entity/conduit/open_eye";
const CLOSED_EYE_TEX = "entity/conduit/closed_eye";

const isActive = (rec: BlockEntityRecord) => rec.props.active === "true";

/** Rotation whose +Z (the eye quad's facing) points toward `toCam`, +X horizontal-right, +Y up — a
 *  spherical billboard so the flat eye always faces the camera. Column-major (cols = right, up, forward). */
function billboard(toCam: Vec3): Mat4 {
  const fl = Math.hypot(toCam[0], toCam[1], toCam[2]) || 1;
  const fx = toCam[0] / fl, fy = toCam[1] / fl, fz = toCam[2] / fl;
  let rx = fz, ry = 0, rz = -fx; // right = up([0,1,0]) × forward
  const rl = Math.hypot(rx, ry, rz);
  if (rl < 1e-4) { rx = 1; rz = 0; } else { rx /= rl; rz /= rl; }
  const ux = fy * rz - fz * ry, uy = fz * rx - fx * rz, uz = fx * ry - fy * rx; // up = forward × right
  const m = identity();
  m[0] = rx; m[1] = ry; m[2] = rz;
  m[4] = ux; m[5] = uy; m[6] = uz;
  m[8] = fx; m[9] = fy; m[10] = fz;
  return m;
}

// Vanilla ConduitBlockEntity.getActiveRotation = tick·-0.0375. The inactive shell reads it as DEGREES
// (≈0.75°/s — nearly static); the active cage reads it as RADIANS (≈0.75 rad/s ≈ 43°/s). 20 ticks/s.
const SHELL_RAD_PER_S = (-0.0375 * 20 * Math.PI) / 180; // ≈ -0.013 rad/s
const CAGE_RAD_PER_S = -0.0375 * 20; // ≈ -0.75 rad/s

export const CONDUIT: BlockEntityDef = {
  types: ["conduit"],
  textures: [BASE_TEX, CAGE_TEX, OPEN_EYE_TEX, CLOSED_EYE_TEX],
  animated: true,
  // The shell/cage spin + the camera-facing eye are a continuous idle loop vanilla always shows (BBE §5).
  // Without this the conduit bakes frozen on the STATIC path, which also has no cameraPos → the billboard
  // eye points at a fixed +Z. idleLoop routes it to the per-frame path (clock + cameraPos) by default.
  idleLoop: true,
  bake(rec: BlockEntityRecord, ctx: BEBakeContext, clock: number, animating: boolean) {
    const o = ctx.sectionOrigin;
    const cx = rec.x - o[0], cy = rec.y - o[1], cz = rec.z - o[2];
    const t = animating ? clock : 0; // idle (static-baked) conduit rests; animating one spins/bobs
    const out = [];

    if (!isActive(rec)) {
      const rect = ctx.uvFor(BASE_TEX);
      if (rect) {
        const base = mul(translation(cx + 0.5, cy + 0.5, cz + 0.5), rotationY(t * SHELL_RAD_PER_S));
        out.push(...bakeModel(SHELL, { texWidth: 32, texHeight: 16, uvRect: rect, base, pass: TerrainPass.Cutout, shade: true }));
      }
      return out.length ? out : null;
    }

    // Active: bob the cage + eye on a vanilla-derived height curve (hh = h²+h, h = sin(t·2)/2+0.5).
    const h = Math.sin(t * 2) / 2 + 0.5;
    const bobY = 0.3 + (h * h + h) * 0.2;
    // Cage (8³) spinning about vanilla's TILTED axis normalize(0.5,1,0.5) — NOT pure Y.
    const cageRect = ctx.uvFor(CAGE_TEX);
    if (cageRect) {
      const base = mul(translation(cx + 0.5, cy + bobY, cz + 0.5), rotationAxis(t * CAGE_RAD_PER_S, 0.40824829, 0.81649658, 0.40824829));
      out.push(...bakeModel(CAGE, { texWidth: 32, texHeight: 16, uvRect: cageRect, base, pass: TerrainPass.Cutout, shade: true }));
    }
    // Eye billboard (open when hunting) — net scale 0.6667 (vanilla 0.5 · 1.3333), facing the camera.
    const eyeRect = ctx.uvFor(rec.props.hunting === "true" ? OPEN_EYE_TEX : CLOSED_EYE_TEX);
    if (eyeRect) {
      // Toward-camera direction in WORLD space (camera & eye are both world; a pure rotation is
      // translation-invariant, so it's valid in the section-local bake). Falls back to +Z when no camera.
      const eyeWorld: Vec3 = [rec.x + 0.5, rec.y + bobY, rec.z + 0.5];
      const toCam: Vec3 = ctx.cameraPos
        ? [ctx.cameraPos[0] - eyeWorld[0], ctx.cameraPos[1] - eyeWorld[1], ctx.cameraPos[2] - eyeWorld[2]]
        : [0, 0, 1];
      const base = mul(translation(cx + 0.5, cy + bobY, cz + 0.5), billboard(toCam), scaling(0.6667, 0.6667, 0.6667));
      out.push(...bakeModel(EYE, { texWidth: 16, texHeight: 16, uvRect: eyeRect, base, pass: TerrainPass.Cutout, shade: false }));
    }
    return out.length ? out : null;
  },
};
