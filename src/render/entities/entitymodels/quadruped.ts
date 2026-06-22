// Generic QuadrupedModel family (vanilla `client/model/QuadrupedModel.java`). RENDERER_PLAN.md §18,
// Phase 4.6. `createBodyMesh(legSize, mirrorL, mirrorR)`: a head + a (x-rotated) body + 4 corner legs;
// the built-in walk swings the legs in antiphase diagonal pairs (cos(pos·0.6662)·1.4·speed). One factory
// + a per-mob head/body override unlocks pig, cow, sheep, and most farm/forest quadrupeds (the key reuse
// lever from ENTITY_RENDERING_FINDINGS.md). Geometry only — GPU-free, like the rest of `mesh/entity`.

import type { EntityCube, ModelPartDef, PartPose } from "../../../mesh/entity/ModelPart";
import type { RenderEntity } from "../EntityScene";

export interface QuadrupedOpts {
  legSize: number;
  /** Head + body parts (each mob overrides these; pivots in model px). */
  head: ModelPartDef;
  body: ModelPartDef;
  legX?: number; // leg X offset from centre (default 3; cow/horse 4)
  legZHind?: number; // default 7
  legZFront?: number; // default -5
  legW?: number; // leg box width/depth (default 4)
  legUv?: [number, number]; // default [0, 16]
  mirrorLeftLeg?: boolean;
  mirrorRightLeg?: boolean;
}

/** Build the head + body + 4 corner legs (vanilla `createBodyMesh` + `createLegs`). */
export function quadrupedMesh(o: QuadrupedOpts): ModelPartDef[] {
  const legX = o.legX ?? 3, zh = o.legZHind ?? 7, zf = o.legZFront ?? -5, w = o.legW ?? 4, uv = o.legUv ?? [0, 16];
  const y = 24 - o.legSize; // vanilla leg pose Y
  const cube = (mirror: boolean): EntityCube => ({ origin: [-w / 2, 0, -2], size: [w, o.legSize, w], uv, mirror });
  const leg = (name: string, x: number, z: number, mirror: boolean): ModelPartDef => ({ name, pivot: [x, y, z], cubes: [cube(mirror)] });
  const ml = o.mirrorLeftLeg ?? false, mr = o.mirrorRightLeg ?? false;
  return [
    o.head,
    o.body,
    leg("right_hind_leg", -legX, zh, mr),
    leg("left_hind_leg", legX, zh, ml),
    leg("right_front_leg", -legX, zf, mr),
    leg("left_front_leg", legX, zf, ml),
  ];
}

/**
 * The QuadrupedModel walk: legs swing in antiphase diagonal pairs (right-hind + left-front together,
 * left-hind + right-front opposite), amplitude = 1.4·speed. The sim has no `walkAnimationPos`, so we
 * synthesize the phase from the render clock and derive speed from velocity.
 */
export function quadrupedWalk(e: RenderEntity, clock: number): Record<string, PartPose> {
  const v = e.velocity;
  const speed = Math.min(Math.hypot(v[0], v[2]) * 4, 1);
  if (speed < 1e-3) return {};
  const phase = clock * 6.0;
  const a = Math.cos(phase) * 1.4 * speed;
  const b = Math.cos(phase + Math.PI) * 1.4 * speed;
  return {
    right_hind_leg: { rotation: [a, 0, 0] },
    left_hind_leg: { rotation: [b, 0, 0] },
    right_front_leg: { rotation: [b, 0, 0] },
    left_front_leg: { rotation: [a, 0, 0] },
  };
}
