// Bake a block (name + properties) into render-ready quads + occlusion. RENDERER_PLAN §24.5–§24.8.
//
// Ties the resolution → bake → atlas-UV → tint → layer → occlusion pipeline together and caches by
// blockstate (cache key includes ALL properties — TRAP 8.A). Pure data (GPU-free): the atlas UV
// rects, sprite opacity, and tint are injected, so this can run in a worker. The mesher consumes
// `BakedBlockModel`s and only does per-quad culling + packing.

import { Direction, TerrainPass, type SpriteUv } from "../../types";
import { resolveBlockState, type BlockProps } from "./BlockStateResolver";
import { bakeParts } from "./FaceBakery";
import type { Vec2 } from "./FaceBakery";
import type { Vec3 } from "./bakeMath";
import { computeOcclusion, type OcclusionShape } from "./OcclusionShape";
import { classifyLayer, isOpaqueSprite, type SpriteOpacity } from "./RenderLayer";
import { deriveSkipGroup } from "./SkipGroup";
import { TintProvider } from "./TintProvider";
import type { RawModelProvider } from "./ModelResolver";
import type { RawBlockState } from "./ModelTypes";
import { NO_SHADE } from "../VertexFormat";

/** A quad ready for the mesher: atlas-mapped UV, resolved tint, render pass. */
export interface BakedRenderQuad {
  positions: [Vec3, Vec3, Vec3, Vec3];
  /** Atlas-space UV [0,1]. */
  atlasUV: [Vec2, Vec2, Vec2, Vec2];
  /** 0..5 facing for shade, or NO_SHADE. */
  normal: number;
  cullface: Direction | null;
  layer: TerrainPass;
  /** Baked tint 0xRRGGBBAA (alpha 255). */
  colorRGBA: number;
  /** Material bits (1 = alpha-cutout). */
  material: number;
}

export interface BakedBlockModel {
  quads: BakedRenderQuad[];
  occlusion: OcclusionShape;
  /** Same non-null group as a neighbor → their shared face is culled (clear glass / panes). */
  skipGroup: string | null;
  /** SL-1: per-block directional-shade multiplier (`block.getShadeBrightness`-style). Smooth
   *  lighting scales a face's corner brightness by this. Absent ⇒ 1.0 (no extra shade). */
  shadeBrightness?: number;
  /** SL-1: the block emits its own light (glowstone, lava). Smooth lighting bakes it FLAT (no AO/shadow
   *  darkening) — parity. Absent ⇒ false. */
  emissive?: boolean;
}

/** One palette entry: how a block id maps to a blockstate to bake. */
export interface BlockEntry {
  name: string;
  props: BlockProps;
  /** Optional explicit skip-render group OVERRIDE. When omitted, the group is derived from the block
   *  type (`deriveSkipGroup`) at bake time, so callers normally leave this unset. */
  cullGroup?: string | null;
}

export interface BakerDeps {
  blockstate: (name: string) => RawBlockState | undefined;
  models: RawModelProvider;
  uvFor: (sprite: string) => SpriteUv | undefined;
  opacityOf: (sprite: string) => SpriteOpacity | undefined;
  tint: TintProvider;
  /** SL-1 (optional): per-block directional-shade multiplier for smooth lighting. Default 1.0. */
  shadeBrightnessOf?: (entry: BlockEntry) => number;
  /** SL-1 (optional): whether a block emits light (baked FLAT under smooth lighting). Default false. */
  emissiveOf?: (entry: BlockEntry) => boolean;
}

const EMPTY: BakedBlockModel = {
  quads: [],
  occlusion: computeOcclusion([], () => true),
  skipGroup: null,
};

export class ModelBaker {
  private readonly cache = new Map<string, BakedBlockModel>();

  constructor(private readonly deps: BakerDeps) {}

  /** Get-or-bake the model for a palette entry (cached by name + full property set). */
  bake(entry: BlockEntry): BakedBlockModel {
    const key = `${entry.name}|${propsKey(entry.props)}`;
    let model = this.cache.get(key);
    if (!model) {
      model = this.bakeFresh(entry);
      this.cache.set(key, model);
    }
    return model;
  }

  private bakeFresh(entry: BlockEntry): BakedBlockModel {
    const state = this.deps.blockstate(entry.name);
    if (!state) return EMPTY;
    const geometry = bakeParts(resolveBlockState(state, entry.props), this.deps.models);

    const quads: BakedRenderQuad[] = geometry.map((q) => {
      const rect = this.deps.uvFor(q.sprite);
      // Missing sprite (not in the atlas): collapse the face to a single atlas texel rather than the raw
      // [0,1] model UV, which would sample the WHOLE atlas and smear unrelated textures across the face
      // (the "broken textures" failure). Sprite discovery should make this unreachable; this is a safety net.
      const atlasUV = q.uvs.map(([u, v]) =>
        rect ? [rect.u + u * rect.width, rect.v + v * rect.height] : [0, 0],
      ) as [Vec2, Vec2, Vec2, Vec2];
      const layer = classifyLayer(this.deps.opacityOf(q.sprite), q.forceTranslucent);
      const tintRGB = this.deps.tint.tintFor(entry.name, q.tintindex, undefined, entry.props);
      return {
        positions: q.positions,
        atlasUV,
        normal: q.shade ? q.normal : NO_SHADE,
        cullface: q.cullface,
        layer,
        colorRGBA: ((tintRGB << 8) | 0xff) >>> 0, // RGB → RGBA, opaque
        material: layer === TerrainPass.Cutout ? 1 : 0,
      };
    });

    const occlusion = computeOcclusion(geometry, (sprite) => isOpaqueSprite(this.deps.opacityOf(sprite)));
    // Skip-render group: an explicit palette `cullGroup` wins; otherwise derive it from the block type
    // so real blocks (glass / stained glass / panes / ice / honey / slime / bars) self-cull in
    // production, not only in hand-authored harness palettes (AUDIT H2 / TRAP 8.A/8.B).
    return {
      quads,
      occlusion,
      skipGroup: entry.cullGroup ?? deriveSkipGroup(entry.name),
      shadeBrightness: this.deps.shadeBrightnessOf?.(entry) ?? 1.0, // SL-1
      emissive: this.deps.emissiveOf?.(entry) ?? false,
    };
  }
}

function propsKey(props: BlockProps): string {
  return Object.keys(props)
    .sort()
    .map((k) => `${k}=${props[k]}`)
    .join(",");
}
