// Opaque GPU resource handles. RENDERER_PLAN.md §2, §23.
//
// These are PHANTOM types: at compile time each handle is a distinct brand so the type system
// stops you mixing a buffer handle up with a texture handle, or constructing one by hand. At
// runtime a handle is just whatever backing object the active backend stores (see each
// backend's `*Rec` records) — the brand has no runtime footprint.
//
// They live in their own file because EVERY layer above core/ traffics in them while staying
// free of real WebGL/WebGPU types (mesh/, world/, workers/ may import these handles but never a
// `GPU*`/`WebGL*` type). Keep this file dependency-free.

declare const BufferBrand: unique symbol;
declare const TextureBrand: unique symbol;
declare const ShaderBrand: unique symbol;
declare const SamplerBrand: unique symbol;
declare const PipelineBrand: unique symbol;
declare const BindingsBrand: unique symbol;
declare const RenderBundleBrand: unique symbol;

/** A GPU vertex/index/uniform buffer. Created via `GraphicsDevice.createBuffer`. */
export type GpuBufferHandle = { readonly [BufferBrand]: true };
/** A GPU texture (atlas, depth target, …). Created via `GraphicsDevice.createTexture`. */
export type GpuTextureHandle = { readonly [TextureBrand]: true };
/** A compiled shader module (one vertex + fragment, or compute, entry set). */
export type ShaderModuleHandle = { readonly [ShaderBrand]: true };
/** A texture sampler. Created via `GraphicsDevice.createSampler`. */
export type SamplerHandle = { readonly [SamplerBrand]: true };
/** An immutable render pipeline (shader + vertex layout + fixed-function state). */
export type PipelineHandle = { readonly [PipelineBrand]: true };
/** A bound resource group (buffers/samplers/textures) for a pipeline. */
export type BindingsHandle = { readonly [BindingsBrand]: true };
/** A pre-recorded sequence of draw commands (WebGPU GPURenderBundle) for cheap replay (F1).
 *  Created via `GraphicsDevice.createRenderBundle`, replayed via `PassEncoder.executeBundles`. */
export type RenderBundleHandle = { readonly [RenderBundleBrand]: true };
