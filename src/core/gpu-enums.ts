// Backend-neutral GPU vocabulary. RENDERER_PLAN.md §2, §10, §12, §23.
//
// These enums are OUR words for the closed sets a GraphicsDevice understands. Each concrete
// backend maps them to its own vocabulary (WebGPU's `GPU*` string unions, WebGL's GLenums) in
// one place — see WebGPUDevice's `GPU_*` tables. Keeping our own enums (rather than leaking
// WebGPU's strings into the neutral interface) is what makes the abstraction boundary real.
//
// They are STRING-valued on purpose: the value is the neutral spelling, so logs and errors read
// `"rgba8-srgb"` not `5`, and a forgotten mapping fails loudly at the Record lookup. (The
// indexing/bit-flag enums — TerrainPass, Direction, DirtyFlag — stay NUMERIC in types.ts because
// they key arrays and masks; see STYLE_GUIDE.md "Enums".)

/** Which concrete `GraphicsDevice` implementation is in use. */
export enum BackendKind {
  WebGL2 = "webgl2",
  WebGPU = "webgpu",
  /** In-memory, GPU-less device for unit tests. */
  Fake = "fake",
}

/** What a buffer is for. Drives the backend's usage flags (vertex/index also get COPY for arena compaction). */
export enum BufferUsage {
  Vertex = "vertex",
  Index = "index",
  Uniform = "uniform",
  /** Read-only shader storage buffer (SSBO). WebGPU-only — used for the per-draw section-origin array the
   *  vertex shader indexes by `instance_index` (F2 completion: 16B/draw packed vs a 256B dynamic-offset slot). */
  Storage = "storage",
  /** Pure transfer scratch (COPY_SRC|COPY_DST). */
  Copy = "copy",
}

/**
 * Texture / render-target formats. `*Srgb` variants sample-decode and store-encode sRGB, so
 * blending happens in linear space (the stained-glass-tint fix, WEBGPU_FINDINGS §17.2).
 */
export enum TextureFormat {
  Rgba8 = "rgba8",
  Rgba8Srgb = "rgba8-srgb",
  /** Single channel, 8-bit (e.g. coverage / mask atlases). */
  R8 = "r8",
  Bgra8 = "bgra8",
  Bgra8Srgb = "bgra8-srgb",
  Depth24Plus = "depth24plus",
  Depth32Float = "depth32float",
}

/** Programmable stage(s) a binding is visible to. */
export enum ShaderStage {
  Vertex = "vertex",
  Fragment = "fragment",
  Compute = "compute",
}

/**
 * Scalar component type of a vertex attribute. `unorm*`/`snorm*` auto-decode to floats in the
 * shader; `uint*` stay integers. RENDERER_PLAN §10 (packed 20-byte vertex).
 */
export enum VertexScalarKind {
  Uint8 = "uint8",
  Uint16 = "uint16",
  Uint32 = "uint32",
  Float32 = "float32",
  Snorm8 = "snorm8",
  Unorm8 = "unorm8",
  Snorm16 = "snorm16",
  Unorm16 = "unorm16",
}

/** Depth comparison. `Greater*` is for reverse-Z (WEBGPU_FINDINGS §17.3). */
export enum CompareFn {
  Less = "less",
  LessEqual = "less-equal",
  Greater = "greater",
  GreaterEqual = "greater-equal",
  Always = "always",
}

/** Texture min/mag/mip filtering. MC terrain is `Nearest` (crisp texels). */
export enum FilterMode {
  Nearest = "nearest",
  Linear = "linear",
}

/** Texture wrap mode for out-of-[0,1] UVs. */
export enum AddressMode {
  Clamp = "clamp",
  Repeat = "repeat",
  Mirror = "mirror",
}

/** Face culling. MC cube faces are wound for `Back` culling. RENDERER_PLAN §12. */
export enum CullMode {
  None = "none",
  Back = "back",
  Front = "front",
}

/** Index buffer element width. */
export enum IndexFormat {
  Uint16 = "uint16",
  Uint32 = "uint32",
}

/** Primitive topology. `TriangleList` is the terrain/entity default; `LineList` draws vertex pairs as
 *  GPU lines (vanilla `RenderPipelines.LINES` — the structure-block wireframe). */
export enum PrimitiveTopology {
  TriangleList = "triangle-list",
  LineList = "line-list",
}

/** Sampled-texture binding component type (a bind-group-layout entry's `texture.sampleType`). */
export enum TextureSampleType {
  Float = "float",
  Uint = "uint",
}

/** Sampler binding type (a bind-group-layout entry's `sampler.type`). `NonFiltering` is required for
 *  textures sampled without interpolation when the device forbids filtering them. */
export enum SamplerBindingType {
  Filtering = "filtering",
  NonFiltering = "non-filtering",
}
