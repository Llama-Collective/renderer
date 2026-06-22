// Decode PNG bytes → RGBA pixels (browser). RENDERER_PLAN §24.9.
//
// Uses createImageBitmap + OffscreenCanvas (both DOM/worker globals). Animated textures decode as
// the full vertical frame strip; the AtlasManager splits frames. Browser-only — verified live, not
// in vitest (node has no canvas).

export interface DecodedImage {
  width: number;
  height: number;
  /** RGBA8, row-major, length width*height*4. */
  rgba: Uint8Array;
}

export async function decodePng(bytes: Uint8Array): Promise<DecodedImage> {
  const bitmap = await createImageBitmap(new Blob([bytes as BlobPart], { type: "image/png" }));
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("TextureDecoder: no 2D context");
    ctx.drawImage(bitmap, 0, 0);
    const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    return { width: bitmap.width, height: bitmap.height, rgba: new Uint8Array(data) };
  } finally {
    bitmap.close();
  }
}
