// Shared atlas pixel blit. RENDERER_PLAN.md §7.

/**
 * Copy a `sw`×`sh` RGBA8 tile from `src` (starting at byte `srcByteOffset`) into `dst` at pixel `(dx, dy)`.
 * `dstStridePx` is the destination atlas WIDTH in pixels. The single row-copy used by both atlas managers
 * (the block atlas wraps it with a frame offset for animated tiles; the entity/item atlas passes offset 0).
 */
export function blitRect(src: Uint8Array, sw: number, sh: number, srcByteOffset: number, dst: Uint8Array, dstStridePx: number, dx: number, dy: number): void {
  const rowBytes = sw * 4;
  for (let row = 0; row < sh; row++) {
    const s = srcByteOffset + row * rowBytes;
    const d = ((dy + row) * dstStridePx + dx) * 4;
    dst.set(src.subarray(s, s + rowBytes), d);
  }
}

/**
 * Replicate a cell's RIGHT column + BOTTOM row (and the corner) one pixel into the surrounding gutter.
 *
 * The UV rects map a sprite EDGE-to-EDGE (`p/N … (p+size)/N`) so explicit/partial face UVs (a trapdoor's
 * 3px side strip `uv:[0,16,16,13]`, a composter's rim) land on the EXACT texel boundary they name — the old
 * half-texel CENTER inset shifted those boundaries by ½ texel and bled the neighbouring texel onto the face.
 * The cost of edge-to-edge mapping is that a face whose UV reaches exactly 1.0 samples the texel JUST past
 * the sprite (the gutter). Mirroring the border into the 1px gutter (the packer always leaves one to the
 * right/below) makes that boundary sample the edge colour instead of the transparent gutter — no seam. Only
 * the far edges need it: u=0/v=0 map to the sprite's own first texel, never the gutter. Bounds-guarded so a
 * sprite flush against the atlas edge is left untouched.
 */
export function padCellEdges(dst: Uint8Array, atlasW: number, atlasH: number, x: number, y: number, size: number): void {
  const px = (cx: number, cy: number): number => (cy * atlasW + cx) * 4;
  if (x + size < atlasW) {
    for (let row = 0; row < size; row++) dst.copyWithin(px(x + size, y + row), px(x + size - 1, y + row), px(x + size - 1, y + row) + 4);
  }
  if (y + size < atlasH) {
    for (let col = 0; col < size; col++) dst.copyWithin(px(x + col, y + size), px(x + col, y + size - 1), px(x + col, y + size - 1) + 4);
  }
  if (x + size < atlasW && y + size < atlasH) dst.copyWithin(px(x + size, y + size), px(x + size - 1, y + size - 1), px(x + size - 1, y + size - 1) + 4);
}
