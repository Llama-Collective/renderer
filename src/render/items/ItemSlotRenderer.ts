// Inventory-slot (GUI) item rendering — the EXPOSED form for the simulator UX. RENDERER_PLAN.md §18.
// Vanilla GUI display: generated items render flat head-on, block items at the iconic 30°/225° isometric
// (scale 0.625). This is an ORTHO overlay pass that LOADS the colour target + clears depth, so it draws
// items over the current frame at 2D screen-space slots. It reuses the entity shader/renderer (the item
// geometry packs to the same vertex format) — so block-icon face shading + cutout alpha come for free.
// Future world container-overlays reuse the exact same geometry + a world matrix instead of `slotMatrix`.

import type { GpuBufferHandle, GpuTextureHandle, GraphicsDevice, TextureFormat } from "../../core/GraphicsDevice";
import { BufferUsage } from "../../core/GraphicsDevice";
import { packVertices } from "../../mesh/VertexFormat";
import { mul, scaling, translation, type Mat4 } from "../../mesh/entity/mat4";
import { TerrainPass, type Vec3 } from "../../types";
import { EntityRenderer, type EntityDraw } from "../entities/EntityRenderer";
import { itemDisplayMatrix, type ItemKind } from "./itemDisplay";
import { GUI_ITEM_LIGHT0, GUI_ITEM_LIGHT1 } from "./itemLighting";
import type { ItemGeometry } from "./ItemGeometry";

/** One inventory slot: an item id at a screen-space pixel rect (top-left `x,y`, `size`×`size`). */
export interface ItemSlot {
  item: string;
  x: number;
  y: number;
  size: number;
}

interface ItemMesh {
  vertex: GpuBufferHandle;
  quadCount: number;
  texture: GpuTextureHandle;
  kind: ItemKind;
}

const ORIGIN: Vec3 = [0, 0, 0];

export class ItemSlotRenderer {
  private readonly renderer: EntityRenderer;
  private readonly cache = new Map<string, ItemMesh | null>();

  constructor(
    private readonly device: GraphicsDevice,
    colorFormat: TextureFormat,
    depthFormat: TextureFormat,
    private readonly geometry: ItemGeometry,
  ) {
    this.renderer = new EntityRenderer(device, colorFormat, depthFormat);
  }

  /** Draw the given item slots over the current frame (call after the world render, before present). */
  render(slots: readonly ItemSlot[], width: number, height: number): void {
    const vp = ortho(width, height);
    const draws: EntityDraw[] = [];
    for (const s of slots) {
      const m = this.mesh(s.item);
      if (!m) continue;
      const model = mul(slotMatrix(s.x, s.y, s.size), itemDisplayMatrix(m.kind, "gui"));
      // Block icons get vanilla's diffuse 3D-item lighting (top brightest, lower faces graded); generated
      // sprites stay flat-shaded. The slot matrix flips Y for screen space, so the lights are Y-flipped too.
      const lights = m.kind !== "generated" ? { light0: GUI_ITEM_LIGHT0, light1: GUI_ITEM_LIGHT1 } : {};
      draws.push({ vertex: m.vertex, quadCount: m.quadCount, pass: TerrainPass.Cutout, texture: m.texture, model, sortPos: ORIGIN, ...lights });
    }
    this.renderer.render(draws, vp, ORIGIN, width, height, { depth: 1 }); // ortho overlay: load colour, clear depth
  }

  /** Upload (once) an item's GPU mesh, or null if unrenderable. */
  private mesh(item: string): ItemMesh | null {
    const key = item.replace(/^minecraft:/, "");
    if (!this.cache.has(key)) {
      const g = this.geometry.get(key);
      if (!g || g.verts.length === 0) {
        this.cache.set(key, null);
      } else {
        const data = new Uint8Array(packVertices(g.verts));
        const vertex = this.device.createBuffer({ sizeBytes: data.byteLength, usage: BufferUsage.Vertex, label: `item-slot-${key}` });
        this.device.writeBuffer(vertex, 0, data);
        this.cache.set(key, { vertex, quadCount: g.verts.length / 4, texture: g.texture, kind: g.kind });
      }
    }
    return this.cache.get(key) ?? null;
  }

  dispose(): void {
    for (const m of this.cache.values()) if (m) this.device.destroyBuffer(m.vertex);
    this.cache.clear();
    this.renderer.dispose();
  }
}

/** Screen px → NDC ortho (y-down screen → y-up NDC; item z mapped into [0,1] over a ±128px range). */
function ortho(w: number, h: number): Mat4 {
  return mul(translation(-1, 1, 0.5), scaling(2 / w, -2 / h, -1 / 256));
}

/** A slot's centred [−0.5,0.5] gui-display model → screen px (slot centre, Y flipped, scaled to `size`). */
function slotMatrix(x: number, y: number, size: number): Mat4 {
  return mul(translation(x + size / 2, y + size / 2, 0), scaling(size, -size, size));
}
