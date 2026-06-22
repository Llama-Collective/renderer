// Schematic parsing for the web page, backed by the `nucleation` WASM library (a high-performance
// Rust parser). Replaces the hand-rolled NBT reader that only understood Sponge v2 — nucleation
// auto-detects and reads Sponge v2/v3 `.schem`, `.litematic`, `.mcstructure`, raw region data, etc.
//
// `SchematicWrapper.blocks()` already returns `{ x, y, z, name, properties }` (i.e. EditableBlock),
// so the conversion to the renderer's `EditableSchematic` is a thin remap. The WASM module is
// initialised once (it fetches an ~8 MB `.wasm`, which Vite emits as an asset).

import init, { SchematicWrapper } from "nucleation";
import type { EditableSchematic, EditableBlock, EditableEntity, Vec3Tuple } from "../../schematic-io/types";

let initPromise: Promise<unknown> | null = null;

/** Kick off (or join) the one-time WASM initialisation. Safe to call eagerly to overlap the
 *  download with other work; `parseSchematic` awaits it anyway. */
export function initParser(): Promise<unknown> {
  if (!initPromise) initPromise = init();
  return initPromise;
}

/** A single mobile entity as nucleation hands it back (`id` is the entity TYPE, e.g. "minecraft:sheep"). */
interface NucleationEntity {
  id: string;
  position: [number, number, number];
  nbt?: string;
}

/**
 * Parse raw schematic bytes into the renderer's `EditableSchematic`. `hint` (a filename or URL) lets
 * nucleation pick the format by extension; otherwise it auto-detects from the bytes.
 */
export async function parseSchematic(
  bytes: Uint8Array,
  name: string,
  hint?: string,
): Promise<EditableSchematic> {
  await initParser();

  const wrapper = new SchematicWrapper();
  try {
    const ext = (hint ?? "").toLowerCase();
    if (ext.endsWith(".litematic")) wrapper.from_litematic(bytes);
    else if (ext.endsWith(".schem") || ext.endsWith(".schematic")) wrapper.from_schematic(bytes);
    else if (ext.endsWith(".mcstructure")) wrapper.from_mcstructure(bytes);
    else wrapper.from_data(bytes); // auto-detect by magic/structure

    const [width, height, length] = wrapper.get_dimensions();

    // `blocks()` returns plain JS objects `{ x, y, z, name, properties }` (non-air only) — already
    // EditableBlock-shaped, copied out of WASM memory, so it survives `wrapper.free()`.
    const blocks = wrapper.blocks() as EditableBlock[];

    // Mobile entities: nucleation's `id` is the entity TYPE; synthesise a unique EditableEntity `id`
    // and carry the full NBT (JSON string) verbatim for the renderer's typed views.
    const rawEntities = (wrapper.get_entities() ?? []) as NucleationEntity[];
    const entities: EditableEntity[] = rawEntities.map((e, i) => ({
      id: `${e.id}-${i}`,
      type: e.id,
      position: [e.position[0], e.position[1], e.position[2]] as Vec3Tuple,
      properties: {},
      nbt: typeof e.nbt === "string" ? e.nbt : undefined,
    }));

    return {
      name: wrapper.getName() || name,
      size: [width, height, length],
      blocks,
      entities,
    };
  } finally {
    wrapper.free(); // release the WASM-side schematic (block/entity data is already copied out)
  }
}
