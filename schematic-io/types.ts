export type Vec3Tuple = [number, number, number];

/** One side of a sign: 4 text lines + dye color + glow (mirrors the sim's SignFace). */
export interface SignFaceText {
	messages: [string, string, string, string];
	color: string;
	hasGlowingText: boolean;
}

/** Editable sign text carried on a sign block so the renderer can bake it. */
export interface SignTextData {
	front: SignFaceText;
	back: SignFaceText;
	isWaxed: boolean;
}

/**
 * A block entity's full NBT, carried losslessly as a typed SNBT string. Typed
 * convenience views (e.g. {@link SignTextData}, inventories) are layered on top
 * for the properties the editor/sim actively manipulate; everything else
 * round-trips verbatim via `snbt`.
 */
export interface BlockEntityNbt {
	/** The block-entity type id (e.g. `minecraft:chest`, `minecraft:sign`). */
	id: string;
	/** The block entity's inner NBT as a typed SNBT string (no `Id`/`Pos`). */
	snbt: string;
}

export interface EditableBlockState {
	name: string;
	properties: Record<string, string>;
	/** Sign text, present only on sign blocks (the renderer bakes it onto the board). */
	signText?: SignTextData;
	/**
	 * Full block-entity NBT (container contents, machine state, banner patterns,
	 * item components, …) as a lossless SNBT carrier, present when the block has a
	 * block entity. Preserved verbatim across load/save even when the editor does
	 * not model it; the typed views (signText, inventories) are the "exceptions"
	 * layered on top.
	 */
	blockEntity?: BlockEntityNbt;
}

export interface EditableBlock extends EditableBlockState {
	x: number;
	y: number;
	z: number;
}

export interface EditableEntity {
	id: string;
	type: string;
	position: Vec3Tuple;
	velocity?: Vec3Tuple;
	properties?: Record<string, string>;
	/**
	 * Full entity NBT (health, inventory, variant data, passengers, …) carried
	 * losslessly as an opaque NBT-JSON string from Nucleation. Preserved verbatim
	 * across load/save; typed views are layered on top for modeled entities.
	 */
	nbt?: string;
}

export interface EditableSchematic {
	name: string;
	size: Vec3Tuple;
	blocks: EditableBlock[];
	entities?: EditableEntity[];
	metadata?: Record<string, unknown>;
}

export interface PickResult {
	type: "block" | "entity" | "ground";
	position: Vec3Tuple;
	normal: Vec3Tuple;
	block?: EditableBlock;
	entity?: EditableEntity;
	/** World-space hit point on the picked surface (for orientation placement). */
	hitPoint?: Vec3Tuple;
}

/** Sub-region of a clicked face, used to choose placement orientation.
 * "centermost" is the innermost square (faces into the surface); "center" is the
 * ring around it (faces back out toward the player); the four edges face sideways. */
export type FaceRegion = "centermost" | "center" | "up" | "down" | "left" | "right";

export interface PlacementHint {
	/** The block position the new block will occupy. */
	position: Vec3Tuple;
	/** Direction of the clicked face (normal of the surface, pointing outward). */
	clickedFace: Vec3Tuple;
	/** Fractional hit inside the clicked block, each component in [0,1). */
	hit: Vec3Tuple;
	/** Which region of the clicked face was hit. */
	region: FaceRegion;
	/** Resolved horizontal facing for the placed block (world direction name). */
	facing: "north" | "south" | "east" | "west" | "up" | "down";
}

export interface EditorStats {
	blocks: number;
	entities: number;
	size: Vec3Tuple;
	renderer?: Record<string, number>;
}
