// Typed access to the real model/blockstate fixtures extracted from public/pack.zip.
// (JSON infers arrays as number[], not our tuples → cast via unknown.)

import fixtures from "./blocks.json";
import type { RawBlockModel, RawBlockState } from "../ModelTypes";
import type { RawModelProvider } from "../ModelResolver";

export const fixtureModels = fixtures.models as unknown as Record<string, RawBlockModel>;
export const fixtureBlockstates = fixtures.blockstates as unknown as Record<string, RawBlockState>;
export const fixtureProvider: RawModelProvider = { getModel: (id) => fixtureModels[id] };
