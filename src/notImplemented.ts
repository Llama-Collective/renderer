/**
 * Scaffold marker. Every stub method body calls this so the structure type-checks while
 * making it obvious at runtime that nothing is wired up yet. Replace call sites with real
 * logic during the phase that owns them (see RENDERER_PLAN.md §19).
 */
export function notImplemented(what: string): never {
  throw new Error(`[schematic-renderer-webgpu] not implemented yet: ${what}`);
}
