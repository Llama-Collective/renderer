import { defineConfig } from "vitest/config";

// Tests are first-class in this package (RENDERER_PLAN.md §22). The data-model and
// Stable-Presentation-Invariant tests (T-PRES-1..5) must be green before any rendering
// work proceeds. Wire this into the root test runner once the first tests land.
export default defineConfig({
  test: {
    include: ["src/**/*.test.ts"],
  },
});
