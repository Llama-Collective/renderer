// The shared renderer pipeline now lives in src/app/scene.ts (so the app-facing layer doesn't depend
// on harness/). This file re-exports it verbatim so every harness page keeps importing from "./scene".
export * from "../src/app/scene";
