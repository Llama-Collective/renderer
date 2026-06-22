// Harness-local entry for the mesher worker, so esbuild keeps every harness bundle flat under dist/
// (a single entry outside harness/ would shift esbuild's outbase and rename ALL bundles). It just runs
// the real worker module for its side effect (installs the onmessage handler).
import "../src/workers/mesher.worker";
