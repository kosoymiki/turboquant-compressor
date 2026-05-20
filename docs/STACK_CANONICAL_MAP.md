# Stack Canonical Map

## Goal

This document defines the canonical source-of-truth boundaries for the standalone TurboQuant repo.
It exists to prevent drift between executable code, tracked driver artifacts, installed driver-root state, forensic evidence,
and the corpus mirror.

## Canonical Layers

### 1. Product Surface

Purpose: public TurboQuant logic unrelated to Android driver recovery.

Canonical paths:
- `src/core/`
- `src/tools/`
- `src/native/`
- `src/runtime/`
- `src/server.ts`

Rules:
- This layer defines product APIs, compression/search behavior, MCP tools, and release behavior.
- It must not depend on generated driver-pack outputs as source truth.
- It may consume runtime evidence, but does not own it.

### 2. Custom GPU Runtime Stack

Purpose: executable implementation of the repo-local Android GPU stack.

Canonical paths:
- `native/opencl/src/`
- `native/opencl/include/`
- `native/opencl/kernels/`
- `native/opencl/test/`
- `native/opencl/driver-pack/build_mesa.sh`
- `native/opencl/driver-pack/pack_driver.sh`
- `native/opencl/driver-pack/manifest.json`
- `native/opencl/driver-pack/tq-driver-pack-adreno-a7xx-a8xx.tar.zst`

Rules:
- This is the only authoritative source for Rusticl/Freedreno/KGSL/Turnip runtime behavior in this repo.
- Historical patch-centric recovery residue has been removed from tracked release truth.
- Any runtime claim must be explainable from these paths first.

### 3. Installed Driver Runtime

Purpose: assembled, executable stack for safe on-device replay.

Canonical paths:
- `$TQ_DRIVER_ROOT/env.sh`
- `$TQ_DRIVER_ROOT/layer1-compute/`
- `$TQ_DRIVER_ROOT/layer2-vulkan/`
- `$TQ_DRIVER_ROOT/kernels/`
- `$TQ_DRIVER_ROOT/meta/manifest.json`
- `$TQ_DRIVER_ROOT/meta/dependencies.txt`

Rules:
- This layer is generated from the Custom GPU Runtime Stack.
- It is not the authoring surface.
- It is the execution surface used by safe replay and export validation.
- Repo-local `native/opencl/driver-root/` is installed from the tracked driver archive.

### 4. Forensic Evidence

Purpose: evidence and truth claims about historical and current runtime behavior.

Canonical paths:
- `forensics/RELEASE_EVIDENCE_MANIFEST.json`
- `forensics/opencl-adreno-report.json`
- `forensics/adreno/loader-report.json`
- `forensics/mesa/driver-mesh-recovery-ready.json`
- `forensics/mcp-stdio-transcript.jsonl`
- `artifacts/*.json`
- `bench/results/open-test-local-*.json`

Rules:
- These artifacts are evidence, not implementation.
- They may justify or falsify runtime claims.
- They must never silently replace source truth.

### 5. Historical Recovery Intelligence

Purpose: preserved recovery context for Mesa/driver reconstruction.

Canonical paths:
- `third_party/mesa-adreno/manifest.lock.json`
- `third_party/mesa-adreno/proof-matrix.json`
- `third_party/mesa-adreno/build-matrix.json`
- `third_party/mesa-adreno/reconstruction-plan.json`
- `third_party/mesa-adreno/file-to-file-map.json`
- `third_party/mesa-adreno/applicability-report.json`

Rules:
- This layer is an evidence graph and reconstruction aid.
- It is not a live source layer for runtime behavior.
- It exists to explain how the current stack was recovered and should evolve.

### 6. Corpus Mirror

Purpose: mirrored release-grade copy for corpus-backed retrieval and operational continuity.

Canonical mirror root:
- sibling mirror repo containing `mcp/turboquant`

Rules:
- Mirror files must match the standalone source on critical stack paths.
- Mirror is not allowed to diverge semantically from standalone.
- Sync status must be proven by `SYNC_MANIFEST`, not assumed.

## Non-Canonical / Generated Zones

Generated and non-authoring zones:
- `native/opencl/build/`
- `native/opencl/build-tq-zero/`
- `native/opencl/driver-pack/out/`
- `native/opencl/driver-pack/out-fresh/`
- `driver/`

Rules:
- These are execution byproducts or imported residue.
- They must not be treated as authoring truth.

## Release Rule

A release is considered structurally valid only when:
- source truth is modified only in canonical authoring layers
- driver artifact is regenerated from canonical stack sources
- driver-root is installed from the tracked driver artifact
- forensic evidence is refreshed intentionally
- mirror parity is proven by `SYNC_MANIFEST`
