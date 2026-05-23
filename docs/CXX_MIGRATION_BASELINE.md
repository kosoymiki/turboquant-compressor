# C++ Migration Baseline

## Goal

This document defines the honest starting point for moving the standalone TurboQuant product toward a `C++`-centric base without collapsing corpus evidence, release gating, or Android device-backed runtime validation.

The target is not "delete TypeScript immediately". The target is:

- `C++` owns executable compression/runtime kernels and OpenCL/Vulkan/driver truth
- JavaScript/TypeScript becomes a thin integration shell for MCP/API bindings
- corpus-control-plane remains a separate operational/evidence product
- device-backed ADB/forensics stays explicit and external to product runtime truth

## Current Reality

### Already Native / Candidate Core

Canonical `C++`-centric surface that already exists:

- `native/opencl/src/`
- `native/opencl/include/`
- `native/opencl/test/`
- `native/opencl/kernels/`
- `native/opencl/CMakeLists.txt`

This layer already owns:

- OpenCL loader/runtime selection
- staged Rusticl/Freedreno/KGSL execution
- kernel execution and parity tests
- frontier smoke / probe surfaces through `tq_opencl_cli`

### Current JS/TS-First Product Path

The public product path is still primarily JS/TS-first:

- `src/core/`
- `src/tools/`
- `src/runtime/`
- `src/server.ts`
- `src/native/wasm_backend.ts`

This layer still owns too much product truth:

- MCP tool exposure
- runtime/backend admission
- hybrid CPU/WASM/GPU orchestration
- evidence interpretation used by release gates

### Corpus / Evidence Coupling Still Inside Product Repo

These paths show that product behavior still reads corpus/evidence state directly:

- `src/runtime/hybrid_training_runtime.ts`
- `src/native/production_policy.ts`
- `scripts/generate-opencl-capability-evidence.mjs`
- `scripts/collect-adb-mesa-forensics.mjs`
- `forensics/`

That coupling is the main blocker for a clean `C++`-centric base. Native runtime truth and corpus evidence are still adjacent enough that the repo behaves as if evidence were part of runtime state.

## Product Boundary After Migration

### Must Move Toward Native Core

These responsibilities should become `C++`-owned or `C++`-first:

- runtime probes and capability classification inputs
- compression/search hot paths that currently depend on JS/WASM execution
- kernel cache and autotune execution logic
- device/runtime route admission based on native observations
- stable CLI entrypoints for probe/smoke/runtime operations

### Can Stay as Thin JS/TS Shell

These can remain in JS/TS if they stop owning runtime truth:

- MCP server transport
- JSON schema validation
- request/response shaping
- archive/release wrappers
- optional fallback adapters when native binary is unavailable

### Must Leave the Product Truth Path

These responsibilities must remain outside the primary product runtime:

- ADB collector orchestration
- `run-as` forensic staging
- long-lived donor corpus ingestion
- release evidence backfilling
- historical Mesa reconstruction intelligence

Those belong to corpus-control-plane or explicit evidence-generation scripts, not to the product runtime itself.

## Immediate Blockers

### 1. Forensics as Runtime Dependency

`src/runtime/hybrid_training_runtime.ts` currently reads:

- `forensics/adreno/opencl-inference-runtime-contract.json`
- `forensics/adreno/opencl-paged-kv-prefix-cache-contract.json`
- `forensics/precision-calibration-runtime-state.json`

That is acceptable for evidence collection, but not for a final `C++`-centric product base. Runtime readiness should come from native probe/contract APIs, with evidence generation as a consumer.

### 2. WASM as Product Math Lane

`src/native/wasm_backend.ts` is still a real execution lane, not just a compatibility shim. If the product claims a `C++`-centric base, WASM should either:

- become a compatibility/fallback layer, or
- be regenerated from the same native core contract rather than treated as a peer runtime plane

### 3. Release Truth Still Depends on Repo-Local Evidence Files

Release and capability scripts still derive truth from repo-local `forensics/*`. That must be inverted:

- native runtime emits machine-readable truth
- evidence scripts collect and freeze it
- release validation consumes explicit artifacts, not implicit filesystem drift

## ADB / Device-Backed Contract

ADB is valid and should remain part of the migration base, but only as an external validation lane.

Current working assumptions:

- live endpoint is supplied by `TQ_ADB_SERIAL` or `ADB_SERIAL`
- `run-as` package is supplied by `TQ_ADB_RUN_AS_PACKAGE`
- staged runtime is replayed via explicit scripts under `scripts/`
- evidence output should be explicit, not inferred from whichever `forensics/opencl-adb-*.json` is newest

This means ADB helps validate the native base; it must not become the hidden source of product truth.

## Target Shape

### Native Base

- `native/core/` or expanded `native/opencl/` becomes the primary runtime/compression engine
- one stable native CLI or shared library exposes probe, smoke, search, quantization, and runtime contracts
- capability classification inputs originate from native output

### Thin Shell

- `src/server.ts` and MCP bindings call native interfaces
- JS/TS keeps protocol glue only
- WASM remains optional fallback, not primary execution truth

### Corpus Plane

- corpus-control-plane owns donor ingestion, ADB deep forensics, release-evidence campaigns, and historical recovery archives
- standalone repo consumes only explicit exported artifacts from that plane

## Next Implementation Steps

1. Replace runtime reads of `forensics/*` with explicit generated contract inputs or native probe outputs.
2. Define one canonical native interface for probe/search/quantize/runtime admission.
3. Move JS/TS hybrid runtime logic from "runtime owner" to "orchestration wrapper".
4. Keep ADB collector scripts, but require explicit `TQ_ADB_SERIAL` and explicit evidence artifact selection.
5. Split release validation into product-native truth vs corpus-evidence truth.
