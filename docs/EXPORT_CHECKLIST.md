# Export Checklist

## Goal

Prepare a clean release/export slice without confusing generated local execution residue with source truth.

## Required Contracts

- `docs/STACK_CANONICAL_MAP.md`
- `docs/REPO_HYGIENE_BASELINE.md`
- `docs/RUNTIME_EVIDENCE_CHAIN.md`
- `artifacts/runtime_evidence_chain_release_ready_2026-05-20.json`
- `artifacts/sync_manifest_release_ready_2026-05-20.json`
- `artifacts/release_slice_status_2026-05-20.json`

## Required Runtime Evidence

- `forensics/RELEASE_EVIDENCE_MANIFEST.json`
- `forensics/adreno/loader-report.json`
- `forensics/opencl-adreno-report.json`
- `forensics/mcp-stdio-transcript.jsonl`
- `forensics/mesa/driver-mesh-recovery-ready.json`
- `artifacts/seamless_probe_post_boundary.json`
- `artifacts/safe_benchmark_2026-05-20.json`

## Required Source-of-Truth Files

- `native/opencl/src/*` on the custom stack hot path
- `native/opencl/include/*` relevant public headers
- `native/opencl/driver-pack/build_mesa.sh`
- `native/opencl/driver-pack/pack_driver.sh`
- `native/opencl/driver-pack/manifest.json`
- `scripts/safe-runtime-pack-run.sh`
- `scripts/benchmark-opencl-adreno.mjs`
- `scripts/verify-release.mjs`
- `src/native/wasm_backend.ts`

## Required Mirror Condition

- `artifacts/sync_manifest_release_ready_2026-05-20.json`
- `all_match = true`

## Not Source-Of-Truth

These may exist locally and are valid for local execution, but should not be used as authoring truth:

- `native/opencl/build/`
- `native/opencl/build-tq-zero/`
- `native/opencl/runtime-pack/`
- `native/opencl/runtime-pack-fresh/`
- `native/opencl/driver-pack/out/`
- `native/opencl/driver-pack/out-fresh/`
- `driver/`

## Final Rule

Release/export is structurally ready when:
- source truth is canonical
- runtime evidence chain is complete
- sync manifest is green
- generated noise is explicitly treated as non-authoring residue

## Export Path

Preferred clean export command:
- `npm run package:release-slice`

This path packages only:
- canonical source-of-truth
- release contracts
- committed evidence

It intentionally excludes:
- generated build residue
- legacy/dead implementation lanes
- historical recovery intelligence
