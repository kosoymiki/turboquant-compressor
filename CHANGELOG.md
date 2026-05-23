
## v4.5.0 (2026-05-22) - Full C++ Migration Ready

### Breaking Changes
- None (stable release)

### Features
- Full corpus integration with web-search at every step
- 17 MCP tools: 13 core TQ + 4 corpus
- C++ native runtime for Adreno/KGSL/Mesa
- Rusticl + Turnip dual compute path
- P0-P3 all closed: 0 unresolved, plan.queue empty

### Bug Fixes
- Fixed FTS query escaping for special chars
- Removed debug artifacts from dist/src

### Performance
- Vector compression: 2/3/4/8 bit support
- KV cache analysis integrated
- Context pack search with FTS

### Docs
- Full corpus docs in corpus-control-plane/
- Driver forensics fully integrated
- Restart-persistent debug protocol

# Changelog

All notable changes to this project are documented in this file.

The format follows Keep a Changelog and the project uses Semantic Versioning.

## [4.1.4] - 2026-05-21

### Changed
- Promoted the shipped public compression/search contract from `LEVEL_1_PUBLIC_BETA` to `LEVEL_1_PUBLIC` across code, tests, open benchmarks, and release-facing documentation.
- Refreshed the shipped open local benchmark to `bench/results/open-test-local-20260521-160651.json`, preserving the contextual lexical public path and lifting committed compression evidence to `7.9304x`.
- Corrected Mesa/Rusticl runtime forensics to reflect live coarse-grain SVM support on `mesa_rusticl_kgsl`, while keeping fine-grain/system SVM claims disabled.
- Runtime-pack OpenCL probes now treat the isolated custom stack as canonical and no longer derive verdicts from ambient host `clinfo`.
- Project release identity bumped to `v4.1.4`.

## [4.1.2] - 2026-05-21

### Changed
- Removed vendored `native/opencl/mesa-source` overlay from the repo release surface.
- Removed legacy `native/opencl/driver-pack/patches/` from the repo release surface.
- Switched the tracked driver release contract to `native/opencl/driver-pack/tq-driver-pack-adreno-a7xx-a8xx.tar.zst`, with `native/opencl/driver-root/` regenerated at install/runtime.
- Simplified Mesa build source resolution to explicit upstream base input instead of repo-local source overlays.
- Public retrieval path now ships `hashed_tfidf` plus `turboquant_beta` codebook metadata in format v3.
- Kernel/runtime surface now derives subgroup/fp16 specialization from the real OpenCL route and no longer carries dead fused-attention fp16 lanes in the shipped tree.
- Added file-based regression coverage for shipped `turboquant_beta` defaults and context-pack build provenance.
- Refreshed the committed open local benchmark artifact to `bench/results/open-test-local-20260521-095918.json`.
- Project release identity bumped to `v4.1.2`.

## [4.1.0] - 2026-05-20

### Added
- Canonical release contracts:
  - `STACK_CANONICAL_MAP`
  - `REPO_HYGIENE_BASELINE`
  - `RUNTIME_EVIDENCE_CHAIN`
  - `EXPORT_CHECKLIST`
  - `EXPORT_VERDICT`
- `package:release-slice` export path for clean release artifacts
- Real OpenCL self-tests for:
  - `mse_score`
  - `qjl_score`
  - `value_dequant`
  - `fused_attention`
- Mirror sync manifest for critical stack files

### Changed
- Project release identity bumped to `v4.1.0`
- Release-facing forensics now reflect repo-local custom stack safe-run truth
- README and install docs now describe the clean release-slice path
- Backend truth classification distinguishes custom stack readiness from vendor OpenCL

### Removed
- Dead release lanes from the clean export path:
  - `native/adreno/`
  - `native/adreno_cmd/`
  - `driver/`
  - `third_party/mesa-adreno/` from minimal clean export
- Unused fused kernel variants from clean export truth:
  - `tq_fused_attention_v2.cl`
  - `tq_fused_attention_v3_fp16.cl`
  - `tq_fused_attention_v4_fp16.cl`
  - `tq_fused_attention_v5_fp16.cl`
- Historical patch-centric driver residue removed from tracked release truth.

## [4.0.1] - 2026-05-20

### Added
- Termux-first MCP server packaging
- 13-tool MCP surface
- Local compressed vector database build/search
- Context-pack build/search
- Cache/cost analysis tools
- KV/cache analysis tool
- Backend/OpenCL/Adreno probe tools
- Release verification and evidence gates
- Public local benchmark artifacts showing 5.5x+ compression
