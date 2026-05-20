# Changelog

All notable changes to this project are documented in this file.

The format follows Keep a Changelog and the project uses Semantic Versioning.

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
- Legacy patch residue from clean export truth:
  - `native/opencl/driver-pack/patches/`

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
