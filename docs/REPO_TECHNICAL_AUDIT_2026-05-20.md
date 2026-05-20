# Repo Technical Audit 2026-05-20

## Scope

This audit covers the standalone TurboQuant repo as release truth.
It separates live product/runtime code from generated residue, historical evidence, and dead implementation lanes.

## Hard Findings

- `native/opencl` is the only live GPU/runtime implementation surface for the custom stack.
- `native/adreno/` and `native/adreno_cmd/` are legacy implementation lanes and are not referenced by the current package/runtime flow.
- `native/opencl/build/` and `driver/` are generated or carried residue and must not remain in release truth.
- `third_party/mesa-adreno/` is recovery intelligence, not runtime source-of-truth.
- release-facing scripts and wrappers previously contained repo-name and host-path assumptions; these have been reduced to env-driven or repo-derived resolution.

## Required Boundaries

- Authoring truth:
  - `src/`
  - `native/opencl/src/`
  - `native/opencl/include/`
  - `native/opencl/kernels/`
  - `scripts/`
  - release contracts in `docs/`
- Generated or imported:
  - `native/opencl/build/`
  - `native/opencl/runtime-pack/`
  - `native/opencl/driver-pack/out/`
  - `driver/`
  - historical evidence under `forensics/` and `third_party/mesa-adreno/`

## Release Verdict

- `ready_for_clean_export`: true
- `custom_stack_only_runtime`: true
- `legacy_adreno_lane_removed_from_release_truth`: true
- `repo_name_hardcodes_in_live_runtime`: false
- `absolute_host_path_hardcodes_in_live_runtime`: false

## Remaining Intentional Exceptions

- forensic/history artifacts may still contain absolute historical paths as evidence payload
- vendor/system library paths remain in runtime probes as explicit diagnostic fallbacks, not repo hardcodes
