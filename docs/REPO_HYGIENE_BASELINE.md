# Repo Hygiene Baseline

## Goal

This document defines which categories of files are tracked as source truth, which are
generated, which are evidence by design, and which must never be confused with authoring layers.

## Categories

### Tracked Source

These paths are expected to be versioned as authoring truth:
- `src/`
- `native/opencl/src/`
- `native/opencl/include/`
- `native/opencl/kernels/`
- `native/opencl/test/`
- `native/opencl/driver-pack/build_mesa.sh`
- `native/opencl/driver-pack/pack_driver.sh`
- `native/opencl/driver-pack/manifest.json`
- `scripts/`
- `docs/`

### Tracked Evidence

These files are intentionally committed as evidence:
- `forensics/*.json`
- `forensics/**/*.json`
- `forensics/**/*.jsonl`
- `artifacts/*.json`
- `bench/results/open-test-local-*.json`

Rules:
- Evidence files must be committed only when they represent a meaningful new state.
- Evidence updates should be explainable by runtime or release changes.

### Generated Build Outputs

These paths are generated and should not be used as source truth:
- `native/opencl/build/`
- `native/opencl/build-tq-zero/`
- `native/opencl/driver-pack/out/`
- `native/opencl/driver-pack/out-fresh/`
- `driver/`

Rules:
- These may exist locally for execution.
- They should not drive design or review decisions.
- They should be ignored or scrubbed from source review unless the task is explicitly about generated output validation.

### Imported External Evidence

These paths contain imported or historical context:
- `forensics/imported/`
- `third_party/mesa-adreno/`

Rules:
- Imported evidence is supporting truth, not live implementation.
- Historical recovery intelligence should be preserved, but not mistaken for runtime source.

## Release Hygiene Gates

Before export/build release:
1. `git status` should clearly separate source changes from generated churn.
2. driver-root and driver-pack outputs should be regenerated intentionally, not accumulated passively.
3. new evidence files should correspond to actual replay, probe, or release assertions.
4. mirror sync must be proven with `SYNC_MANIFEST`.

## Review Rule

When reviewing diffs:
- source changes are evaluated for implementation correctness
- evidence changes are evaluated for truthfulness
- generated output changes are ignored unless the task explicitly targets generated outputs
