# Export Verdict

## Status

The standalone TurboQuant repo is ready for a clean export slice.

## What goes out

Only these categories belong in the export set:
- canonical source-of-truth for the custom stack
- release contracts
- committed runtime/forensic evidence
- optional historical recovery intelligence

## What stays out

These zones are explicitly excluded from export truth:
- `native/opencl/build/`
- `native/opencl/build-tq-zero/`
- `native/opencl/driver-pack/out/`
- `native/opencl/driver-pack/out-fresh/`
- `native/adreno/`
- `native/adreno_cmd/`
- `driver/`
- `third_party/mesa-adreno/`
- `dist/`
- `node_modules/`

## Engineering Verdict

The current system is no longer blocked by runtime bring-up.
The main release risk was truth drift between:
- executable code
- native/opencl/driver-pack/tq-driver-pack-adreno-a7xx-a8xx.tar.zst
- installed native/opencl/driver-root state
- forensic evidence
- mirror copy

That drift is now bounded by:
- `STACK_CANONICAL_MAP`
- `REPO_HYGIENE_BASELINE`
- `RUNTIME_EVIDENCE_CHAIN`
- `SYNC_MANIFEST`
- `EXPORT_MANIFEST`

## Final Read

This is not a “fully cleaned repo” in the absolute sense.
It is a **clean release slice**:
- canonical
- reproducible
- mirror-synchronized
- evidence-backed
- generated-noise-aware
