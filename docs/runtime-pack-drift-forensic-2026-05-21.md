# Runtime Pack Drift Forensic — 2026-05-21

## Scope

Determine why the repo-level OpenCL probes and production policy could disagree with the known runtime route `mesa_rusticl_kgsl`.

## Live findings

- Upstream Mesa source in `/data/data/com.termux/files/home/mesa-upstream-26.2-devel` already contains the KGSL/Rusticl VM path:
  - `src/freedreno/drm/kgsl/kgsl_pipe.c` exports `FD_VA_SIZE`
  - `src/gallium/drivers/freedreno/freedreno_resource.c` wires `alloc_vm`, `free_vm`, `resource_assign_vma`
  - `src/gallium/frontends/rusticl/mesa/pipe/screen.rs` gates VM-backed SVM on that triad
- The assembled custom driver stack exists and is installed under `native/opencl/driver-root/`.
- `scripts/safe-runtime-pack-run.sh probe` with the installed driver root reports a Rusticl/Freedreno device:
  - device name `FD725`
  - version `OpenCL 3.1`
  - `recommendedBackend = mesa_rusticl_kgsl`
- the current live probe also reports:
  - `hasSvm = true`
  - `hasSvmCoarse = true`
  - `hasSvmFine = false`
  - `hasSvmAtomics = false`
- The TypeScript deep probe had been using ambient `clinfo`, which observed the host/vendor OpenCL path instead of the runtime-pack-isolated Rusticl path.
- `src/native/production_policy.ts` pointed at nonexistent evidence files:
  - `forensics/mesa/rusticl-ready.json`
  - `forensics/mesa/turnip-ready.json`

## Root cause

The dominant drift was not Mesa runtime regression. It was repo-side forensic drift:

1. `src/native/opencl_probe.ts` classified deep OpenCL readiness from ambient `clinfo` and reduced success to `opencl_adreno` or `opencl_generic`.
2. That bypassed the runtime-pack launcher that sets `OCL_ICD_VENDORS` and `LD_LIBRARY_PATH` for Rusticl.
3. `src/native/production_policy.ts` used stale evidence paths, so production readiness could degrade to `diagnostic_only` despite current repo evidence.

## Fixes applied

- `src/native/opencl_probe.ts`
  - now runs `scripts/safe-runtime-pack-run.sh probe` when available
  - treats a successful runtime-pack probe as canonical for custom-stack detection
  - can now return `recommendedBackend = mesa_rusticl_kgsl`
- `src/native/production_policy.ts`
  - now resolves production evidence from live repo artifacts, primarily `forensics/mesa/driver-mesh-recovery-ready.json` and `forensics/opencl-adreno-report.json`
- `src/tools/turboquant_adreno_loader_probe.ts`
  - now reports `READY` as custom-stack-ready when the probe returns `mesa_rusticl_kgsl`
- `scripts/verify-adreno-loader.mjs`
  - timeout increased to accommodate the slower runtime-pack probe path

## Current conclusion

- The current unresolved issue is not a proven absence of KGSL VM hooks in source.
- The previously observed mismatch was explained by probe/evidence drift in repo tooling.
- Any remaining SVM investigation should now be based on runtime-pack probe output and Rusticl live traces, not ambient `clinfo`.
- The surviving capability question is tiering, not existence: live runtime currently supports coarse-grain SVM, not fine-grain/system SVM.
