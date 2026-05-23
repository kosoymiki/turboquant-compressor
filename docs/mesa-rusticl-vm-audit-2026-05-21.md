# Mesa Rusticl VM Audit 2026-05-21

This note captures the current Rusticl/KGSL SVM state on the TurboQuant Android stack after probe drift was removed from the repo tooling.

## Confirmed runtime truth

- `mesa_rusticl_kgsl` is the active OpenCL route.
- `SPIR-V` IL is available and working.
- `subgroups` are present via device capability queries.
- Live runtime-pack evidence now reports `hasSvm = true` and `hasSvmCoarse = true`.
- Live runtime-pack evidence still reports `hasSvmFine = false` and `hasSvmAtomics = false`.

Primary live artifact:

- `forensics/opencl-adreno-report.json`

This materially changes the earlier verdict. The current shipped runtime is not "SVM unavailable". It exposes coarse-grain buffer SVM, but not fine-grain/system SVM.

## Rusticl gate semantics

Upstream Rusticl distinguishes two layers:

- `svm_supported()` is true when `system_svm_supported()` is true, or when Gallium VM hooks are available through `screen.is_vm_supported()`.
- `api_svm_supported()` is true when `system_svm_supported()` is true, or when `screen.is_vm_supported()` is true and platform VM initialization succeeded.

In the current upstream source:

- `api/device.rs` reports `CL_DEVICE_SVM_COARSE_GRAIN_BUFFER` when `api_svm_supported()` is true.
- `api/device.rs` reports `CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM` only when `system_svm_supported()` is true.

That matches the live TurboQuant artifact exactly:

- coarse-grain SVM is exposed
- fine-grain/system SVM is not exposed

## Gallium/KGSL VM path

`screen.is_vm_supported()` in Rusticl requires the Gallium hook triad:

- `resource_assign_vma`
- `alloc_vm`
- `free_vm`

Current upstream Mesa source already contains that path:

- `src/gallium/frontends/rusticl/mesa/pipe/screen.rs` gates VM support on the triad
- `src/gallium/drivers/freedreno/freedreno_resource.c` wires `fd_alloc_vm`, `fd_free_vm`, and `fd_resource_assign_vma`
- the hooks are installed when `FD_FEATURE_USE_CPU_MAP` is present
- `src/gallium/drivers/freedreno/freedreno_screen.c` now provides a conservative userspace VM window even when `FD_VA_SIZE` comes back zero

The upstream-source question is therefore closed. The KGSL/Rusticl VM path exists in current source and is compatible with the live coarse-grain SVM result.

## Forensic correction

The previous framing was wrong in one specific way:

- it treated missing fine-grain/system SVM as if all API-level SVM were absent

What the live evidence now supports is narrower and cleaner:

- coarse-grain buffer SVM is available on the active Rusticl/KGSL stack
- fine-grain buffer SVM is not available
- fine-grain system SVM is not available
- SVM atomics are not available

That is an expected shape for a Gallium VM-backed path without `system_svm`.

## Remaining engineering target

The next meaningful target is no longer "prove SVM exists at all". That question is already answered.

The remaining target is to keep all release claims aligned with the actual capability tier:

- `mesa_rusticl_kgsl` with coarse-grain SVM
- not fine-grain/system SVM
- not SVM atomics

If deeper Mesa work continues, the relevant frontier is not the old hook triad existence claim. It is whether a future kernel/driver path can legitimately expose stronger `system_svm` semantics without overstating what the current runtime does.

## Trace contract

For future closure work, `system_svm` is not allowed to move by claim alone.

Required forensic bundle:

- `RUSTICL_DEBUG=memory` traces from `core/platform.rs`
- per-device record of:
  - `system_svm_supported()`
  - VM hook triad presence
  - VMA window
- live artifact confirming `CL_DEVICE_SVM_*` flags on the active route

Canonical contract:

- `docs/frontier-tracing-contract-2026-05-21.md`
