# System SVM Donor Corpus 2026-05-22

This corpus layer pins the donor base for the active `mesa_system_svm` campaign.

It is not a generic bookmark list. It is the implementation map for the exact lane:

- `CL_DEVICE_SVM_FINE_GRAIN_SYSTEM`
- `CL_DEVICE_SVM_ATOMICS`
- `clSetKernelExecInfo(..., CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM, ...)`
- raw-pointer materialization for indirect kernel access

## Canonical Intake

```sh
cd /data/data/com.termux/files/home/tmp_turboquant
node scripts/ingest-system-svm-donor-corpus.mjs
```

It writes:

- [system-svm-donor-corpus-ingest-2026-05-22.json](/data/data/com.termux/files/home/tmp_turboquant/forensics/system-svm-donor-corpus-ingest-2026-05-22.json)
- [mesa-local-system-svm-system-svm-corpus-manifest-2026-05-22.json](/data/data/com.termux/files/home/tmp_turboquant/forensics/mesa-local-system-svm-system-svm-corpus-manifest-2026-05-22.json)
- [pocl-system-svm-system-svm-corpus-manifest-2026-05-22.json](/data/data/com.termux/files/home/tmp_turboquant/forensics/pocl-system-svm-system-svm-corpus-manifest-2026-05-22.json)
- [clvk-svm-reference-system-svm-corpus-manifest-2026-05-22.json](/data/data/com.termux/files/home/tmp_turboquant/forensics/clvk-svm-reference-system-svm-corpus-manifest-2026-05-22.json)

## Donor Roles

`mesa-local-system-svm`
- active implementation lane
- Rusticl `clSetKernelExecInfo` gate
- Rusticl context/kernel materialization path
- Freedreno/KGSL userptr donor path

`pocl-system-svm`
- strongest external donor for OpenCL runtime semantics
- explicit `clSetKernelExecInfo` device hook contract
- indirect/raw pointer tracking and migration logic
- LLVM-side SVM pointer offset donor (`SVMOffset`)

`clvk-svm-reference`
- negative donor for this lane
- useful to prove what is absent or stubbed
- not a primary implementation donor for fine/system SVM

## Engineering Verdict

Current donor hierarchy for `mesa_system_svm`:

1. local Mesa Rusticl/Freedreno/KGSL path
2. PoCL runtime/device-path donors
3. clvk only as contrast/reference, not as primary donor

## Rule

Any future `system_svm` work must preserve:

- exact donor revision
- exact entry files
- explicit donor role
- whether the donor is positive, negative, or contrast-only
