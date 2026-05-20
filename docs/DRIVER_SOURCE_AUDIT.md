# Mesa Driver Full Audit

- generated: 2026-05-20T11:13:37.062538+00:00
- mesa upstream head: `388665c442f2f84fb8bab03eb9042a1fc1c6b97a`
- local driver residue sources: `18`
- local driver residue build files: `402`

## Hard conclusions

- `driver/` is not a canonical Mesa source tree; it is mostly build residue plus a narrow set of retained source/header files.
- recovery must use the designated Mesa source checkout as canonical file map and treat `driver/` as narrow source residue only.
- donor repos are useful as `packaging`, `android_stub`, `kgsl patch intent`, and `A8xx bringup` signals, not as primary truth.
- `native/opencl` remains authoritative for TurboQuant runtime, probes, loader behavior, and parity targets.
- the 18 retained `driver/*` sources are generated/carried residue, not simple copies of current upstream `src/*` files.

## Donor roles

### mesa-tu8
- head: `d185f62a24305a2ea0d2fa7f3d156e0595025d54`
- role: `a8xx_turnip_bringup_fork`
- copy policy: `semantic_a8xx_diff_and_ci_evidence_only`
- file count: `12576`
- keyword hits: `freedreno-kmds=kgsl, gallium-rusticl, rusticl, RUSTICL_ENABLE, android_stub, cutils, kgsl, a8xx, a7xx, turnip, opencl, OpenCL, gmem, sysmem, subgroup, wave128, wave64`

### mesa-for-android-container
- head: `dff4db21c0309c790353f449e6c63139d1b36813`
- role: `android_container_distribution_and_android_stub_reference`
- copy policy: `android_stub_packaging_and_build_contract_only`
- file count: `12475`
- keyword hits: `freedreno-kmds=kgsl, gallium-rusticl, rusticl, RUSTICL_ENABLE, android_stub, cutils, kgsl, a8xx, a7xx, turnip, opencl, OpenCL, gmem, sysmem, subgroup, wave128, wave64`

### mesa-freedreno-patchs
- head: `607ffa0719fe58bae2d45defe9fc9e8f4591b0e0`
- role: `legacy_kgsl_patch_inventory`
- copy policy: `patch_inventory_and_intent_only`
- file count: `48`
- keyword hits: `freedreno-kmds=kgsl, kgsl, turnip, gmem`

### mesa-turnip-android-driver
- head: `1af600c6584dadb08f993f0e3aad9bc76de1508f`
- role: `android_app_private_packaging_and_android_stub_patch`
- copy policy: `android_stub_and_loader_patterns_only`
- file count: `37`
- keyword hits: `android_stub, cutils, turnip`

### Mesa-Turnip-Builder
- head: `35176597fcc238a0f1dcce841513b849fcfc2575`
- role: `magisk_and_emulator_build_packaging`
- copy policy: `build_script_patterns_only`
- file count: `35`
- keyword hits: `freedreno-kmds=kgsl, kgsl, turnip`

### Adreno-Tools-Drivers
- head: `50cbd613e7f6f10e6bc36cfde51e9c76c23a441d`
- role: `adrenotools_packaging_and_gen8_patch_bundle`
- copy policy: `inspiration_only_no_blind_copy`
- file count: `49`
- keyword hits: `freedreno-kmds=kgsl, android_stub, cutils, kgsl, a8xx, a7xx, turnip, gmem, sysmem`

### libadrenotools
- head: `8fae8ce254dfc1344527e05301e43f37dea2df80`
- role: `android_app_private_driver_loader_library`
- copy policy: `loader_and_namespace_patterns_only`
- file count: `72`
- keyword hits: `cutils, kgsl, turnip, gmem`

### freedreno_turnip-CI-ilhan
- head: `208e52e62c13e895cc014f1e65f50eb4aa75eab2`
- role: `early_turnip_builder_lineage`
- copy policy: `builder_lineage_and_release_patterns_only`
- file count: `35`
- keyword hits: `freedreno-kmds=kgsl, kgsl, turnip`

### freedreno_turnip-CI-weab
- head: `f4d5a2e615931b7583116d099d55f4f05ff17004`
- role: `adrenotools_builder_with_patch_lane`
- copy policy: `patch_inventory_and_builder_patterns_only`
- file count: `41`
- keyword hits: `freedreno-kmds=kgsl, kgsl, a7xx, turnip, gmem, sysmem`

### freedreno-CI-mrpurple
- head: `ce36aac7cd0f82e8eb837ed53e1c772a50d61371`
- role: `android_turnip_ci_variant`
- copy policy: `ci_patterns_only`
- file count: `33`
- keyword hits: `freedreno-kmds=kgsl, android_stub, kgsl, turnip`

### freedreno-builder-pojav
- head: `fe43b002aa59cc23017efe60a82b76d4a0421e1a`
- role: `pojav_launcher_builder_variant`
- copy policy: `builder_diff_only`
- file count: `36`
- keyword hits: `freedreno-kmds=kgsl, kgsl, turnip`

### mesa-turnip-kgsl-grima
- head: `50ed40fb8ec486490dcbed29680621ec5cb03125`
- role: `turnip_kgsl_chroot_compile_fix_set`
- copy policy: `compile_fix_intent_only`
- file count: `8336`
- keyword hits: `android_stub, cutils, kgsl, turnip, opencl, OpenCL, gmem, sysmem, subgroup, wave64`

### AdrenoToolsDrivers-K11
- head: `278cdc7522e0873fed1e3aa963539a0c374c7862`
- role: `release_discipline_and_variant_catalog`
- copy policy: `release_matrix_and_variant_signals_only`
- file count: `82`
- keyword hits: ``

### old-turnip-base-v3ktor
- head: `fb73b0736ca016d7e41af0ac1ee319b6d9a5bbe3`
- role: `historical_builder_base`
- copy policy: `historical_builder_patterns_only`
- file count: `34`
- keyword hits: `freedreno-kmds=kgsl, kgsl, turnip`

### Adreno-ToolsDriversMagisk-archived
- head: `e9ca6d048f810c3b0ab8a430e63ecc85a3ae1dcb`
- role: `archived_magisk_release_line`
- copy policy: `historical_release_policy_only`
- file count: `31`
- keyword hits: `turnip`

### llama.cpp-opencl
- head: `e6b4acfe86af380c4e631973b9caa14337954423`
- role: `adreno_opencl_compute_backend_reference`
- copy policy: `compute_kernel_memory_queue_patterns_only`
- file count: `2893`
- keyword hits: `opencl, OpenCL, sysmem, subgroup`

### adreno-llms
- head: `887bd7e927e22374c56126fbc03bcb552b418481`
- role: `handwritten_adreno_opencl_kernel_reference`
- copy policy: `kernel_shape_and_scheduler_patterns_only`
- file count: `401`
- keyword hits: `kgsl, opencl, OpenCL, gmem, subgroup`

### termux-packages
- head: `d0078353fba5531e90a51cb2e6e0b669af5d829c`
- role: `termux_android_packaging_and_patch_boundary_reference`
- copy policy: `packaging_and_android_build_boundary_only`
- file count: `9793`
- keyword hits: `gallium-rusticl, rusticl, cutils, kgsl, a8xx, a7xx, turnip, opencl, OpenCL, gmem, sysmem`

## Source-of-truth order

1. designated Mesa upstream checkout
2. `native/opencl`
3. `third_party/mesa-adreno/*matrix*.json` and recovery plans
4. retained source/header files in `driver/`
5. donor repos for semantic hints only
