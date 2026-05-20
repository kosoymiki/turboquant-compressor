# Mesa Driver File-to-File Map

- generated: 2026-05-20T10:57:55.605700+00:00
- mesa head: `388665c442f2f84fb8bab03eb9042a1fc1c6b97a`

## kgsl_backend_core

- goal: recover or validate the userspace KGSL backend surfaces used by Gallium/Freedreno and Turnip on Android

### Upstream
- `src/freedreno/drm/meson.build`
- `src/freedreno/vulkan/tu_knl_kgsl.cc`

### Missing upstream surfaces (must be rebuilt semantically/out-of-tree)
- `src/freedreno/drm/kgsl/kgsl_bo.c`
- `src/freedreno/drm/kgsl/kgsl_device.c`
- `src/freedreno/drm/kgsl/kgsl_pipe.c`
- `src/freedreno/drm/kgsl/kgsl_priv.h`
- `src/freedreno/drm/kgsl/kgsl_ringbuffer_sp.c`

### TurboQuant native
- `native/opencl/src/tq_driver_loader.cpp`
- `native/opencl/src/tq_opencl_probe.cpp`
- `native/opencl/src/tq_vk_probe.cpp`

### Historical evidence
- `raw/forensics/android_evidence_pack_20260515/mesa-turnip-build-evidence.json`
- `raw/forensics/android_evidence_pack_20260515/rusticl-build-evidence.json`
- `raw/forensics/android_evidence_pack_20260515/patch-42R-final-gpu-submit.json`
- `raw/forensics/android_evidence_pack_20260515/patch-43R-decision-gate-close.json`

### Donor hints
- `raw/repos/mesa-driver-donors/mesa-freedreno-patchs/freedreno-kgsl-patchs-for-24.2.8/0011-freedreno-drm-kgsl-Add-KGSL-backend-for-freedreno.patch`
- `raw/repos/mesa-driver-donors/mesa-freedreno-patchs/freedreno-kgsl-patchs-for-24.2.8/0012-freedreno-drm-Add-more-APIs-to-per-backend-API.patch`

### Proof targets
- `forensics/adb-driver-truth.json`
- `forensics/mesa/driver-mesh-recovery-ready.json`
- `forensics/opencl-adreno-report.json`

## pipe_loader_and_driver_selection

- goal: rebuild deterministic driver selection and probe order without reviving stale legacy loader logic

### Upstream
- `src/gallium/auxiliary/pipe-loader/pipe_loader_drm.c`
- `src/gallium/auxiliary/target-helpers/drm_helper.h`
- `src/gallium/targets/dril/dril_target.c`
- `meson.build`
- `meson.options`

### TurboQuant native
- `native/opencl/src/tq_driver_loader.cpp`
- `native/opencl/src/tq_driver_manifest.cpp`
- `native/opencl/include/tq_driver_loader.h`

### Historical evidence
- `raw/forensics/android_evidence_pack_20260515/kgsl-pipe-loader-plan.json`
- `raw/forensics/android_evidence_pack_20260515/loader-report.json`

### Donor hints
- `raw/repos/mesa-driver-donors/mesa-freedreno-patchs/freedreno-kgsl-patchs-for-24.2.8/0015-termux-x11-kgsl.patch`
- `raw/repos/mesa-driver-donors/mesa-for-android-container/meson.build`

### Proof targets
- `third_party/mesa-adreno/reconstruction-plan.json`
- `third_party/mesa-adreno/proof-matrix.json`

## android_stub_and_cutils_surface

- goal: restore Android stub and cutils compatibility without brittle duplicated stub hacks

### Upstream
- `src/android_stub/meson.build`
- `src/android_stub/hardware_stub.cpp`
- `src/android_stub/log_stub.cpp`
- `src/android_stub/sync_stub.cpp`
- `include/android_stub/cutils/native_handle.h`
- `include/android_stub/vndk/hardware_buffer.h`
- `include/android_stub/android/hardware_buffer.h`

### TurboQuant native
- `native/opencl/driver-pack/manifest.json`
- `native/opencl/driver-pack/build_mesa.sh`

### Historical evidence
- `raw/forensics/android_evidence_pack_20260515/mesa-turnip-build-evidence.json`
- `raw/forensics/android_evidence_pack_20260515/rusticl-build-evidence.json`

### Donor hints
- `raw/repos/mesa-driver-donors/mesa-turnip-android-driver/mesa-implement-android_stub.patch`
- `raw/repos/mesa-driver-donors/mesa-for-android-container/bin/update-android-headers.sh`
- `raw/repos/mesa-driver-donors/Adreno-Tools-Drivers/build_turnip.sh`

### Proof targets
- `forensics/mesa/driver-mesh-recovery-ready.json`
- `third_party/mesa-adreno/build-matrix.json`

## turnip_a7xx_a8xx_behavior

- goal: port A7xx/A8xx-specific Turnip behavior semantically instead of replaying force-wave hacks

### Upstream
- `src/freedreno/vulkan/tu_device.cc`
- `src/freedreno/vulkan/tu_knl_kgsl.cc`
- `src/freedreno/vulkan/tu_shader.cc`
- `src/freedreno/common/freedreno_devices.py`
- `src/freedreno/common/fd6_gmem_cache.h`

### TurboQuant native
- `native/opencl/src/tq_vk_probe.cpp`
- `native/opencl/src/tq_kernel_tune.cpp`
- `native/opencl/src/tq_gpu_profile.cpp`

### Historical evidence
- `raw/forensics/android_evidence_pack_20260515/device-identity.json`
- `raw/forensics/android_evidence_pack_20260515/turnip-rusticl-compute-diff.json`
- `raw/forensics/android_evidence_pack_20260515/vulkan-compute-benchmark.json`

### Donor hints
- `raw/repos/mesa-driver-donors/Adreno-Tools-Drivers/tu_gen8.patch`
- `raw/repos/mesa-driver-donors/Adreno-Tools-Drivers/patches/force_sysmem_no_autotuner.patch`
- `raw/repos/mesa-driver-donors/Adreno-Tools-Drivers/patches/quest3.patch`
- `raw/repos/mesa-driver-donors/mesa-tu8/src/freedreno`

### Proof targets
- `forensics/adb-driver-truth.json`
- `forensics/mesa/driver-mesh-recovery-ready.json`

## rusticl_compute_path

- goal: revalidate Rusticl over Freedreno/KGSL against old transfer failures and current Mesa surfaces

### Upstream
- `src/gallium/frontends/rusticl`
- `meson.options`
- `meson.build`

### TurboQuant native
- `native/opencl/src/tq_opencl_context.cpp`
- `native/opencl/src/tq_opencl_program.cpp`
- `native/opencl/src/tq_opencl_memory.cpp`
- `native/opencl/src/tq_opencl_qjl_score.cpp`
- `native/opencl/src/tq_opencl_mse_score.cpp`
- `native/opencl/src/tq_opencl_value_dequant.cpp`
- `native/opencl/src/tq_opencl_fused_attention.cpp`
- `native/opencl/src/tq_opencl_fused_attention_tiled.cpp`

### Historical evidence
- `raw/forensics/android_evidence_pack_20260515/rusticl-kgsl-final-decision.json`
- `raw/forensics/android_evidence_pack_20260515/rusticl-transfer-final-decision.json`
- `raw/forensics/android_evidence_pack_20260515/memory-visibility-report.json`
- `raw/forensics/android_evidence_pack_20260515/direct-bo-transfer.json`

### Donor hints
- `raw/repos/mesa-driver-donors/mesa-tu8/meson.build`
- `raw/repos/mesa-driver-donors/mesa-for-android-container/meson.build`

### Proof targets
- `native/opencl/test/test_qjl_parity.cpp`
- `native/opencl/test/test_mse_parity.cpp`
- `native/opencl/test/test_value_dequant_parity.cpp`
- `native/opencl/test/test_fused_attention_parity.cpp`

## packaging_and_release_delivery

- goal: retain delivery/packaging strengths without letting packaging repos define backend truth

### Upstream
- `docs.mesa3d.org/android.html`

### TurboQuant native
- `native/opencl/driver-pack/build_mesa.sh`
- `native/opencl/driver-pack/pack_driver.sh`
- `native/opencl/driver-pack/manifest.json`

### Historical evidence
- `raw/forensics/android_evidence_pack_20260515/RELEASE_EVIDENCE_MANIFEST.json`
- `raw/forensics/android_evidence_pack_20260515/release-truth-lock.json`

### Donor hints
- `raw/repos/mesa-driver-donors/Mesa-Turnip-Builder/build-turnip.sh`
- `raw/repos/mesa-driver-donors/mesa-turnip-android-driver/README.md`
- `raw/repos/mesa-driver-donors/Adreno-Tools-Drivers/build_turnip.sh`

### Proof targets
- `forensics/RELEASE_EVIDENCE_MANIFEST.json`
- `forensics/mesa/driver-mesh-recovery-ready.json`

