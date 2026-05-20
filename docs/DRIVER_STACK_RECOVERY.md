# TurboQuant Driver Stack Recovery

This recovery track treats the custom Android GPU path as a driver mesh, not a single backend.

## Layers

1. `mesa_kgsl_native_or_proot`
   - Rusticl over Freedreno Gallium on KGSL
   - Turnip Vulkan on KGSL
   - current target: Mesa `26.2.0-devel`

2. `android_companion_service`
   - app-private bridge for namespace-isolated Android environments
   - not a replacement for native Termux recovery, but a parallel delivery route

3. `adrenotools_turnip`
   - hot-swap packaging path for app-side Turnip deployment
   - use for packaging and runtime delivery patterns, not as the whole stack

4. `adb_forensics_replay`
   - ADB-backed device truth
   - KGSL sysfs, Vulkan `vkjson`, runtime observability, bugreport/perfetto capture

## Source-of-truth files

- `native/opencl/driver-pack/build_mesa.sh`
- `native/opencl/driver-pack/pack_driver.sh`
- `native/opencl/driver-pack/manifest.json`
- `third_party/mesa-adreno/upstream.lock.json`
- `third_party/mesa-adreno/manifest.lock.json`
- `third_party/mesa-adreno/proof-matrix.json`
- `third_party/mesa-adreno/build-matrix.json`
- `third_party/mesa-adreno/applicability-report.json`
- `native/opencl/src/tq_driver_loader.cpp`
- `native/opencl/src/tq_opencl_probe.cpp`
- `native/opencl/src/tq_vk_probe.cpp`
- `native/opencl/src/tq_gpu_profile.cpp`
- `native/opencl/src/tq_kernel_tune.cpp`
- `native/opencl/test/*parity*.cpp`

## Current recovery reality

- old `0001` and `0002` patch files are not trustworthy as direct apply artifacts
- old `0003` does not apply cleanly to current Mesa 26.2 Freedreno/Turnip code
- Mesa 26.2 already exposes `-Dfreedreno-kmds=kgsl` upstream
- driver recovery should prefer semantic reconstruction over blind re-apply

## Recovery order

1. pin upstream Mesa and keep that pin visible
2. make `driver-pack` build and manifest contracts current
3. prove live device truth through `adb_forensics_replay`
4. reconstruct KGSL/Freedreno glue patch semantically
5. reconstruct Android stub/cutils patch semantically
6. port Turnip A7xx/A8xx behavior semantically
7. rerun parity and runtime forensics
