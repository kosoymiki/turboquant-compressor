# Android Linker Namespace & OpenCL

## Problem

On Android 7+, the dynamic linker isolates shared libraries into namespaces. Termux processes run in the `default` namespace, which cannot `dlopen()` libraries in the `vendor` or `sphal` namespace (like `/vendor/lib64/libOpenCL.so`).

This is not a bug — it's Android's security design.

## Symptoms

```
WARNING: linker: library "/vendor/lib64/libOpenCL.so" is not accessible for the namespace
```

`dlopen()` returns NULL. `clGetPlatformIDs` never runs.

## Loader States

| State | Meaning |
|-------|---------|
| `NO_LIBRARY` | No libOpenCL.so found anywhere |
| `LIBRARY_EXISTS_NOT_PROVEN_LOADABLE` | File exists but load not attempted |
| `BLOCKED_BY_ANDROID_LINKER_NAMESPACE` | dlopen fails due to namespace isolation |
| `LOADABLE_NO_PLATFORMS` | Library loads but no OpenCL platforms |
| `READY` | clGetPlatformIDs succeeds |

## Solutions (no root required)

1. **CPU/NEON backend** — always available, no GPU needed
2. **Vulkan compute** — may be accessible via Android graphics stack
3. **Mesa/Rusticl + freedreno** — userspace OpenCL via Mesa, uses /dev/kgsl-3d0
4. **Android companion service** — APK with JNI that has vendor namespace access, communicates via socket

## What we do NOT do

- No root
- No editing `/vendor/etc/public.libraries.txt`
- No `adb remount`
- No symlinks into vendor
- No LD_LIBRARY_PATH hacks (they don't bypass namespace isolation)
- No waiting for "maybe it works on some ROMs"

## Diagnosis

```bash
# Via MCP
{"method": "tools/call", "params": {"name": "turboquant_adreno_loader_probe", "arguments": {"deep": true}}}

# Via ADB (from host machine)
bash scripts/adb-adreno-diagnose.sh
```

## References

- AOSP linker namespace: bionic/linker/linker_namespaces.cpp
- AOSP libvndksupport: android_load_sphal_library()
- Qualcomm OpenCL on Adreno: requires vendor namespace access
