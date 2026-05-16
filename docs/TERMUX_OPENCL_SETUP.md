# Termux OpenCL Setup Guide

## Target Platform

Snapdragon 8+ Gen 1 / Adreno 730-class GPU / Android + Termux

## Prerequisites

```bash
pkg install clang cmake make opencl-headers ocl-icd clinfo
```

## OpenCL Library Detection

The probe checks these paths in order:

1. `/system/vendor/lib64/libOpenCL.so`
2. `/vendor/lib64/libOpenCL.so`
3. `/system/lib64/libOpenCL.so`
4. `/vendor/lib64/libOpenCL_adreno.so`
5. `/system/vendor/lib64/libOpenCL_adreno.so`
6. `$PREFIX/lib/libOpenCL.so`
7. `$PREFIX/opt/vendor/lib/libOpenCL.so`

## Quick Probe

```bash
bash scripts/termux-opencl-probe.sh
```

Or via MCP:

```json
{"method": "tools/call", "params": {"name": "turboquant_opencl_probe", "arguments": {"deep": false}}}
```

## Deep Probe (requires clinfo)

```json
{"method": "tools/call", "params": {"name": "turboquant_opencl_probe", "arguments": {"deep": true, "timeoutMs": 3000}}}
```

## Backend States

| State | Meaning |
|-------|---------|
| `unavailable` | No libOpenCL.so found |
| `opencl_generic` | OpenCL present, vendor unknown |
| `opencl_adreno` | Adreno/Qualcomm detected |

## Troubleshooting

### No OpenCL library found

Some vendor ROMs do not expose OpenCL userspace libraries. Options:

1. Check if library exists but is not in expected path: `find / -name "libOpenCL*" 2>/dev/null`
2. Symlink if found: `ln -s /actual/path/libOpenCL.so $PREFIX/lib/libOpenCL.so`
3. Some devices require root to access vendor libraries

### clinfo fails

```bash
LD_LIBRARY_PATH="/vendor/lib64:/system/vendor/lib64:$PREFIX/lib" clinfo
```

### Performance mode

Sustained GPU clocks may require root or thermal governor changes. Benchmark results without performance mode reflect real-world throttled behavior.

## Verification

```bash
# Passes if OpenCL available, or with --allow-unavailable
node scripts/verify-adreno-opencl.mjs

# Skip if OpenCL not present
node scripts/verify-adreno-opencl.mjs --allow-unavailable
```

## Architecture

```
TypeScript MCP server (src/server.ts)
  ↓ JSON IPC / spawnSync
Native OpenCL sidecar (native/opencl/tq_opencl_cli)
  ↓ clCreateContext / clBuildProgram / clEnqueueNDRangeKernel
Adreno GPU
```

The MCP server never directly owns OpenCL handles. The native sidecar isolates GPU crashes from the MCP process.
