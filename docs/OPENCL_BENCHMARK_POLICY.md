# OpenCL Benchmark Policy

## Principle

No performance or optimization claims without recorded evidence.

## Regulated Phrases

The following phrases are **forbidden** in docs/README unless `forensics/opencl-adreno-report.json` exists with `claim_safe: true`:

- Phrases matching: `/OpenCL\s+accelerat/i`
- Phrases matching: `/Adreno\s+optimiz/i`
- Phrases matching: `/Snapdragon.*ready/i`
- Phrases matching: `/mobile\s+high.?end\s+backend/i`

## Permitted Phrases (no evidence required)

- "OpenCL/Adreno target backend"
- "Adreno backend source layer"
- "Snapdragon-first design target"
- "Mobile high-end readiness requires benchmark evidence"

## Required Evidence for Claims

1. `forensics/opencl-adreno-report.json` exists
2. `claim_safe: true` (all parity pass AND OpenCL available)
3. Device metadata recorded

## Benchmark Procedure

```bash
# 1. Build native sidecar
npm run build:opencl

# 2. Run benchmark (always writes report, even if unavailable)
node scripts/benchmark-opencl-adreno.mjs

# 3. Verify claims are backed
node scripts/verify-opencl-claims.mjs
```

## Benchmark States

| State | Meaning |
|-------|---------|
| `ADRENO_NATIVE_NOT_BUILT` | `tq_opencl_cli` binary missing |
| `CUSTOM_RUSTICL_STACK_PARITY_FAIL` | The shipped custom stack loads, but parity tests fail |
| `CUSTOM_RUSTICL_STACK_READY` | The shipped custom stack loads and parity passes |
| `ADRENO_VENDOR_OPENCL_PARITY_FAIL` | Diagnostic vendor OpenCL route loads, but parity fails |
| `ADRENO_VENDOR_OPENCL_READY` | Diagnostic vendor OpenCL route loads and parity passes |
| `ADRENO_VENDOR_OPENCL_BLOCKED_BY_NAMESPACE` | Library exists, dlopen blocked |
| `ADRENO_VENDOR_OPENCL_UNAVAILABLE` | No library or no platforms |

## Thermal Policy

- Benchmarks reflect real-world throttled behavior unless performance mode is explicitly enabled
- Sustained clock rates are NOT assumed
- Report must note if thermal governor was modified

## Claims Gate

`scripts/verify-opencl-claims.mjs` scans README.md and docs/ for regulated phrase patterns. If found without matching evidence, the gate fails.

## Release Summary Integration

`verify:release` does NOT require OpenCL. The release summary reports:

- `CUSTOM_RUSTICL_STACK_READY` — if the tracked custom driver route passes parity
- `ADRENO_VENDOR_OPENCL_READY` — diagnostic-only vendor route passes parity
- `ADRENO_VENDOR_OPENCL_BLOCKED_BY_NAMESPACE` — expected on no-root Android
- `ADRENO_VENDOR_OPENCL_UNAVAILABLE` — no library found
- `ADRENO_NATIVE_NOT_BUILT` — binary not compiled
