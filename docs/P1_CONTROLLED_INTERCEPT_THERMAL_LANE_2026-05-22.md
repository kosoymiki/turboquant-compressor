# P1 Controlled Intercept And Thermal Lane

This lane separates external intercept integration and sustained thermal policy from smoke CI.

Artifacts:

- `forensics/adreno/opencl-intercept-manifest.json`
- `forensics/adreno/opencl-intercept.config`
- `forensics/adreno/opencl-intercept-build.json`
- `forensics/adreno/opencl-intercept-smoke.json`
- `forensics/adreno/opencl-performance-policy.json`
- `forensics/adreno/opencl-sustained-policy.json`
- `forensics/adreno/opencl-controlled-profile-lane.json`
- `forensics/adreno/opencl-controlled-sustained-lane.json`

Entry points:

- `npm run build:opencl:intercept`
- `npm run forensics:opencl-intercept-manifest`
- `npm run forensics:opencl-intercept-smoke`
- `npm run bench:opencl:controlled`
- `npm run bench:opencl:controlled:intercept`
- `npm run bench:opencl:controlled:sustained`

Policy:

- default execution is `dry-run`
- default profile is `smoke`
- default sustained limits are bounded by `--max-runs`, `--minutes`, and `--timeout-seconds`
- external intercept is formalized through `opencl-intercept-manifest.json`
- `ready=true` means contract/config materialized
- `runtimeReady=true` requires a real intercept library
- `selectedBy=auto_discovery` is valid for config-only contract readiness
- the runtime library is built from vendored upstream `third_party/OpenCL-Intercept-Layer` and mirrored as `libOpenCLIntercept.so`
- live runtime validation must use the one-call `opencl-intercept-smoke` lane before any benchmark or sustained execution
- intercept execution can be forced to fail closed with `--require-ready-intercept`
- any real sustained execution must be opt-in with `--execute`
