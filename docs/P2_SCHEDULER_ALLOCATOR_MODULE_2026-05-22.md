# P2 Scheduler/Allocator Module 2026-05-22

This document locks the first `P2` implementation wave as a concrete `TQ` module, not as a free-form frontier label.

## Module

`runtime_scheduler`

Source:

- [tq_runtime_scheduler.h](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/include/tq_runtime_scheduler.h:1)
- [tq_runtime_scheduler.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_runtime_scheduler.cpp:1)
- [tq_opencl_kernel_manager.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_kernel_manager.cpp:1)
- [tq_opencl_cli.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_cli.cpp:1)

## Scope

This `P2` module now materializes the full repo-controlled `runtime_scheduler` frontier for the current OpenCL kernel set:

1. compile-intent dedupe across kernel loads
2. cache residency visibility across binary cache and precompiled library lanes
3. runtime memory-budget planning anchored to live `CL_DEVICE_GLOBAL_MEM_SIZE`
4. build-path planning and eviction signaling for resident kernel artifacts
5. residency policy across `kernel-cache`, `precompiled-kernel-library`, and `autotune-cache`
6. compile-orchestrator policy across staged `precompiled/binary/IL/source` candidates with admission control
7. execution-admission policy across multi-queue dispatch budgeting, compile-vs-execute arbitration, and queue-pressure control
8. dispatch-latency policy across live latency class, queue aging, and urgency
9. kernel-class policy matrix across multiple real loaded kernels with per-class admission and aging evidence
10. class-aware dispatch redistribution with starvation prevention, lane throttling, and pressure recovery

The execution-admission lane is grounded in a bounded live `value_dequant_opencl_profiled` dispatch and exports:

- queue budget split between compile and execute
- queue-pressure band and queue-guard reason
- last dispatch kernel identity and launch geometry
- compile-vs-execute balance and arbitration verdict

The dispatch-latency-policy layer extends that lane with:

- latency class for the live dispatch kernel class
- compile-queue and dispatch-queue aging bands
- latency admission verdict and urgency class

The kernel-class-matrix layer extends this again with:

- per-class policy rows for multiple real loaded kernels
- per-class admission and aging evidence
- explicit live-dispatch anchor marking inside the class matrix

The final class-aware-dispatch layer closes the remaining `runtime_scheduler` frontier with:

- per-class budget redistribution between `value_dequant`, `mse`, `qjl`, and `attention` lanes
- starvation scores and aging boosts for queued kernel classes
- explicit throttle targets under queue pressure
- pressure recovery and live-anchor protection reasoning in the canonical artifact

This does not claim that every possible scheduler research direction is solved. It does claim that the repo-controlled `runtime_scheduler` frontier for the current kernel classes is now materialized as a shipped `TQ` module with live evidence.

## Controlled Lane

Safe one-shot smoke:

```sh
cd /data/data/com.termux/files/home/tmp_turboquant
node scripts/collect-opencl-runtime-scheduler-contract.mjs
```

Native command:

```sh
bash scripts/safe-runtime-pack-run.sh runtime-scheduler-smoke
```

## Artifact

- [opencl-runtime-scheduler-contract.json](/data/data/com.termux/files/home/tmp_turboquant/forensics/adreno/opencl-runtime-scheduler-contract.json:1)
- [runtime-scheduler-state.json](/data/data/com.termux/files/home/tmp_turboquant/forensics/runtime-scheduler-state.json:1)

## Engineering Meaning

`P2` is now grounded through a shipped `TQ` module with live evidence:

- compile plane
- cache plane
- memory-plan plane
- planner plane
- residency-policy plane
- compile-orchestrator plane
- execution-admission plane
- dispatch-latency plane
- kernel-class-matrix plane
- class-aware-dispatch plane

That closes the current repo-controlled scheduler/allocator frontier in the same disciplined form as `P1`: live runtime evidence, bounded smoke, and canonical contract artifacts instead of speculative aggregate rows.
