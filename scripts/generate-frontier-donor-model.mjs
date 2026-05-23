#!/usr/bin/env node

import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.join(__dirname, '..');
const forensicsDir = path.join(rootDir, 'forensics');
const docsDir = path.join(rootDir, 'docs');

mkdirSync(forensicsDir, { recursive: true });

const donors = [
  { id: 'mesa-rusticl-core-context', lane: 'mesa_system_svm', role: 'primary-local', kind: 'local-source', ref: '/data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/gallium/frontends/rusticl/core/context.rs', rationale: 'system SVM region resolution and raw-pointer materialization entrypoint' },
  { id: 'mesa-rusticl-core-kernel', lane: 'mesa_system_svm', role: 'primary-local', kind: 'local-source', ref: '/data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/gallium/frontends/rusticl/core/kernel.rs', rationale: 'kernel-side SVM arg lowering and device-address fallback path' },
  { id: 'mesa-rusticl-api-kernel', lane: 'mesa_system_svm', role: 'primary-local', kind: 'local-source', ref: '/data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/gallium/frontends/rusticl/api/kernel.rs', rationale: 'clSetKernelExecInfo gate for fine-grain system SVM' },
  { id: 'mesa-rusticl-core-device', lane: 'mesa_system_svm', role: 'primary-local', kind: 'local-source', ref: '/data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/gallium/frontends/rusticl/core/device.rs', rationale: 'system_svm_supported, fine_grain_buffer_svm_supported, svm_atomics_supported' },
  { id: 'mesa-rusticl-pipe-screen', lane: 'mesa_system_svm', role: 'primary-local', kind: 'local-source', ref: '/data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/gallium/frontends/rusticl/mesa/pipe/screen.rs', rationale: 'resource_create_buffer_from_user bridge to pipe_screen callback' },
  { id: 'mesa-rusticl-pipe-resource', lane: 'mesa_system_svm', role: 'primary-local', kind: 'local-source', ref: '/data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/gallium/frontends/rusticl/mesa/pipe/resource.rs', rationale: 'resource_get_address observation point for GPU address materialization' },
  { id: 'freedreno-resource-from-user', lane: 'mesa_system_svm', role: 'primary-local', kind: 'local-source', ref: '/data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/gallium/drivers/freedreno/freedreno_resource.c', rationale: 'hostptr user-memory resource creation in Gallium driver' },
  { id: 'freedreno-screen-caps', lane: 'mesa_system_svm', role: 'primary-local', kind: 'local-source', ref: '/data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/gallium/drivers/freedreno/freedreno_screen.c', rationale: 'resource_from_user_memory_compute_only and screen capability exposure' },
  { id: 'kgsl-bo-hostptr', lane: 'mesa_system_svm', role: 'primary-local', kind: 'local-source', ref: '/data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/freedreno/drm/kgsl/kgsl_bo.c', rationale: 'MAP_USER_MEM and GPUMEM_GET_INFO donor path for host pointers' },
  { id: 'kgsl-priv', lane: 'mesa_system_svm', role: 'primary-local', kind: 'local-source', ref: '/data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/freedreno/drm/kgsl/kgsl_priv.h', rationale: 'KGSL lower-contract declarations used by hostptr path' },
  { id: 'tq-cli-system-svm-smoke', lane: 'mesa_system_svm', role: 'primary-local', kind: 'local-source', ref: '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_cli.cpp', rationale: 'isolated live semantic smoke for raw pointers and atomics' },
  { id: 'tq-opencl-evidence', lane: 'mesa_system_svm', role: 'primary-local', kind: 'local-artifact', ref: '/data/data/com.termux/files/home/tmp_turboquant/forensics/opencl-capability-evidence.json', rationale: 'current canonical machine verdict for the lane' },
  { id: 'tq-pass16-stderr', lane: 'mesa_system_svm', role: 'primary-local', kind: 'local-artifact', ref: '/data/data/com.termux/files/home/tmp_turboquant/forensics/system-svm-direct-pass16.stderr', rationale: 'latest live trace before resource_from_user_memory/device-address materialization' },
  { id: 'tq-pass16-streamed-stderr', lane: 'mesa_system_svm', role: 'primary-local', kind: 'local-artifact', ref: '/data/data/com.termux/files/home/tmp_turboquant/forensics/system-svm-direct-pass16-streamed.stderr', rationale: 'fresh-sandbox crash-class after byte-identical restage' },
  { id: 'renderdoc-corpus-master', lane: 'mesa_system_svm', role: 'support-local', kind: 'local-doc', ref: '/data/data/com.termux/files/home/tmp_turboquant/docs/RENDERDOC_GPU_DEBUG_CORPUS_2026-05-22.md', rationale: 'stack-wide GPU debug donor policy and authority ordering' },
  { id: 'renderdoc-corpus-ingest', lane: 'mesa_system_svm', role: 'support-local', kind: 'local-artifact', ref: '/data/data/com.termux/files/home/tmp_turboquant/forensics/renderdoc-gpu-debug-corpus-ingest.json', rationale: 'ingested GPU-debug donor corpus metadata' },
  { id: 'pocl-clSetKernelExecInfo', lane: 'mesa_system_svm', role: 'primary-external', kind: 'github-local-clone', ref: '/data/data/com.termux/files/home/.cache/tq-svm-donors/pocl/lib/CL/clSetKernelExecInfo.c', rationale: 'best donor for system SVM exec-info runtime contract' },
  { id: 'pocl-ndrange-kernel', lane: 'mesa_system_svm', role: 'primary-external', kind: 'github-local-clone', ref: '/data/data/com.termux/files/home/.cache/tq-svm-donors/pocl/lib/CL/pocl_ndrange_kernel.c', rationale: 'indirect raw-pointer tracking and migration donor' },
  { id: 'pocl-svmoffset', lane: 'mesa_system_svm', role: 'primary-external', kind: 'github-local-clone', ref: '/data/data/com.termux/files/home/.cache/tq-svm-donors/pocl/lib/llvmopencl/SVMOffset.cc', rationale: 'LLVM-side SVM pointer normalization and offset donor' },
  { id: 'pocl-level0-exec-info', lane: 'mesa_system_svm', role: 'support-external', kind: 'github-local-clone', ref: '/data/data/com.termux/files/home/.cache/tq-svm-donors/pocl/lib/CL/devices/level0/pocl-level0.cc', rationale: 'device-side system SVM exec-info donor' },
  { id: 'pocl-basic-exec-info', lane: 'mesa_system_svm', role: 'contrast-external', kind: 'github-local-clone', ref: '/data/data/com.termux/files/home/.cache/tq-svm-donors/pocl/lib/CL/devices/basic/basic.c', rationale: 'minimal positive exec-info path useful as contrast donor' },
  { id: 'pocl-common-utils-svmcaps', lane: 'mesa_system_svm', role: 'support-external', kind: 'github-local-clone', ref: '/data/data/com.termux/files/home/.cache/tq-svm-donors/pocl/lib/CL/devices/common_utils.c', rationale: 'svm capability bit donor patterns' },
  { id: 'pocl-memory-management-doc', lane: 'mesa_system_svm', role: 'support-external', kind: 'web-doc', ref: 'https://portablecl.org/docs/html/memory_management.html', rationale: 'documents fine-grained SVM assumptions and broken coarse-grained host_ptr cases' },
  { id: 'pocl-github', lane: 'mesa_system_svm', role: 'support-external', kind: 'web-github', ref: 'https://github.com/pocl/pocl', rationale: 'canonical upstream donor repo for OpenCL runtime semantics' },
  { id: 'clvk-api-stub', lane: 'mesa_system_svm', role: 'contrast-external', kind: 'github-local-clone', ref: '/data/data/com.termux/files/home/.cache/tq-svm-donors/clvk/src/api.cpp', rationale: 'negative donor showing clSetKernelExecInfo stub and absent practical SVM path' },
  { id: 'clvk-github', lane: 'mesa_system_svm', role: 'contrast-external', kind: 'web-github', ref: 'https://github.com/kpet/clvk', rationale: 'negative donor repo for contrast and boundary setting' },
  { id: 'rusticl-svm-phoronix', lane: 'mesa_system_svm', role: 'support-external', kind: 'web-article', ref: 'https://www.phoronix.com/news/Rusticl-Cross-Vendor-SVM', rationale: 'historical cross-vendor SVM implementation context for Rusticl' },
  { id: 'rusticl-status-pdf', lane: 'mesa_system_svm', role: 'support-external', kind: 'web-pdf', ref: 'https://indico.freedesktop.org/event/4/contributions/207/attachments/130/191/main.pdf', rationale: 'explicit donor statement: use resource_from_user_memory when memory should remain on host' },
  { id: 'opencl-api-spec', lane: 'mesa_system_svm', role: 'authority-external', kind: 'web-spec', ref: 'https://bashbaug.github.io/OpenCL-Docs/html/OpenCL_API.html', rationale: 'normative SVM and clSetKernelExecInfo semantics' },
  { id: 'intel-svm-overview', lane: 'mesa_system_svm', role: 'support-external', kind: 'web-pdf', ref: 'https://www.intel.com/content/dam/develop/external/us/en/documents/svmoverview.pdf', rationale: 'clear SVM semantic overview for raw pointers and kernel arg handling' },
  { id: 'stack-overflow-svm-exec-info', lane: 'mesa_system_svm', role: 'support-external', kind: 'web-discussion', ref: 'https://stackoverflow.com/questions/72807280/intel-hd-graphics-violates-opencl-specification-regarding-svm/72996267', rationale: 'useful contrast on when non-arg SVM buffers must be listed in clSetKernelExecInfo' },
  { id: 'linux-sync-file-doc', lane: 'mesa_system_svm', role: 'support-external', kind: 'web-doc', ref: 'https://docs.kernel.org/6.2/driver-api/sync_file.html', rationale: 'lower synchronization/coherency background for host-visible shared memory behavior' },
  { id: 'linux-drm-uapi-doc', lane: 'mesa_system_svm', role: 'support-external', kind: 'web-doc', ref: 'https://docs.kernel.org/6.11/gpu/drm-uapi.html', rationale: 'driver/uapi authority for lower memory and synchronization contracts' },
  { id: 'renderdoc-github', lane: 'mesa_system_svm', role: 'support-external', kind: 'web-github', ref: 'https://github.com/baldurk/renderdoc', rationale: 'canonical RenderDoc upstream donor for GPU-debug methodology' },
  { id: 'renderdoc-android-doc', lane: 'mesa_system_svm', role: 'support-external', kind: 'web-doc', ref: 'https://documentation.ubuntu.com/anbox-cloud/howto/android/debug-graphics-renderdoc/', rationale: 'Android RenderDoc capture workflow donor for device-side replay discipline' },
  { id: 'cli-anything-donor', lane: 'all-frontier-lanes', role: 'support-external', kind: 'web-github', ref: 'https://github.com/HKUDS/CLI-Anything', rationale: 'agent CLI donor for orchestrated corpus-driven workflows' },
  { id: 'codegraph-donor', lane: 'all-frontier-lanes', role: 'support-external', kind: 'web-github', ref: 'https://github.com/colbymchenry/codegraph', rationale: 'semantic graph donor for codebase indexing and retrieval' }
];

const payload = {
  generatedAt: new Date().toISOString(),
  totalDonors: donors.length,
  rule: 'Every active lane requires an explicit donor model before major implementation work.',
  donors,
};

writeFileSync(path.join(forensicsDir, 'frontier-donor-model-2026-05-22.json'), JSON.stringify(payload, null, 2));
console.log(JSON.stringify({ generatedAt: payload.generatedAt, totalDonors: donors.length }, null, 2));
