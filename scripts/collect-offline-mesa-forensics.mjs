import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const rootDir = path.resolve(new URL('..', import.meta.url).pathname);
const mesaRoot = path.resolve(rootDir, '../mesa-upstream-26.2-devel');
const forensicsDir = path.join(rootDir, 'forensics');
const outPath = path.join(forensicsDir, 'offline-mesa-forensics.json');
const driverRoot = path.join(rootDir, 'native/opencl/driver-root');
const usrLib = '/data/data/com.termux/files/usr/lib';

fs.mkdirSync(forensicsDir, { recursive: true });

function run(cmd, args) {
  const res = spawnSync(cmd, args, { encoding: 'utf8' });
  return {
    ok: res.status === 0,
    code: res.status,
    stdout: res.stdout ?? '',
    stderr: res.stderr ?? '',
  };
}

function readelfNeeded(file) {
  const res = run('readelf', ['-d', file]);
  if (!res.ok) return { file, error: res.stderr || `readelf failed: ${res.code}` };
  const needed = [];
  for (const line of res.stdout.split('\n')) {
    const m = line.match(/\(NEEDED\).*Shared library: \[(.+?)\]/);
    if (m) needed.push(m[1]);
  }
  const runpath = [];
  for (const line of res.stdout.split('\n')) {
    const m = line.match(/\((RUNPATH|RPATH)\).*Library runpath: \[(.+?)\]/);
    if (m) runpath.push(m[2]);
  }
  return { file, needed, runpath };
}

function exists(p) {
  try {
    fs.accessSync(p);
    return true;
  } catch {
    return false;
  }
}

function collectDriverRoot() {
  const layer1 = path.join(driverRoot, 'layer1-compute');
  const layer2 = path.join(driverRoot, 'layer2-vulkan');
  const expected = [
    'layer1-compute/libRusticlOpenCL.so.1.0.0',
    'layer1-compute/libRusticlOpenCL.so.1',
    'layer1-compute/libRusticlOpenCL.so',
    'layer1-compute/rusticl.icd',
    'layer2-vulkan/libvulkan_freedreno.so',
    'layer2-vulkan/freedreno_icd.aarch64.json',
    'env.sh',
    'meta/dependencies.txt',
    'meta/manifest.json',
  ];
  return {
    driverRoot,
    expected,
    missing: expected.filter(rel => !exists(path.join(driverRoot, rel))),
    layer1Needed: readelfNeeded(path.join(layer1, 'libRusticlOpenCL.so.1.0.0')),
    layer2Needed: readelfNeeded(path.join(layer2, 'libvulkan_freedreno.so')),
  };
}

function collectUserspaceAvailability() {
  const deps = [
    'libOpenCL.so',
    'libc++_shared.so',
    'libdrm.so',
    'libz.so.1',
    'libzstd.so.1',
    'libLLVM.so',
    'libclang-cpp.so',
    'libffi.so',
    'libxml2.so.16',
    'libiconv.so',
    'libicuuc.so.78',
    'libicudata.so.78',
  ];
  return {
    usrLib,
    deps: deps.map(name => ({
      name,
      present: exists(path.join(usrLib, name)),
      resolvedVariants: fs.existsSync(usrLib)
        ? fs.readdirSync(usrLib).filter(entry => entry === name || entry.startsWith(`${name}.`)).slice(0, 8)
        : [],
    })),
  };
}

function rgCount(pattern, targets) {
  const res = run('rg', ['-n', pattern, ...targets]);
  if (!res.ok && res.code !== 1) return { count: 0, error: res.stderr };
  const lines = res.stdout.trim() ? res.stdout.trim().split('\n') : [];
  return { count: lines.length, samples: lines.slice(0, 20) };
}

function rgLines(pattern, targets) {
  const res = run('rg', ['-n', pattern, ...targets]);
  if (!res.ok && res.code !== 1) return { count: 0, samples: [], error: res.stderr };
  const lines = res.stdout.trim() ? res.stdout.trim().split('\n') : [];
  return { count: lines.length, samples: lines.slice(0, 20) };
}

function collectTraceCoverage() {
  const tqTargets = [
    path.join(rootDir, 'native/opencl'),
    path.join(rootDir, 'src'),
    path.join(rootDir, 'scripts'),
  ];
  const mesaTargets = [
    path.join(mesaRoot, 'src/gallium/frontends/rusticl'),
    path.join(mesaRoot, 'src/gallium/drivers/freedreno'),
    path.join(mesaRoot, 'src/freedreno/vulkan'),
  ];
  return {
    tqOpenclTrace: rgCount('TQ_OPENCL_TRACE|tq-trace|DRV_TRACE', tqTargets),
    rusticlForensics: rgCount('forensics::trace|external_memory_trace|external_semaphore_trace', mesaTargets),
    pipeLayerForensics: rgCount('\"pipe-screen\"|\"pipe-resource\"|\"pipe-fence\"|\"pipe-context\"', mesaTargets),
    freedrenoTurnipEnv: rgCount('FD_RD_DUMP|FD_MESA_DEBUG|TU_DEBUG|TU_PERFETTO', [
      path.join(rootDir, 'src/native'),
      path.join(rootDir, 'scripts'),
      path.join(rootDir, 'docs'),
    ]),
    unresolvedInvalidOperation: rgCount('CL_INVALID_OPERATION', mesaTargets),
  };
}

function collectContracts() {
  const docs = [
    'docs/frontier-tracing-contract-2026-05-21.md',
    'docs/MESA_STACK_FORENSICS_MODULE_2026-05-21.md',
    'docs/MESA_TQ_FORENSIC_AUDIT_2026-05-21.md',
    'docs/tz-frontier-capability-matrix-2026-05-21.md',
  ];
  return {
    docs,
    missing: docs.filter(rel => !exists(path.join(rootDir, rel))),
  };
}

function collectLayerCoverage() {
  const mesaTargets = [
    path.join(mesaRoot, 'src/gallium/frontends/rusticl'),
    path.join(mesaRoot, 'src/gallium/drivers/freedreno'),
    path.join(mesaRoot, 'src/freedreno/vulkan'),
  ];
  const tqTargets = [
    path.join(rootDir, 'native/opencl'),
    path.join(rootDir, 'src'),
    path.join(rootDir, 'scripts'),
  ];
  return {
    tq_loader: rgLines('TQ_DRIVER_TRACE|DRV_TRACE|driver_trace_enabled', tqTargets),
    tq_probe: rgLines('frontier-smoke|probeOpenCl|turboquantNativeScope', tqTargets),
    rusticl_platform: rgLines('\"platform-vm\"|\"platform\"', mesaTargets),
    rusticl_device: rgLines('\"device-cap\"|\"device\".*external_|system_svm_supported|api_svm_supported', mesaTargets),
    rusticl_context: rgLines('\"context\"|resource_assign_vma|resource_import_dmabuf', mesaTargets),
    rusticl_memory: rgLines('external_memory_trace|\"external-memory\"', mesaTargets),
    rusticl_semaphore: rgLines('external_semaphore_trace|\"external-semaphore\"', mesaTargets),
    pipe_screen: rgLines('\"pipe-screen\"', mesaTargets),
    pipe_resource: rgLines('\"pipe-resource\"', mesaTargets),
    pipe_fence: rgLines('\"pipe-fence\"', mesaTargets),
    pipe_context: rgLines('\"pipe-context\"', mesaTargets),
    freedreno_import: rgLines('resource_from_handle|WINSYS_HANDLE_TYPE_FD|DMA-BUF|dmabuf', mesaTargets),
    freedreno_rd: rgLines('FD_RD_DUMP|FD_MESA_DEBUG', [
      path.join(rootDir, 'src/native'),
      path.join(rootDir, 'scripts'),
      path.join(rootDir, 'docs'),
    ]),
    turnip_vk: rgLines('TU_DEBUG|TU_PERFETTO|VK_ICD_FILENAMES|VK_DRIVER_FILES', [
      path.join(rootDir, 'src/native'),
      path.join(rootDir, 'scripts'),
      path.join(rootDir, 'docs'),
    ]),
    android_loader: rgLines('linker64|namespace|dlopen|OCL_ICD_VENDORS', tqTargets),
  };
}

function classifyFrontierLanes() {
  const mesaTargets = [
    path.join(mesaRoot, 'src/gallium/frontends/rusticl'),
    path.join(mesaRoot, 'src/gallium/drivers/freedreno'),
    path.join(mesaRoot, 'src/freedreno/vulkan'),
  ];
  const tqTargets = [
    path.join(rootDir, 'native/opencl'),
    path.join(rootDir, 'src'),
    path.join(rootDir, 'scripts'),
    path.join(rootDir, 'docs'),
  ];
  return {
    initialize_memory: {
      classification: 'trace-backed_source_lane_without_live_proof',
      sourceEvidence: rgLines('CL_CONTEXT_MEMORY_INITIALIZE_KHR|initialize_memory', [...mesaTargets, ...tqTargets]),
      nextFixSurface: 'rusticl context/property exposure and live replay',
    },
    external_memory: {
      classification: 'real_source_lane_with_import_risk',
      sourceEvidence: rgLines('clCreateBufferWithProperties|resource_import_dmabuf|external_memory_trace|CL_INVALID_OPERATION', mesaTargets),
      nextFixSurface: 'Rusticl/Gallium DMA-BUF import semantics after resource_from_handle',
    },
    external_semaphore: {
      classification: 'real_source_lane_with_export_gate_risk',
      sourceEvidence: rgLines('clCreateSemaphoreWithPropertiesKHR|external_semaphore_trace|create_fence_fd|fence_get_fd|CL_INVALID_OPERATION', mesaTargets),
      nextFixSurface: 'semaphore advertise/create/export sync-fd path',
    },
    system_svm: {
      classification: 'coarse_only_source_lane',
      sourceEvidence: rgLines('system_svm_supported|api_svm_supported|alloc_vm|resource_assign_vma', mesaTargets),
      nextFixSurface: 'screen/system_svm capability bridge and live VM-backed proof',
    },
    command_buffer: {
      classification: 'absent_or_unimplemented_subsystem',
      sourceEvidence: rgLines('clCreateCommandBufferKHR|command_buffer|CL_INVALID_OPERATION', mesaTargets),
      nextFixSurface: 'new Rusticl command-buffer subsystem',
    },
    loader_namespace: {
      classification: 'offline_closed_live_replay_pending',
      sourceEvidence: rgLines('linker64|namespace|dlopen|OCL_ICD_VENDORS|VK_ICD_FILENAMES', tqTargets),
      nextFixSurface: 'device replay only',
    },
    staged_dependency_chain: {
      classification: 'offline_closed_live_replay_pending',
      sourceEvidence: rgLines('libz.so.1|libffi.so|libxml2.so.16|libicuuc.so.78|dependencies.txt', [
        path.join(rootDir, 'docs'),
        path.join(rootDir, 'forensics'),
        path.join(rootDir, 'native/opencl/driver-root/meta'),
      ]),
      nextFixSurface: 'device replay only',
    },
    turnip_submission_trace: {
      classification: 'env_profile_ready_live_artifact_missing',
      sourceEvidence: rgLines('TU_DEBUG|TU_PERFETTO', [
        path.join(rootDir, 'src/native'),
        path.join(rootDir, 'scripts'),
        path.join(rootDir, 'docs'),
      ]),
      nextFixSurface: 'device replay only',
    },
    freedreno_rd_dump: {
      classification: 'env_profile_ready_live_artifact_missing',
      sourceEvidence: rgLines('FD_RD_DUMP|FD_MESA_DEBUG', [
        path.join(rootDir, 'src/native'),
        path.join(rootDir, 'scripts'),
        path.join(rootDir, 'docs'),
      ]),
      nextFixSurface: 'device replay only',
    },
  };
}

const payload = {
  generatedAt: new Date().toISOString(),
  mode: 'offline_no_adb',
  driverRoot: collectDriverRoot(),
  userspace: collectUserspaceAvailability(),
  traceCoverage: collectTraceCoverage(),
  layerCoverage: collectLayerCoverage(),
  frontierLanes: classifyFrontierLanes(),
  contracts: collectContracts(),
  engineeringVerdict: {
    meaning: [
      'This artifact is the canonical no-adb forensic snapshot for the TQ+Mesa stack.',
      'It does not replace live runtime evidence, but it closes local uncertainty around driver-root completeness, dependency graph, trace coverage, and replay contract presence.',
      'Frontier lanes are now classified with source evidence and named next-fix surfaces, so unresolved false states are no longer opaque in offline mode.',
    ],
  },
};

fs.writeFileSync(outPath, JSON.stringify(payload, null, 2));
console.log(outPath);
