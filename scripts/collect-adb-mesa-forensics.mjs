#!/usr/bin/env node

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import {
  classifyExternalSemaphoreContract,
  parseExternalSemaphoreTrace,
} from './lib/external-semaphore-contract.mjs';
import {
  classifyCommandBufferContract,
  classifyInitializeMemoryContract,
  classifySvmContract,
} from './lib/frontier-contracts.mjs';
import {
  classifyLoaderNamespaceContract,
  classifyRouteSelectionContract,
} from './lib/route-selection-contract.mjs';
import {
  classifyRenderNodePolicyContract,
  classifySpirvCapabilityContract,
  classifyStagedDependencyChainContract,
} from './lib/kernel-blocker-contracts.mjs';
import { classifySyncobjSubstrateContract } from './lib/syncobj-substrate-contract.mjs';
import {
  classifyCommandBufferSubstrateContract,
  classifySystemSvmSubstrateContract,
} from './lib/remaining-substrate-contracts.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.join(__dirname, '..');
const forensicsDir = path.join(rootDir, 'forensics');
const stageScript = path.join(rootDir, 'scripts', 'stage-adb-rusticl-runtime.sh');
const mesaLib = path.join(rootDir, 'native', 'opencl', 'driver-root', 'layer1-compute', 'libRusticlOpenCL.so.1.0.0');
const mesaRoot = '/data/data/com.termux/files/home/mesa-upstream-26.2-devel';
const libclc32 = '/data/data/com.termux/files/usr/share/clc/spirv-mesa3d-.spv';
const libclc64 = '/data/data/com.termux/files/usr/share/clc/spirv64-mesa3d-.spv';
const runAsPackage = process.env.TQ_ADB_RUN_AS_PACKAGE || '';

const endpoint = process.argv[2] || process.env.TQ_ADB_SERIAL || process.env.ADB_SERIAL || '';
const forensicsOutputDir = process.env.TQ_FORENSICS_OUTPUT_DIR
  ? path.resolve(process.env.TQ_FORENSICS_OUTPUT_DIR)
  : forensicsDir;
if (!endpoint) {
  console.error('usage: node scripts/collect-adb-mesa-forensics.mjs <host:port>');
  process.exit(1);
}

mkdirSync(forensicsOutputDir, { recursive: true });

function run(cmd, args, opts = {}) {
  return spawnSync(cmd, args, {
    encoding: 'utf8',
    ...opts,
  });
}

function adb(args, opts = {}) {
  return run('adb', ['-s', endpoint, ...args], opts);
}

function shell(command, opts = {}) {
  return adb(['shell', command], opts);
}

function shellViaRunAs(command, opts = {}) {
  if (!runAsPackage) {
    return {
      status: -1,
      stdout: '',
      stderr: 'run-as package not configured',
    };
  }
  return adb(['shell', `run-as ${runAsPackage} sh -lc '${command.replaceAll("'", "'\\''")}'`], opts);
}

function buildRunAsEnv(extraDebug = '') {
  const debugFlags = ['memory'];
  if (extraDebug) debugFlags.push(...extraDebug.split(',').filter(Boolean));
  return [
    'LD_LIBRARY_PATH=/data/user/0/com.termux/files/tq-rusticl',
    'OCL_ICD_VENDORS=/data/user/0/com.termux/files/tq-rusticl/icd',
    'TQ_OPENCL_TRACE=1',
    'TQ_DRIVER_TRACE=1',
    'TQ_MESA_FORENSICS=1',
    `RUSTICL_DEBUG=${debugFlags.join(',')}`,
  ].join(' ');
}

function buildRunAsStageCommand() {
  return [
    'mkdir -p files/tq-rusticl/icd files/tq-rusticl/clc',
    'cp /data/local/tmp/tq_opencl_cli files/tq_opencl_cli',
    'cp /data/local/tmp/tq-rusticl/libOpenCL.so files/tq-rusticl/',
    'cp /data/local/tmp/tq-rusticl/libRusticlOpenCL.so.1.0.0 files/tq-rusticl/',
    'cp /data/local/tmp/tq-rusticl/libc++_shared.so files/tq-rusticl/',
    'cp /data/local/tmp/tq-rusticl/libdrm.so files/tq-rusticl/',
    'cp /data/local/tmp/tq-rusticl/libz.so.1.3.2 files/tq-rusticl/',
    'cp /data/local/tmp/tq-rusticl/libzstd.so.1.5.7 files/tq-rusticl/',
    'cp /data/local/tmp/tq-rusticl/libLLVM.so files/tq-rusticl/',
    'cp /data/local/tmp/tq-rusticl/libclang-cpp.so files/tq-rusticl/',
    'cp /data/local/tmp/tq-rusticl/libffi.so files/tq-rusticl/',
    'cp /data/local/tmp/tq-rusticl/libxml2.so.16.1.3 files/tq-rusticl/',
    'cp /data/local/tmp/tq-rusticl/libiconv.so files/tq-rusticl/',
    'cp /data/local/tmp/tq-rusticl/libicuuc.so.78.3 files/tq-rusticl/',
    'cp /data/local/tmp/tq-rusticl/libicudata.so.78.3 files/tq-rusticl/',
    'cp /data/local/tmp/tq-rusticl/icd/rusticl.icd files/tq-rusticl/icd/',
    'cp /data/local/tmp/tq-rusticl/clc/spirv-mesa3d-.spv files/tq-rusticl/clc/',
    'cp /data/local/tmp/tq-rusticl/clc/spirv64-mesa3d-.spv files/tq-rusticl/clc/',
    'ln -sf libRusticlOpenCL.so.1.0.0 files/tq-rusticl/libRusticlOpenCL.so.1',
    'ln -sf libRusticlOpenCL.so.1 files/tq-rusticl/libRusticlOpenCL.so',
    'ln -sf libz.so.1.3.2 files/tq-rusticl/libz.so.1',
    'ln -sf libzstd.so.1.5.7 files/tq-rusticl/libzstd.so.1',
    'ln -sf libxml2.so.16.1.3 files/tq-rusticl/libxml2.so.16',
    'ln -sf libicuuc.so.78.3 files/tq-rusticl/libicuuc.so.78',
    'ln -sf libicudata.so.78.3 files/tq-rusticl/libicudata.so.78',
    'chmod 700 files/tq_opencl_cli files/tq-rusticl/libOpenCL.so files/tq-rusticl/libRusticlOpenCL.so.1.0.0',
  ].join(' && ');
}

function trim(text) {
  return `${text || ''}`.trim();
}

function tryJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function lines(text) {
  return trim(text)
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
}

function parseAccessSummary(text) {
  const output = `${text || ''}`;
  const renderMatch = output.match(/render:(\d+)/);
  const kgslMatch = output.match(/kgsl:(\d+)/);
  return {
    raw: trim(output),
    renderNodeAccessible: renderMatch ? renderMatch[1] === '0' : null,
    kgslAccessible: kgslMatch ? kgslMatch[1] === '0' : null,
  };
}

function findMissingLibsFromForensicsArtifacts(forensicsRoot, targetEndpoint) {
  const candidates = [];
  if (existsSync(forensicsRoot)) {
    for (const name of readdirSync(forensicsRoot)) {
      if (/^opencl-adb-.*-\d{4}-\d{2}-\d{2}\.json$/.test(name)) {
        candidates.push(path.join(forensicsRoot, name));
      }
    }
  }
  const endpointStem = targetEndpoint.replaceAll('.', '-').replace(':', '-');
  candidates.sort((a, b) => a.localeCompare(b));
  for (const candidate of candidates) {
    if (!existsSync(candidate)) continue;
    if (!path.basename(candidate).includes(endpointStem)) continue;
    try {
      const parsed = JSON.parse(readFileSync(candidate, 'utf-8'));
      const failures = parsed?.customRouteAttempt?.postFixFailures;
      if (Array.isArray(failures)) {
        const libs = [];
        for (const line of failures) {
          const match = `${line}`.match(/(libz\.so\.1|libffi\.so|libxml2\.so\.16|libicuuc\.so\.78)/);
          if (match && !libs.includes(match[1])) libs.push(match[1]);
        }
        return libs;
      }
    } catch {
      continue;
    }
  }
  return [];
}

function stringsGrep(file, pattern) {
  const res = run('sh', ['-lc', `strings '${file}' | grep -E '${pattern}' | sed -n '1,120p'`]);
  return lines(res.stdout);
}

function sourceContains(file, pattern) {
  if (!existsSync(file)) return false;
  return readFileSync(file, 'utf-8').includes(pattern);
}

const connect = run('adb', ['connect', endpoint]);
const model = trim(shell('getprop ro.product.model').stdout);
const boardPlatform = trim(shell('getprop ro.board.platform').stdout);
const fingerprint = trim(shell('getprop ro.build.fingerprint').stdout);
const kernelRelease = trim(shell('uname -r').stdout);
const eglRenderer = lines(shell('dumpsys SurfaceFlinger | grep -i -m 1 "GLES:"').stdout)[0] || '';
const renderNodeMeta = trim(shell('ls -lZ /dev/dri/renderD128 2>&1').stdout);
const shellIdentity = trim(shell('id; getenforce').stdout);
const vendorOpenclLibrary = trim(shell('ls /vendor/lib64/libOpenCL.so 2>/dev/null').stdout);
const vendorIcdDirCheck = trim(shell('ls /vendor/etc/OpenCL/vendors 2>&1').stdout);
const dmaHeapNodes = lines(shell('ls /dev/dma_heap 2>/dev/null').stdout);

const stage = run('bash', [stageScript, '/data/local/tmp/tq-rusticl'], {
  env: { ...process.env, ANDROID_SERIAL: endpoint },
});

const pushClc32 = existsSync(libclc32)
  ? adb(['push', libclc32, '/data/local/tmp/tq-rusticl/spirv-mesa3d-.spv'])
  : null;
const pushClc64 = existsSync(libclc64)
  ? adb(['push', libclc64, '/data/local/tmp/tq-rusticl/spirv64-mesa3d-.spv'])
  : null;
const runAsStage = runAsPackage
  ? shellViaRunAs(buildRunAsStageCommand(), { timeout: 30000 })
  : { status: -1, stdout: '', stderr: 'run-as package not configured' };

const vendorProbe = shell(
  'TQ_OPENCL_TRACE=1 TQ_DRIVER_TRACE=1 /data/local/tmp/tq_opencl_cli probe',
  { timeout: 15000 },
);
const customProbe = shell(
  'LD_LIBRARY_PATH=/data/local/tmp/tq-rusticl OCL_ICD_VENDORS=/data/local/tmp/tq-rusticl/icd TQ_OPENCL_TRACE=1 TQ_DRIVER_TRACE=1 TQ_MESA_FORENSICS=1 RUSTICL_DEBUG=memory /data/local/tmp/tq_opencl_cli probe',
  { timeout: 20000 },
);
const customFrontierSmoke = shell(
  'LD_LIBRARY_PATH=/data/local/tmp/tq-rusticl OCL_ICD_VENDORS=/data/local/tmp/tq-rusticl/icd TQ_OPENCL_TRACE=1 TQ_DRIVER_TRACE=1 TQ_MESA_FORENSICS=1 RUSTICL_DEBUG=memory /data/local/tmp/tq_opencl_cli frontier-smoke',
  { timeout: 20000 },
);
const runAsBaseEnv = buildRunAsEnv();
const runAsInvalidSpirvEnv = buildRunAsEnv('allow_invalid_spirv');

const customProbeRunAs = shellViaRunAs(
  `cd /data/user/0/com.termux/files && ${runAsBaseEnv} ./tq_opencl_cli probe`,
  { timeout: 30000 },
);
const customFrontierRunAs = shellViaRunAs(
  `cd /data/user/0/com.termux/files && ${runAsBaseEnv} ./tq_opencl_cli frontier-smoke`,
  { timeout: 60000 },
);
const customProbeRunAsAllowInvalidSpirv = shellViaRunAs(
  `cd /data/user/0/com.termux/files && ${runAsInvalidSpirvEnv} ./tq_opencl_cli probe`,
  { timeout: 30000 },
);
const customFrontierRunAsAllowInvalidSpirv = shellViaRunAs(
  `cd /data/user/0/com.termux/files && ${runAsInvalidSpirvEnv} ./tq_opencl_cli frontier-smoke`,
  { timeout: 60000 },
);
const linkerVendor = shell('/system/bin/linker64 --list /data/local/tmp/tq_opencl_cli 2>&1 | sed -n "1,220p"');
const linkerCustom = shell('LD_LIBRARY_PATH=/data/local/tmp/tq-rusticl /system/bin/linker64 --list /data/local/tmp/tq_opencl_cli 2>&1 | sed -n "1,260p"');
const renderOpenAttempt = shell('toybox head -c 0 /dev/dri/renderD128; echo rc:$?');
const logcatSlice = shell('logcat -d -b all | grep -i -E "renderD128|rusticl|libclc|avc:|denied" | tail -n 120', {
  timeout: 15000,
});
const stagedFiles = shell('ls -l /data/local/tmp/tq-rusticl | sed -n "1,200p"');
const runAsNodeAccess = shellViaRunAs(
  'toybox head -c 0 /dev/dri/renderD128 >/dev/null 2>&1; echo render:$?; toybox head -c 0 /dev/kgsl-3d0 >/dev/null 2>&1; echo kgsl:$?',
  { timeout: 10000 },
);

const customProbeJson = tryJson(customProbe.stdout);
const vendorProbeJson = tryJson(vendorProbe.stdout);
const customFrontierJson = tryJson(customFrontierSmoke.stdout);
const runAsProbeJson = tryJson(customProbeRunAs.stdout);
const runAsFrontierJson = tryJson(customFrontierRunAs.stdout);
const semaphoreTrace = parseExternalSemaphoreTrace(
  `${customProbeRunAs.stderr || ''}\n${customFrontierRunAs.stderr || ''}\n${logcatSlice.stdout || ''}`,
);
const runAsAccess = parseAccessSummary(runAsNodeAccess.stdout || runAsNodeAccess.stderr);
const runAsDevice = runAsProbeJson?.devices?.[0] || null;
const runAsSmokeCaps = runAsFrontierJson?.capabilities || {};
const runAsSemaphoreSmoke = runAsFrontierJson?.externalSemaphoreSmoke || {};
const allowInvalidProbeJson = tryJson(customProbeRunAsAllowInvalidSpirv.stdout);
const allowInvalidFrontierJson = tryJson(customFrontierRunAsAllowInvalidSpirv.stdout);
const externalSemaphoreContractInputs = {
  freedrenoHasSyncobj: semaphoreTrace.freedrenoHasSyncobj,
  freedrenoFenceSignal: semaphoreTrace.freedrenoFenceSignal,
  freedrenoFenceGetFd: semaphoreTrace.freedrenoFenceGetFd,
  freedrenoSemaphoreCreate: semaphoreTrace.freedrenoSemaphoreCreate,
  probeHasExternalSemaphore:
    typeof runAsDevice?.hasExternalSemaphore === 'boolean' ? runAsDevice.hasExternalSemaphore : null,
  smokeHasExternalSemaphore:
    typeof runAsSmokeCaps.externalSemaphore === 'boolean' ? runAsSmokeCaps.externalSemaphore : null,
  createdExportable:
    typeof runAsSemaphoreSmoke.createdExportable === 'boolean'
      ? runAsSemaphoreSmoke.createdExportable
      : null,
  signaled:
    typeof runAsSemaphoreSmoke.signaled === 'boolean' ? runAsSemaphoreSmoke.signaled : null,
  exportedSyncFd:
    typeof runAsSemaphoreSmoke.exportedSyncFd === 'boolean' ? runAsSemaphoreSmoke.exportedSyncFd : null,
  importedSyncFd:
    typeof runAsSemaphoreSmoke.importedSyncFd === 'boolean' ? runAsSemaphoreSmoke.importedSyncFd : null,
  waitedImported:
    typeof runAsSemaphoreSmoke.waitedImported === 'boolean' ? runAsSemaphoreSmoke.waitedImported : null,
  renderNodeAccessible: runAsAccess.renderNodeAccessible,
  kgslAccessible: runAsAccess.kgslAccessible,
  deviceVersion: semaphoreTrace.deviceVersion,
  kernelRelease,
};
const externalSemaphoreContractVerdict = classifyExternalSemaphoreContract(
  externalSemaphoreContractInputs,
);
const initializeMemoryContractInputs = {
  probeHasInitializeMemory:
    typeof runAsDevice?.hasInitializeMemory === 'boolean' ? runAsDevice.hasInitializeMemory : null,
  smokeHasInitializeMemory:
    typeof runAsSmokeCaps.initializeMemory === 'boolean' ? runAsSmokeCaps.initializeMemory : null,
  initContext: typeof runAsFrontierJson?.initContext === 'boolean' ? runAsFrontierJson.initContext : null,
};
const commandBufferContractInputs = {
  probeHasCommandBuffer:
    typeof runAsDevice?.hasCommandBuffer === 'boolean' ? runAsDevice.hasCommandBuffer : null,
  smokeHasCommandBuffer:
    typeof runAsSmokeCaps.commandBuffer === 'boolean' ? runAsSmokeCaps.commandBuffer : null,
  commandBufferCapabilities:
    typeof runAsDevice?.commandBufferCapabilities === 'number'
      ? runAsDevice.commandBufferCapabilities
      : null,
  probeHasMutableDispatch:
    typeof runAsDevice?.hasCommandBufferMutableDispatch === 'boolean'
      ? runAsDevice.hasCommandBufferMutableDispatch
      : null,
};
const svmContractInputs = {
  probeHasSvm: typeof runAsDevice?.hasSvm === 'boolean' ? runAsDevice.hasSvm : null,
  probeHasSvmCoarse:
    typeof runAsDevice?.hasSvmCoarse === 'boolean' ? runAsDevice.hasSvmCoarse : null,
  probeHasSvmFine: typeof runAsDevice?.hasSvmFine === 'boolean' ? runAsDevice.hasSvmFine : null,
  probeHasSvmAtomics:
    typeof runAsDevice?.hasSvmAtomics === 'boolean' ? runAsDevice.hasSvmAtomics : null,
  smokeHasSvm: typeof runAsSmokeCaps.svm === 'boolean' ? runAsSmokeCaps.svm : null,
  smokeHasSvmCoarse:
    typeof runAsSmokeCaps.svmCoarse === 'boolean' ? runAsSmokeCaps.svmCoarse : null,
  smokeHasSvmFine:
    typeof runAsSmokeCaps.svmFine === 'boolean' ? runAsSmokeCaps.svmFine : null,
  smokeHasSvmAtomics:
    typeof runAsSmokeCaps.svmAtomics === 'boolean' ? runAsSmokeCaps.svmAtomics : null,
  smokeSystemRawPointerOk:
    typeof runAsFrontierJson?.systemSvmSmoke?.rawPointerOk === 'boolean'
      ? runAsFrontierJson.systemSvmSmoke.rawPointerOk
      : null,
  smokeSystemAtomicOk:
    typeof runAsFrontierJson?.systemSvmSmoke?.atomicOk === 'boolean'
      ? runAsFrontierJson.systemSvmSmoke.atomicOk
      : null,
};
const customLinkerUsesStagedOpenCl = lines(linkerCustom.stdout).some((line) =>
  line.includes('libOpenCL.so => /data/local/tmp/tq-rusticl/libOpenCL.so'),
);
const loaderNamespaceContractInputs = {
  vendorProbeStatus: vendorProbe.status,
  vendorProbeStderr: trim(vendorProbe.stderr),
  customProbeStatus: customProbe.status,
  customRunAsProbeStatus: customProbeRunAs.status,
  customRunAsPlatformCount:
    typeof runAsProbeJson?.platformCount === 'number' ? runAsProbeJson.platformCount : null,
  customRunAsDeviceCount:
    typeof runAsProbeJson?.deviceCount === 'number' ? runAsProbeJson.deviceCount : null,
  customLoadedLibraryPath:
    typeof runAsProbeJson?.loadedLibraryPath === 'string' ? runAsProbeJson.loadedLibraryPath : null,
  runAsStageStatus: runAsStage.status,
  renderNodeAccessible: runAsAccess.renderNodeAccessible,
  kgslAccessible: runAsAccess.kgslAccessible,
  customLinkerUsesStagedOpenCl,
};
const loaderNamespaceContractVerdict = classifyLoaderNamespaceContract(
  loaderNamespaceContractInputs,
);
const routeSelectionContractInputs = {
  vendorLoader: loaderNamespaceContractVerdict.classification,
  customLoader: loaderNamespaceContractVerdict.classification,
  customRunAsPlatformCount: loaderNamespaceContractInputs.customRunAsPlatformCount,
  customRunAsDeviceCount: loaderNamespaceContractInputs.customRunAsDeviceCount,
  customHasExternalMemory:
    typeof runAsDevice?.hasExternalMemory === 'boolean' ? runAsDevice.hasExternalMemory : null,
  customHasSvm: typeof runAsDevice?.hasSvm === 'boolean' ? runAsDevice.hasSvm : null,
  vendorProbeUsable: vendorProbe.status === 0,
};
const routeSelectionContractVerdict = classifyRouteSelectionContract(
  routeSelectionContractInputs,
);
const renderNodePolicyInputs = {
  renderNodeAccessible: runAsAccess.renderNodeAccessible,
  kgslAccessible: runAsAccess.kgslAccessible,
  renderNodeMeta,
  customRunAsPlatformCount: loaderNamespaceContractInputs.customRunAsPlatformCount,
  customRunAsDeviceCount: loaderNamespaceContractInputs.customRunAsDeviceCount,
  customHasExternalMemory:
    typeof runAsDevice?.hasExternalMemory === 'boolean' ? runAsDevice.hasExternalMemory : null,
  customHasSvm: typeof runAsDevice?.hasSvm === 'boolean' ? runAsDevice.hasSvm : null,
};
const renderNodePolicyVerdict = classifyRenderNodePolicyContract(renderNodePolicyInputs);
const stagedDependencyChainInputs = {
  runAsStageStatus: runAsStage.status,
  customRunAsPlatformCount: loaderNamespaceContractInputs.customRunAsPlatformCount,
  missingLibs: findMissingLibsFromForensicsArtifacts(forensicsOutputDir, endpoint),
  vendorAbiMismatch: loaderNamespaceContractVerdict.classification === 'vendor_abi_mismatch',
};
const stagedDependencyChainVerdict = classifyStagedDependencyChainContract(
  stagedDependencyChainInputs,
);
const baselineProbe = JSON.stringify(runAsProbeJson || {});
const allowProbe = JSON.stringify(allowInvalidProbeJson || {});
const baselineFrontier = JSON.stringify(runAsFrontierJson || {});
const allowFrontier = JSON.stringify(allowInvalidFrontierJson || {});
const spirvCapabilityInputs = {
  genericPointerWarningSeen:
    `${customProbeRunAs.stderr || ''}\n${customFrontierRunAs.stderr || ''}`.includes(
      'SpvCapabilityGenericPointer',
    ),
  allowInvalidSpirvChangesProbe: baselineProbe !== allowProbe,
  allowInvalidSpirvChangesFrontier: baselineFrontier !== allowFrontier,
};
const spirvCapabilityVerdict = classifySpirvCapabilityContract(spirvCapabilityInputs);
const turnipKgslSourceText = run(
  'sh',
  [
    '-lc',
    "rg -n \"struct kgsl_syncobj|kgsl_syncobj_(init|export|import|wait|merge|dup)\" /data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/freedreno/vulkan/tu_knl_kgsl.cc || true",
  ],
  { encoding: 'utf8' },
);
const turnipKgslSource = `${turnipKgslSourceText.stdout || ''}\n${turnipKgslSourceText.stderr || ''}`;
const syncobjSubstrateInputs = {
  freedrenoHasSyncobj: semaphoreTrace.freedrenoHasSyncobj,
  freedrenoFenceSignal: semaphoreTrace.freedrenoFenceSignal,
  freedrenoSemaphoreCreate: semaphoreTrace.freedrenoSemaphoreCreate,
  turnipKgslSyncobjSourcePresent: turnipKgslSource.includes('struct kgsl_syncobj'),
  turnipKgslSyncobjExportPresent: turnipKgslSource.includes('kgsl_syncobj_export'),
  turnipKgslSyncobjImportPresent: turnipKgslSource.includes('kgsl_syncobj_import'),
  turnipKgslSyncobjWaitPresent: turnipKgslSource.includes('kgsl_syncobj_wait'),
};
const syncobjSubstrateVerdict = classifySyncobjSubstrateContract(syncobjSubstrateInputs);
const rusticlIcdPath = path.join(
  mesaRoot,
  'src/gallium/frontends/rusticl/api/icd.rs',
);
const rusticlApiDevicePath = path.join(
  mesaRoot,
  'src/gallium/frontends/rusticl/api/device.rs',
);
const rusticlApiCommandBufferPath = path.join(
  mesaRoot,
  'src/gallium/frontends/rusticl/api/command_buffer.rs',
);
const rusticlCoreDevicePath = path.join(
  mesaRoot,
  'src/gallium/frontends/rusticl/core/device.rs',
);
const rusticlCoreCommandBufferPath = path.join(
  mesaRoot,
  'src/gallium/frontends/rusticl/core/command_buffer.rs',
);
const clExtHeaderPath = path.join(mesaRoot, 'include/CL/cl_ext.h');
const llvmpipeScreenPath = path.join(mesaRoot, 'src/gallium/drivers/llvmpipe/lp_screen.c');
const nouveauScreenPath = path.join(mesaRoot, 'src/gallium/drivers/nouveau/nvc0/nvc0_screen.c');
const freedrenoResourcePath = path.join(
  mesaRoot,
  'src/gallium/drivers/freedreno/freedreno_resource.c',
);
const commandBufferSubstrateInputs = {
  runtimeReady:
    typeof runAsDevice?.hasCommandBuffer === 'boolean' &&
    typeof runAsSmokeCaps.commandBuffer === 'boolean'
      ? runAsDevice.hasCommandBuffer && runAsSmokeCaps.commandBuffer
      : null,
  headerDeclared: sourceContains(clExtHeaderPath, 'clCreateCommandBufferKHR'),
  rusticlIcdEntrypointsPresent: sourceContains(rusticlIcdPath, 'clCreateCommandBufferKHR'),
  rusticlApiSymbolsPresent:
    sourceContains(rusticlApiCommandBufferPath, 'clCreateCommandBufferKHR') ||
    sourceContains(rusticlApiDevicePath, 'CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR') ||
    sourceContains(rusticlApiDevicePath, 'cl_khr_command_buffer'),
  rusticlCoreSubsystemPresent:
    sourceContains(rusticlCoreCommandBufferPath, 'pub struct CommandBuffer') ||
    sourceContains(rusticlCoreDevicePath, 'command_buffer'),
};
const commandBufferSubstrateVerdict = classifyCommandBufferSubstrateContract(
  commandBufferSubstrateInputs,
);
const systemSvmSubstrateInputs = {
  runtimeFineSystemReady:
    typeof runAsDevice?.hasSvmFine === 'boolean' &&
    typeof runAsDevice?.hasSvmAtomics === 'boolean' &&
    typeof runAsSmokeCaps.svmFine === 'boolean' &&
    typeof runAsSmokeCaps.svmAtomics === 'boolean'
      ? runAsDevice.hasSvmFine &&
        runAsDevice.hasSvmAtomics &&
        runAsSmokeCaps.svmFine &&
        runAsSmokeCaps.svmAtomics
      : null,
  runtimeCoarseOnly:
    typeof runAsDevice?.hasSvmCoarse === 'boolean' &&
    typeof runAsDevice?.hasSvmFine === 'boolean' &&
    typeof runAsDevice?.hasSvmAtomics === 'boolean'
      ? runAsDevice.hasSvmCoarse && !runAsDevice.hasSvmFine && !runAsDevice.hasSvmAtomics
      : null,
  freedrenoVmHooksPresent:
    sourceContains(freedrenoResourcePath, 'pscreen->alloc_vm = fd_alloc_vm;') &&
    sourceContains(freedrenoResourcePath, 'pscreen->resource_assign_vma = fd_resource_assign_vma;'),
  rusticlGateBoundToSystemCap:
    sourceContains(rusticlCoreDevicePath, 'self.screen.caps().system_svm') &&
    sourceContains(rusticlApiDevicePath, 'CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM'),
  foreignDriverSystemSvmDonorPresent:
    sourceContains(llvmpipeScreenPath, 'caps->system_svm = true;') ||
    sourceContains(nouveauScreenPath, 'caps->system_svm = screen->base.has_svm;'),
};
const systemSvmSubstrateVerdict = classifySystemSvmSubstrateContract(systemSvmSubstrateInputs);

const compiledLookupEvidence = stringsGrep(mesaLib, 'spirv(64)?-mesa3d-|/clc/');
const outName = `opencl-adb-${endpoint.replaceAll('.', '-').replace(':', '-')}-${new Date().toISOString().slice(0, 10)}.json`;
const outPath = path.join(forensicsOutputDir, outName);

const payload = {
  capturedAt: new Date().toISOString().slice(0, 10),
  adbEndpoint: endpoint,
  connect: {
    ok: connect.status === 0 || /already connected|connected to/.test(`${connect.stdout}${connect.stderr}`),
    stdout: trim(connect.stdout),
    stderr: trim(connect.stderr),
  },
  device: {
    model,
    boardPlatform,
    fingerprint,
    kernelRelease,
    shellIdentity,
  },
  systemForensics: {
    eglRenderer,
    vendorOpenclLibrary,
    vendorIcdDirCheck,
    renderNodeMeta,
    renderOpenAttempt: trim(renderOpenAttempt.stdout || renderOpenAttempt.stderr),
    dmaHeapNodes,
  },
  staging: {
    script: stageScript,
    ok: stage.status === 0,
    stdout: trim(stage.stdout),
    stderr: trim(stage.stderr),
    libclc32Pushed: pushClc32 ? pushClc32.status === 0 : false,
    libclc64Pushed: pushClc64 ? pushClc64.status === 0 : false,
    libclcTargetDir: '/data/local/tmp/tq-rusticl/clc',
    runAsStageStatus: runAsStage.status,
    runAsStageStdout: trim(runAsStage.stdout),
    runAsStageStderr: trim(runAsStage.stderr),
    stagedFiles: lines(stagedFiles.stdout),
  },
  vendorRouteAttempt: {
    probeStdout: trim(vendorProbe.stdout),
    probeStderr: trim(vendorProbe.stderr),
    probeStatus: vendorProbe.status,
    probeJson: vendorProbeJson,
    linkerList: lines(linkerVendor.stdout),
  },
  customRouteAttempt: {
    probeStdout: trim(customProbe.stdout),
    probeStderr: trim(customProbe.stderr),
    probeStatus: customProbe.status,
    probeJson: customProbeJson,
    frontierSmokeStdout: trim(customFrontierSmoke.stdout),
    frontierSmokeStderr: trim(customFrontierSmoke.stderr),
    frontierSmokeStatus: customFrontierSmoke.status,
    frontierSmokeJson: customFrontierJson,
    runAsPackage,
    runAsProbeStdout: trim(customProbeRunAs.stdout),
    runAsProbeStderr: trim(customProbeRunAs.stderr),
    runAsProbeStatus: customProbeRunAs.status,
    runAsProbeJson,
    runAsFrontierStdout: trim(customFrontierRunAs.stdout),
    runAsFrontierStderr: trim(customFrontierRunAs.stderr),
    runAsFrontierStatus: customFrontierRunAs.status,
    runAsFrontierJson,
    runAsAllowInvalidSpirvProbeStdout: trim(customProbeRunAsAllowInvalidSpirv.stdout),
    runAsAllowInvalidSpirvProbeStderr: trim(customProbeRunAsAllowInvalidSpirv.stderr),
    runAsAllowInvalidSpirvProbeStatus: customProbeRunAsAllowInvalidSpirv.status,
    runAsAllowInvalidSpirvProbeJson: tryJson(customProbeRunAsAllowInvalidSpirv.stdout),
    runAsAllowInvalidSpirvFrontierStdout: trim(customFrontierRunAsAllowInvalidSpirv.stdout),
    runAsAllowInvalidSpirvFrontierStderr: trim(customFrontierRunAsAllowInvalidSpirv.stderr),
    runAsAllowInvalidSpirvFrontierStatus: customFrontierRunAsAllowInvalidSpirv.status,
    runAsAllowInvalidSpirvFrontierJson: tryJson(customFrontierRunAsAllowInvalidSpirv.stdout),
    runAsNodeAccess,
    loaderNamespaceContract: {
      inputs: loaderNamespaceContractInputs,
      verdict: loaderNamespaceContractVerdict,
    },
    routeSelectionContract: {
      inputs: routeSelectionContractInputs,
      verdict: routeSelectionContractVerdict,
    },
    renderNodePolicyContract: {
      inputs: renderNodePolicyInputs,
      verdict: renderNodePolicyVerdict,
    },
    stagedDependencyChainContract: {
      inputs: stagedDependencyChainInputs,
      verdict: stagedDependencyChainVerdict,
    },
    spirvCapabilityContract: {
      inputs: spirvCapabilityInputs,
      verdict: spirvCapabilityVerdict,
    },
    syncobjSubstrateContract: {
      inputs: syncobjSubstrateInputs,
      verdict: syncobjSubstrateVerdict,
    },
    commandBufferSubstrateContract: {
      inputs: commandBufferSubstrateInputs,
      verdict: commandBufferSubstrateVerdict,
    },
    systemSvmSubstrateContract: {
      inputs: systemSvmSubstrateInputs,
      verdict: systemSvmSubstrateVerdict,
    },
    initializeMemoryContract: {
      inputs: initializeMemoryContractInputs,
      verdict: classifyInitializeMemoryContract(initializeMemoryContractInputs),
    },
    commandBufferContract: {
      inputs: commandBufferContractInputs,
      verdict: classifyCommandBufferContract(commandBufferContractInputs),
    },
    svmContract: {
      inputs: svmContractInputs,
      verdict: classifySvmContract(svmContractInputs),
    },
    freedrenoSemaphoreTrace: semaphoreTrace,
    externalSemaphoreContract: {
      inputs: externalSemaphoreContractInputs,
      verdict: externalSemaphoreContractVerdict,
    },
    linkerList: lines(linkerCustom.stdout),
    compiledLookupEvidence,
  },
  logcatSlice: lines(logcatSlice.stdout),
};

writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
console.log(outPath);
