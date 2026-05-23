#!/usr/bin/env node
import { spawnSync } from 'child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const forensicsDir = join(rootDir, 'forensics');
const adrenoDir = join(forensicsDir, 'adreno');
const safeRunner = join(rootDir, 'scripts', 'safe-runtime-pack-run.sh');
mkdirSync(adrenoDir, { recursive: true });

function loadJson(path) {
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch {
    return null;
  }
}

const timeoutSeconds = process.argv.includes('--timeout-seconds')
  ? parseInt(process.argv[process.argv.indexOf('--timeout-seconds') + 1], 10) || 20
  : 20;

const env = {
  ...process.env,
  TQ_OPENCL_TIMEOUT_SECONDS: String(timeoutSeconds),
  TQ_OPENCL_SILENCE_STDERR: '1',
};

const run = spawnSync(safeRunner, ['runtime-scheduler-smoke'], {
  cwd: rootDir,
  encoding: 'utf-8',
  timeout: Math.max(30000, timeoutSeconds * 2000),
  env,
});

let stdoutJson = null;
try {
  stdoutJson = JSON.parse(run.stdout || '');
} catch {}

const runtimeStatePath = join(forensicsDir, 'runtime-scheduler-state.json');
const runtimeState = loadJson(runtimeStatePath);
const autotuneCachePath = join(forensicsDir, 'autotune-cache.json');
const kernelCacheDir = join(forensicsDir, 'kernel-cache');
const precompiledDir = join(forensicsDir, 'precompiled-kernel-library');

const artifact = {
  timestamp: new Date().toISOString(),
  type: 'opencl_runtime_scheduler_contract',
  route: 'mesa_rusticl_kgsl',
  statusCode: run.status,
  smokePresent: !!stdoutJson,
  smokeStatus: stdoutJson?.status || null,
  runtimeStatePresent: !!runtimeState,
  runtimeStatePath,
  compilePlane: {
    compileIntents: runtimeState?.compileIntents ?? stdoutJson?.compileIntents ?? 0,
    dedupedCompileIntents: runtimeState?.dedupedCompileIntents ?? stdoutJson?.dedupedCompileIntents ?? 0,
    programReuseHits: runtimeState?.programReuseHits ?? stdoutJson?.programReuseHits ?? 0,
    sourceBuilds: runtimeState?.sourceBuilds ?? stdoutJson?.sourceBuilds ?? 0,
    ilBuilds: runtimeState?.ilBuilds ?? stdoutJson?.ilBuilds ?? 0,
    binaryBuilds: runtimeState?.binaryBuilds ?? stdoutJson?.binaryBuilds ?? 0,
    asyncBuildCallbacks: runtimeState?.asyncBuildCallbacks ?? stdoutJson?.asyncBuildCallbacks ?? 0,
    kernelReadyCount: runtimeState?.kernelReadyCount ?? stdoutJson?.kernelReadyCount ?? 0,
  },
  cachePlane: {
    binaryCacheHits: runtimeState?.binaryCacheHits ?? stdoutJson?.binaryCacheHits ?? 0,
    precompiledBinaryHits: runtimeState?.precompiledBinaryHits ?? stdoutJson?.precompiledBinaryHits ?? 0,
    autotuneEntries: runtimeState?.autotuneEntries ?? stdoutJson?.autotuneEntries ?? 0,
    autotuneCachePresent: existsSync(autotuneCachePath),
    kernelCacheDirPresent: existsSync(kernelCacheDir),
    precompiledLibraryPresent: existsSync(precompiledDir),
  },
  memoryPlan: runtimeState?.memoryPlan || stdoutJson?.memoryPlan || null,
  planner: runtimeState?.planner || stdoutJson?.planner || null,
  compileOrchestrator: {
    preferredBuildPath:
      runtimeState?.planner?.preferredBuildPath ??
      stdoutJson?.planner?.preferredBuildPath ??
      null,
    compileQueueDepth:
      runtimeState?.planner?.compileQueueDepth ??
      stdoutJson?.planner?.compileQueueDepth ??
      0,
    admittedCompileCount:
      runtimeState?.planner?.admittedCompileCount ??
      stdoutJson?.planner?.admittedCompileCount ??
      0,
    deferredCompileCount:
      runtimeState?.planner?.deferredCompileCount ??
      stdoutJson?.planner?.deferredCompileCount ??
      0,
    speculativeCandidateCount:
      runtimeState?.planner?.speculativeCandidateCount ??
      stdoutJson?.planner?.speculativeCandidateCount ??
      0,
    compileCandidates: runtimeState?.compileCandidates ?? stdoutJson?.compileCandidates ?? [],
  },
  executionAdmission: {
    executionQueueSlots:
      runtimeState?.planner?.executionQueueSlots ??
      stdoutJson?.planner?.executionQueueSlots ??
      0,
    executionBudgetSlots:
      runtimeState?.planner?.executionBudgetSlots ??
      stdoutJson?.planner?.executionBudgetSlots ??
      0,
    compileBudgetSlots:
      runtimeState?.planner?.compileBudgetSlots ??
      stdoutJson?.planner?.compileBudgetSlots ??
      0,
    compileDeferredForExecution:
      runtimeState?.planner?.compileDeferredForExecution ??
      stdoutJson?.planner?.compileDeferredForExecution ??
      0,
    dispatchBudgetPermille:
      runtimeState?.planner?.dispatchBudgetPermille ??
      stdoutJson?.planner?.dispatchBudgetPermille ??
      0,
    compileBudgetPermille:
      runtimeState?.planner?.compileBudgetPermille ??
      stdoutJson?.planner?.compileBudgetPermille ??
      0,
    executionAdmission:
      runtimeState?.planner?.executionAdmission ??
      stdoutJson?.planner?.executionAdmission ??
      null,
    executionAdmissionReason:
      runtimeState?.planner?.executionAdmissionReason ??
      stdoutJson?.planner?.executionAdmissionReason ??
      null,
    arbitrationMode:
      runtimeState?.planner?.arbitrationMode ??
      stdoutJson?.planner?.arbitrationMode ??
      null,
    arbitrationReason:
      runtimeState?.planner?.arbitrationReason ??
      stdoutJson?.planner?.arbitrationReason ??
      null,
    multiQueueMode:
      runtimeState?.planner?.multiQueueMode ??
      stdoutJson?.planner?.multiQueueMode ??
      null,
    queueGuardTriggered:
      runtimeState?.planner?.queueGuardTriggered ??
      stdoutJson?.planner?.queueGuardTriggered ??
      null,
    queueGuardReason:
      runtimeState?.planner?.queueGuardReason ??
      stdoutJson?.planner?.queueGuardReason ??
      null,
    compileExecuteBalancePermille:
      runtimeState?.planner?.compileExecuteBalancePermille ??
      stdoutJson?.planner?.compileExecuteBalancePermille ??
      0,
    compileQueueAgeTicks:
      runtimeState?.planner?.compileQueueAgeTicks ??
      stdoutJson?.planner?.compileQueueAgeTicks ??
      0,
    dispatchQueueAgeTicks:
      runtimeState?.planner?.dispatchQueueAgeTicks ??
      stdoutJson?.planner?.dispatchQueueAgeTicks ??
      0,
    starvationPreventedCount:
      runtimeState?.planner?.starvationPreventedCount ??
      stdoutJson?.planner?.starvationPreventedCount ??
      0,
    fairnessDebtPermille:
      runtimeState?.planner?.fairnessDebtPermille ??
      stdoutJson?.planner?.fairnessDebtPermille ??
      0,
    compileQueueAgeBand:
      runtimeState?.planner?.compileQueueAgeBand ??
      stdoutJson?.planner?.compileQueueAgeBand ??
      null,
    dispatchQueueAgeBand:
      runtimeState?.planner?.dispatchQueueAgeBand ??
      stdoutJson?.planner?.dispatchQueueAgeBand ??
      null,
    latencyAdmission:
      runtimeState?.planner?.latencyAdmission ??
      stdoutJson?.planner?.latencyAdmission ??
      null,
    latencyAdmissionReason:
      runtimeState?.planner?.latencyAdmissionReason ??
      stdoutJson?.planner?.latencyAdmissionReason ??
      null,
    dispatchUrgency:
      runtimeState?.planner?.dispatchUrgency ??
      stdoutJson?.planner?.dispatchUrgency ??
      null,
    classAwareDispatchMode:
      runtimeState?.planner?.classAwareDispatchMode ??
      stdoutJson?.planner?.classAwareDispatchMode ??
      null,
    redistributionMode:
      runtimeState?.planner?.redistributionMode ??
      stdoutJson?.planner?.redistributionMode ??
      null,
    redistributionReason:
      runtimeState?.planner?.redistributionReason ??
      stdoutJson?.planner?.redistributionReason ??
      null,
    starvationPreventionMode:
      runtimeState?.planner?.starvationPreventionMode ??
      stdoutJson?.planner?.starvationPreventionMode ??
      null,
    starvationPreventionReason:
      runtimeState?.planner?.starvationPreventionReason ??
      stdoutJson?.planner?.starvationPreventionReason ??
      null,
    starvationTargetKernel:
      runtimeState?.planner?.starvationTargetKernel ??
      stdoutJson?.planner?.starvationTargetKernel ??
      null,
    throttleTargetKernel:
      runtimeState?.planner?.throttleTargetKernel ??
      stdoutJson?.planner?.throttleTargetKernel ??
      null,
    pressureRecoveryAction:
      runtimeState?.planner?.pressureRecoveryAction ??
      stdoutJson?.planner?.pressureRecoveryAction ??
      null,
    pressureRecoveryReason:
      runtimeState?.planner?.pressureRecoveryReason ??
      stdoutJson?.planner?.pressureRecoveryReason ??
      null,
    kernelClassPolicyCount:
      runtimeState?.planner?.kernelClassPolicyCount ??
      stdoutJson?.planner?.kernelClassPolicyCount ??
      0,
    dispatchCount:
      runtimeState?.execution?.dispatchCount ??
      stdoutJson?.execution?.dispatchCount ??
      0,
    profiledDispatchCount:
      runtimeState?.execution?.profiledDispatchCount ??
      stdoutJson?.execution?.profiledDispatchCount ??
      0,
    dispatchWaveSlots:
      runtimeState?.execution?.dispatchWaveSlots ??
      stdoutJson?.execution?.dispatchWaveSlots ??
      0,
    activeDispatchSlots:
      runtimeState?.execution?.activeDispatchSlots ??
      stdoutJson?.execution?.activeDispatchSlots ??
      0,
    queuePressurePermille:
      runtimeState?.execution?.queuePressurePermille ??
      stdoutJson?.execution?.queuePressurePermille ??
      0,
    queuePressureBand:
      runtimeState?.execution?.queuePressureBand ??
      stdoutJson?.execution?.queuePressureBand ??
      null,
    dispatchUtilizationPermille:
      runtimeState?.execution?.dispatchUtilizationPermille ??
      stdoutJson?.execution?.dispatchUtilizationPermille ??
      0,
    compileUtilizationPermille:
      runtimeState?.execution?.compileUtilizationPermille ??
      stdoutJson?.execution?.compileUtilizationPermille ??
      0,
    lastDispatchKernelName:
      runtimeState?.execution?.lastDispatchKernelName ??
      stdoutJson?.execution?.lastDispatchKernelName ??
      null,
    lastDispatchLatencyClass:
      runtimeState?.execution?.lastDispatchLatencyClass ??
      stdoutJson?.execution?.lastDispatchLatencyClass ??
      null,
    lastDispatchLatencyReason:
      runtimeState?.execution?.lastDispatchLatencyReason ??
      stdoutJson?.execution?.lastDispatchLatencyReason ??
      null,
    lastDispatchGlobalSize:
      runtimeState?.execution?.lastDispatchGlobalSize ??
      stdoutJson?.execution?.lastDispatchGlobalSize ??
      0,
    lastDispatchLocalSize:
      runtimeState?.execution?.lastDispatchLocalSize ??
      stdoutJson?.execution?.lastDispatchLocalSize ??
      0,
    executionPressureBand:
      runtimeState?.execution?.executionPressureBand ??
      stdoutJson?.execution?.executionPressureBand ??
      null,
  },
  kernelClassMatrix: {
    kernelClassPolicyCount:
      runtimeState?.planner?.kernelClassPolicyCount ??
      stdoutJson?.planner?.kernelClassPolicyCount ??
      0,
    kernelClassPolicies:
      runtimeState?.kernelClassPolicies ??
      stdoutJson?.kernelClassPolicies ??
      [],
  },
  residencyPolicy: {
    residentPressureBand:
      runtimeState?.memoryPlan?.residentPressureBand ??
      stdoutJson?.memoryPlan?.residentPressureBand ??
      null,
    evictionRecommended:
      runtimeState?.memoryPlan?.evictionRecommended ??
      stdoutJson?.memoryPlan?.evictionRecommended ??
      null,
    compactionRecommended:
      runtimeState?.memoryPlan?.compactionRecommended ??
      stdoutJson?.memoryPlan?.compactionRecommended ??
      null,
    persistentResidencyReady:
      runtimeState?.memoryPlan?.persistentResidencyReady ??
      stdoutJson?.memoryPlan?.persistentResidencyReady ??
      null,
    evictionCandidateCount:
      runtimeState?.planner?.evictionCandidateCount ??
      stdoutJson?.planner?.evictionCandidateCount ??
      0,
    evictionCandidates: runtimeState?.evictionCandidates ?? stdoutJson?.evictionCandidates ?? [],
  },
  classification:
    run.status === 0 &&
    (runtimeState?.compileIntents ?? stdoutJson?.compileIntents ?? 0) > 0 &&
    (runtimeState?.kernelReadyCount ?? stdoutJson?.kernelReadyCount ?? 0) > 0 &&
    !!(runtimeState?.planner || stdoutJson?.planner) &&
    !!(runtimeState?.memoryPlan || stdoutJson?.memoryPlan) &&
    (runtimeState?.planner?.compileQueueDepth ?? stdoutJson?.planner?.compileQueueDepth ?? 0) > 0 &&
    (runtimeState?.execution?.dispatchCount ?? stdoutJson?.execution?.dispatchCount ?? 0) > 0 &&
    (runtimeState?.planner?.kernelClassPolicyCount ?? stdoutJson?.planner?.kernelClassPolicyCount ?? 0) > 1 &&
    !!(runtimeState?.planner?.classAwareDispatchMode ?? stdoutJson?.planner?.classAwareDispatchMode)
      ? 'module_ready'
      : 'module_incomplete_or_unproven',
  stdout: (run.stdout || '').slice(0, 16000),
  stderr: (run.stderr || '').slice(0, 16000),
};

const outPath = join(adrenoDir, 'opencl-runtime-scheduler-contract.json');
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(JSON.stringify({ status: 'ok', outPath, classification: artifact.classification }, null, 2));
