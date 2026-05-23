import { initWasmBackend, isWasmReady } from '../native/wasm_backend.js';
import { loadRuntimeContractBundle, runtimeEvidenceExists } from '../native/runtime_contracts.js';
import { probeBackends } from './backend_probe.js';

export interface HybridRuntimePhaseRoute {
  phase: string;
  primaryLane: 'native_cpu_contract' | 'native_opencl_contract';
  fallbackLane: 'native_cpu_contract' | 'wasm_cpu' | 'none';
  readiness: boolean;
  evidenceSource: string;
}

export interface HybridCpuGpuTrainingRuntimeState {
  timestamp: string;
  module: 'hybrid_cpu_gpu_training_runtime';
  runtimeProfile: 'artifact_first_phase_partition_contract';
  cpuControlPlaneReady: boolean;
  wasmAssistReady: boolean;
  gpuInferencePlaneReady: boolean;
  hybridPartitionReady: boolean;
  runtimeArtifactReady: boolean;
  executionOwner: 'native_cli_contract' | 'diagnostic_runtime';
  nativeBackend: string;
  recommendedBackend: string;
  productionBackendAllowed: boolean;
  wasmInitialized: boolean;
  phaseRoutes: HybridRuntimePhaseRoute[];
  openclRoute: string | null;
  inferenceContractClassification: string | null;
  pagedKvContractClassification: string | null;
  activationStepSizeMean: number;
  weightStepSizeMean: number;
  kvGroupStepSizeMean: number;
  warnings: string[];
  anchors: string[];
  residualFrontier: string[];
}

export async function buildHybridCpuGpuTrainingRuntimeState(rootDir: string): Promise<HybridCpuGpuTrainingRuntimeState> {
  const backendProbe = probeBackends({ deep: false, timeoutMs: 500 });
  const wasmInitialized = await initWasmBackend();
  const wasmReady = wasmInitialized && isWasmReady();
  const contractBundle = loadRuntimeContractBundle(rootDir);
  const { paths, inferenceRuntimeContract, pagedKvRuntimeContract, precisionCalibrationState } = contractBundle;

  const gpuInferencePlaneReady =
    runtimeEvidenceExists(paths.inferenceRuntimeContractPath) &&
    runtimeEvidenceExists(paths.pagedKvRuntimeContractPath) &&
    inferenceRuntimeContract?.classification === 'module_ready' &&
    pagedKvRuntimeContract?.classification === 'module_ready';

  const nativeContractOwnerReady = backendProbe.executionOwner === 'native_cli_contract';
  const cpuControlPlaneReady = nativeContractOwnerReady;
  const wasmAssistReady = wasmReady;

  const phaseRoutes: HybridRuntimePhaseRoute[] = [
    {
      phase: 'observer_calibration_and_optimizer_control',
      primaryLane: 'native_cpu_contract',
      fallbackLane: 'none',
      readiness: cpuControlPlaneReady,
      evidenceSource: `${paths.precisionCalibrationStatePath} + backend_probe.nativeBackend`,
    },
    {
      phase: 'teacher_student_logit_math',
      primaryLane: 'native_cpu_contract',
      fallbackLane: wasmAssistReady ? 'wasm_cpu' : 'native_cpu_contract',
      readiness: cpuControlPlaneReady,
      evidenceSource: 'shared-contract admission + deterministic distillation contract',
    },
    {
      phase: 'prefill_decode_attention_execution',
      primaryLane: 'native_opencl_contract',
      fallbackLane: wasmAssistReady ? 'wasm_cpu' : 'native_cpu_contract',
      readiness: gpuInferencePlaneReady,
      evidenceSource: `${paths.inferenceRuntimeContractPath} + ${paths.pagedKvRuntimeContractPath}`,
    },
  ];

  const hybridPartitionReady =
    cpuControlPlaneReady &&
    phaseRoutes.every((route) => route.readiness) &&
    phaseRoutes.some((route) => route.primaryLane === 'native_opencl_contract' && route.readiness) &&
    phaseRoutes.some((route) => route.primaryLane === 'native_cpu_contract' && route.readiness);

  return {
    timestamp: new Date().toISOString(),
    module: 'hybrid_cpu_gpu_training_runtime',
    runtimeProfile: 'artifact_first_phase_partition_contract',
    cpuControlPlaneReady,
    wasmAssistReady,
    gpuInferencePlaneReady,
    hybridPartitionReady,
    runtimeArtifactReady: cpuControlPlaneReady && gpuInferencePlaneReady && hybridPartitionReady,
    executionOwner: backendProbe.executionOwner,
    nativeBackend: backendProbe.nativeBackend,
    recommendedBackend: backendProbe.recommendedBackend,
    productionBackendAllowed: backendProbe.productionBackendAllowed,
    wasmInitialized,
    phaseRoutes,
    openclRoute: inferenceRuntimeContract?.route ?? null,
    inferenceContractClassification: inferenceRuntimeContract?.classification ?? null,
    pagedKvContractClassification: pagedKvRuntimeContract?.classification ?? null,
    activationStepSizeMean: precisionCalibrationState?.activationStepSizeMean ?? 0,
    weightStepSizeMean: precisionCalibrationState?.weightStepSizeMean ?? 0,
    kvGroupStepSizeMean: precisionCalibrationState?.kvGroupStepSizeMean ?? 0,
    warnings: [
      ...backendProbe.warnings,
      ...(wasmAssistReady ? [] : ['WASM backend unavailable; native/shared-contract runtime stays on native CPU control lane']),
      ...(gpuInferencePlaneReady ? [] : [`OpenCL inference contracts missing or not module_ready: ${paths.inferenceRuntimeContractPath}, ${paths.pagedKvRuntimeContractPath}`]),
    ],
    anchors: [
      'src/runtime/backend_probe.ts',
      'src/native/runtime_contracts.ts',
      'src/native/wasm_backend.ts',
      paths.inferenceRuntimeContractPath,
      paths.pagedKvRuntimeContractPath,
      paths.precisionCalibrationStatePath,
    ],
    residualFrontier: [
      'full_qat_training_loop',
      'dataset_backed_distillation',
      'multi_device_gradient_synchronization',
      'live_gpu_backward_path',
    ],
  };
}
