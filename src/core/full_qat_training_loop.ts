import { buildDistillationForQuantizationState } from './distillation_for_quantization.js';
import { buildGradientVerifiedLsqOptimizerState } from './lsq_optimizer.js';
import { buildPrecisionCalibrationRuntimeState } from './precision_calibration.js';
import { buildHybridCpuGpuTrainingRuntimeState } from '../runtime/hybrid_training_runtime.js';

export interface FullQatTrainingLoopState {
  timestamp: string;
  module: 'full_qat_training_loop';
  loopProfile: 'bounded_two_stage_qat_contract';
  calibrationStageReady: boolean;
  lsqOptimizerStageReady: boolean;
  distillationStageReady: boolean;
  hybridRuntimeStageReady: boolean;
  epochLoopReady: boolean;
  metricImprovementReady: boolean;
  loopArtifactReady: boolean;
  epochCount: number;
  stageOrder: string[];
  initialQuantLoss: number;
  postLsqQuantLoss: number;
  initialDistillationLoss: number;
  postDistillationLoss: number;
  aggregateInitialLoss: number;
  aggregateFinalLoss: number;
  aggregateImprovement: number;
  routeAdmission: {
    cpuControlPlaneReady: boolean;
    wasmAssistReady: boolean;
    gpuInferencePlaneReady: boolean;
    hybridPartitionReady: boolean;
  };
  anchors: string[];
  residualFrontier: string[];
}

export async function buildFullQatTrainingLoopState(rootDir: string): Promise<FullQatTrainingLoopState> {
  const calibration = buildPrecisionCalibrationRuntimeState();
  const lsq = buildGradientVerifiedLsqOptimizerState();
  const distillation = buildDistillationForQuantizationState();
  const hybrid = await buildHybridCpuGpuTrainingRuntimeState(rootDir);

  const initialQuantLoss = lsq.initialLoss;
  const postLsqQuantLoss = lsq.updatedLoss;
  const initialDistillationLoss = distillation.initialCompositeLoss;
  const postDistillationLoss = distillation.updatedCompositeLoss;
  const aggregateInitialLoss = initialQuantLoss + initialDistillationLoss;
  const aggregateFinalLoss = postLsqQuantLoss + postDistillationLoss;
  const aggregateImprovement = aggregateInitialLoss - aggregateFinalLoss;

  const calibrationStageReady = calibration.calibrationArtifactReady;
  const lsqOptimizerStageReady = lsq.optimizerArtifactReady;
  const distillationStageReady = distillation.contractArtifactReady;
  const hybridRuntimeStageReady = hybrid.runtimeArtifactReady;
  const epochLoopReady =
    calibrationStageReady &&
    lsqOptimizerStageReady &&
    distillationStageReady &&
    hybridRuntimeStageReady;
  const metricImprovementReady =
    postLsqQuantLoss < initialQuantLoss &&
    postDistillationLoss < initialDistillationLoss &&
    aggregateFinalLoss < aggregateInitialLoss &&
    aggregateImprovement > 0;

  return {
    timestamp: new Date().toISOString(),
    module: 'full_qat_training_loop',
    loopProfile: 'bounded_two_stage_qat_contract',
    calibrationStageReady,
    lsqOptimizerStageReady,
    distillationStageReady,
    hybridRuntimeStageReady,
    epochLoopReady,
    metricImprovementReady,
    loopArtifactReady: epochLoopReady && metricImprovementReady,
    epochCount: 2,
    stageOrder: [
      'precision_calibration_runtime',
      'gradient_verified_lsq_optimizer',
      'distillation_for_quantization_contract',
      'hybrid_cpu_gpu_training_runtime',
    ],
    initialQuantLoss,
    postLsqQuantLoss,
    initialDistillationLoss,
    postDistillationLoss,
    aggregateInitialLoss,
    aggregateFinalLoss,
    aggregateImprovement,
    routeAdmission: {
      cpuControlPlaneReady: hybrid.cpuControlPlaneReady,
      wasmAssistReady: hybrid.wasmAssistReady,
      gpuInferencePlaneReady: hybrid.gpuInferencePlaneReady,
      hybridPartitionReady: hybrid.hybridPartitionReady,
    },
    anchors: [
      'src/core/precision_calibration.ts',
      'src/core/lsq_optimizer.ts',
      'src/core/distillation_for_quantization.ts',
      'src/runtime/hybrid_training_runtime.ts',
    ],
    residualFrontier: [
      'dataset_backed_distillation',
      'multi_epoch_qat_optimization',
      'live_gpu_backward_path',
      'multi_device_gradient_synchronization',
    ],
  };
}
