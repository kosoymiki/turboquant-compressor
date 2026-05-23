import { mkdirSync, writeFileSync } from 'fs';
import { join } from 'path';
import { buildHybridCpuGpuTrainingRuntimeState } from './hybrid_training_runtime.js';

describe('buildHybridCpuGpuTrainingRuntimeState', () => {
  test('materializes a bounded hybrid CPU+GPU phase partition contract from existing artifacts', async () => {
    const rootDir = process.cwd();
    const adrenoDir = join(rootDir, 'forensics', 'adreno');
    mkdirSync(adrenoDir, { recursive: true });
    writeFileSync(join(adrenoDir, 'opencl-inference-runtime-contract.json'), JSON.stringify({
      classification: 'module_ready',
      route: 'mesa_rusticl_kgsl',
    }));
    writeFileSync(join(adrenoDir, 'opencl-paged-kv-prefix-cache-contract.json'), JSON.stringify({
      classification: 'module_ready',
    }));
    writeFileSync(join(rootDir, 'forensics', 'precision-calibration-runtime-state.json'), JSON.stringify({
      activationStepSizeMean: 0.05,
      weightStepSizeMean: 0.11,
      kvGroupStepSizeMean: 0.09,
    }));

    const state = await buildHybridCpuGpuTrainingRuntimeState(rootDir);

    expect(state.module).toBe('hybrid_cpu_gpu_training_runtime');
    expect(state.cpuControlPlaneReady).toBe(true);
    expect(state.gpuInferencePlaneReady).toBe(true);
    expect(state.hybridPartitionReady).toBe(true);
    expect(state.runtimeArtifactReady).toBe(true);
    expect(state.phaseRoutes.map((route) => route.phase)).toEqual([
      'observer_calibration_and_optimizer_control',
      'teacher_student_logit_math',
      'prefill_decode_attention_execution',
    ]);
    expect(state.phaseRoutes.some((route) => route.primaryLane === 'native_opencl_contract')).toBe(true);
    expect(state.phaseRoutes.some((route) => route.primaryLane === 'native_cpu_contract')).toBe(true);
    expect(state.phaseRoutes.every((route) => route.primaryLane !== 'typescript_cpu')).toBe(true);
  });
});
