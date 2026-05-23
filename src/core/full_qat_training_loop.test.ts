import { mkdirSync, writeFileSync } from 'fs';
import { join } from 'path';
import { buildFullQatTrainingLoopState } from './full_qat_training_loop.js';

describe('buildFullQatTrainingLoopState', () => {
  test('materializes a bounded QAT loop contract with improving aggregate loss', async () => {
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

    const state = await buildFullQatTrainingLoopState(rootDir);

    expect(state.module).toBe('full_qat_training_loop');
    expect(state.calibrationStageReady).toBe(true);
    expect(state.lsqOptimizerStageReady).toBe(true);
    expect(state.distillationStageReady).toBe(true);
    expect(state.hybridRuntimeStageReady).toBe(true);
    expect(state.epochLoopReady).toBe(true);
    expect(state.metricImprovementReady).toBe(true);
    expect(state.loopArtifactReady).toBe(true);
    expect(state.aggregateFinalLoss).toBeLessThan(state.aggregateInitialLoss);
  }, 15000);
});
