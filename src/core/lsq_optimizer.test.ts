import { buildGradientVerifiedLsqOptimizerState } from './lsq_optimizer.js';

describe('buildGradientVerifiedLsqOptimizerState', () => {
  test('verifies LSQ-style step-size gradient and reduces loss after one update', () => {
    const state = buildGradientVerifiedLsqOptimizerState();

    expect(state.module).toBe('gradient_verified_lsq_optimizer');
    expect(state.gradientVerificationReady).toBe(true);
    expect(state.optimizerStepReady).toBe(true);
    expect(state.lossReductionReady).toBe(true);
    expect(state.optimizerArtifactReady).toBe(true);
    expect(state.updatedLoss).toBeLessThan(state.initialLoss);
    expect(state.relativeGradientError).toBeLessThan(0.55);
    expect(state.cosineSimilarityAfterUpdate).toBeGreaterThan(0.99);
  });
});
