import { buildPrecisionCalibrationRuntimeState } from './precision_calibration.js';

describe('buildPrecisionCalibrationRuntimeState', () => {
  test('emits a bounded calibration artifact with observer, fake-quant, and step-size planes', () => {
    const state = buildPrecisionCalibrationRuntimeState();

    expect(state.module).toBe('precision_calibration_runtime');
    expect(state.surfaceCount).toBe(3);
    expect(state.observerReady).toBe(true);
    expect(state.fakeQuantPolicyReady).toBe(true);
    expect(state.stepSizeContractReady).toBe(true);
    expect(state.calibrationArtifactReady).toBe(true);
    expect(state.surfaces.map((surface) => surface.role)).toEqual([
      'activation',
      'weight',
      'kv_value',
    ]);
    for (const surface of state.surfaces) {
      expect(surface.elementCount).toBeGreaterThan(0);
      expect(surface.fakeQuant.cosineSimilarity).toBeGreaterThan(0.9);
      expect(surface.stepSize.initialStepSize).toBeGreaterThan(0);
    }
    expect(state.betaCodebookRange.msePerCoord).toBeGreaterThan(0);
    expect(state.residualFrontier).toContain('full_qat_training_loop');
  });
});
