import { buildDistillationForQuantizationState } from './distillation_for_quantization.js';

describe('buildDistillationForQuantizationState', () => {
  test('materializes a bounded teacher-student distillation contract and reduces composite loss', () => {
    const state = buildDistillationForQuantizationState();

    expect(state.module).toBe('distillation_for_quantization_contract');
    expect(state.teacherStudentPairReady).toBe(true);
    expect(state.softenedTargetReady).toBe(true);
    expect(state.distillationLossReady).toBe(true);
    expect(state.studentUpdateReady).toBe(true);
    expect(state.contractArtifactReady).toBe(true);
    expect(state.updatedCompositeLoss).toBeLessThan(state.initialCompositeLoss);
    expect(state.updatedKlMean).toBeLessThan(state.initialKlMean);
  });
});
