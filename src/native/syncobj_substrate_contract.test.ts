import { classifySyncobjSubstrateContract } from './syncobj_substrate_contract.js';

describe('syncobj substrate contract', () => {
  test('classifies kgsl donor as present when Gallium lacks DRM syncobj but Turnip has KGSL syncobj substrate', () => {
    const verdict = classifySyncobjSubstrateContract({
      freedrenoHasSyncobj: false,
      freedrenoFenceSignal: false,
      freedrenoSemaphoreCreate: false,
      turnipKgslSyncobjSourcePresent: true,
      turnipKgslSyncobjExportPresent: true,
      turnipKgslSyncobjImportPresent: true,
      turnipKgslSyncobjWaitPresent: true,
    });

    expect(verdict.classification).toBe('kgsl_syncobj_donor_present');
    expect(verdict.ready).toBe(false);
  });
});
