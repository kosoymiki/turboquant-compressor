import {
  classifyCommandBufferSubstrateContract,
  classifySystemSvmSubstrateContract,
} from './remaining_substrate_contracts.js';

describe('remaining substrate contracts', () => {
  test('classifies system SVM as VM-only with foreign donor present', () => {
    const verdict = classifySystemSvmSubstrateContract({
      runtimeFineSystemReady: false,
      runtimeFineBufferOnly: true,
      runtimeCoarseOnly: false,
      freedrenoVmHooksPresent: true,
      rusticlGateBoundToSystemCap: true,
      foreignDriverSystemSvmDonorPresent: true,
    });

    expect(verdict.classification).toBe('vm_only_foreign_donor_present');
    expect(verdict.ready).toBe(false);
  });

  test('classifies command buffer as headers-only when Rusticl subsystem is absent', () => {
    const verdict = classifyCommandBufferSubstrateContract({
      runtimeReady: false,
      headerDeclared: true,
      rusticlIcdEntrypointsPresent: false,
      rusticlApiSymbolsPresent: false,
      rusticlCoreSubsystemPresent: false,
    });

    expect(verdict.classification).toBe('headers_only_subsystem_absent');
    expect(verdict.ready).toBe(false);
  });

  test('classifies command buffer as source-present but runtime-disabled when subsystem is in-tree', () => {
    const verdict = classifyCommandBufferSubstrateContract({
      runtimeReady: false,
      headerDeclared: true,
      rusticlIcdEntrypointsPresent: true,
      rusticlApiSymbolsPresent: true,
      rusticlCoreSubsystemPresent: true,
    });

    expect(verdict.classification).toBe('source_subsystem_present_runtime_disabled');
    expect(verdict.ready).toBe(false);
  });
});
