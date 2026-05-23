import {
  classifyRenderNodePolicyContract,
  classifySpirvCapabilityContract,
  classifyStagedDependencyChainContract,
} from './kernel_blocker_contracts.js';

describe('kernel blocker contracts', () => {
  test('classifies render-node policy gate', () => {
    const verdict = classifyRenderNodePolicyContract({
      renderNodeAccessible: false,
      kgslAccessible: true,
      renderNodeMeta: '/dev/dri/renderD128',
    });
    expect(verdict.classification).toBe('render_node_policy_gate');
    expect(verdict.ready).toBe(false);
  });

  test('classifies kgsl route as operational when custom route already enumerates a device', () => {
    const verdict = classifyRenderNodePolicyContract({
      renderNodeAccessible: false,
      kgslAccessible: true,
      renderNodeMeta: '/dev/dri/renderD128',
      customRunAsPlatformCount: 1,
      customRunAsDeviceCount: 1,
      customHasExternalMemory: true,
      customHasSvm: true,
    });
    expect(verdict.classification).toBe('kgsl_route_operational');
    expect(verdict.ready).toBe(true);
  });

  test('classifies staged dependency chain as ready when platform enumerates', () => {
    const verdict = classifyStagedDependencyChainContract({
      runAsStageStatus: 0,
      customRunAsPlatformCount: 1,
      missingLibs: [],
      vendorAbiMismatch: true,
    });
    expect(verdict.classification).toBe('ready');
    expect(verdict.ready).toBe(true);
  });

  test('classifies generic pointer as warning-only when allow_invalid does not change outcome', () => {
    const verdict = classifySpirvCapabilityContract({
      genericPointerWarningSeen: true,
      allowInvalidSpirvChangesProbe: false,
      allowInvalidSpirvChangesFrontier: false,
    });
    expect(verdict.classification).toBe('warning_only_generic_pointer');
    expect(verdict.ready).toBe(true);
  });
});
