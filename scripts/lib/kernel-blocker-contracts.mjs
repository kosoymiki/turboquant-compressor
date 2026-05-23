export function classifyRenderNodePolicyContract(inputs) {
  const blockers = [];
  if (inputs.renderNodeAccessible === false) blockers.push('render_node_permission_denied');
  if (inputs.kgslAccessible === true) blockers.push('kgsl_accessible');
  if (inputs.renderNodeMeta && !inputs.renderNodeMeta.includes('renderD128')) blockers.push('render_node_missing');

  if (inputs.renderNodeAccessible === true) {
    return {
      classification: 'ready',
      ready: true,
      summary: 'The active route can open the DRM render node directly.',
      blockers: [],
    };
  }

  if (
    inputs.renderNodeAccessible === false &&
    inputs.kgslAccessible === true &&
    (inputs.customRunAsPlatformCount ?? 0) > 0 &&
    (inputs.customRunAsDeviceCount ?? 0) > 0
  ) {
    return {
      classification: 'kgsl_route_operational',
      ready: true,
      summary:
        'The higher DRM render-node path remains policy-blocked, but the active custom route is already operational through KGSL and should not be treated as blocked at the route-selection layer.',
      blockers,
    };
  }

  if (inputs.renderNodeAccessible === false && inputs.kgslAccessible === true) {
    return {
      classification: 'render_node_policy_gate',
      ready: false,
      summary:
        'KGSL remains reachable, but DRM render-node access is blocked by runtime policy, so Gallium/Freedreno cannot use the higher route directly.',
      blockers,
    };
  }

  if (inputs.renderNodeAccessible === false) {
    return {
      classification: 'device_node_missing',
      ready: false,
      summary: 'The active route cannot reach a usable render node.',
      blockers,
    };
  }

  return {
    classification: 'unknown',
    ready: false,
    summary: 'Render-node policy closure remains unresolved outside the current classified buckets.',
    blockers,
  };
}

export function classifyStagedDependencyChainContract(inputs) {
  const blockers = [];
  const missing = inputs.missingLibs || [];
  blockers.push(...missing.map((name) => `missing_${name}`));
  if (inputs.vendorAbiMismatch === true) blockers.push('vendor_abi_mismatch');

  if ((inputs.customRunAsPlatformCount ?? 0) > 0 && missing.length === 0 && inputs.runAsStageStatus === 0) {
    return {
      classification: 'ready',
      ready: true,
      summary: 'The staged dependency chain is materially closed for the active custom route.',
      blockers,
    };
  }

  if (missing.length > 0) {
    return {
      classification: 'dependency_chain_incomplete',
      ready: false,
      summary:
        'The staged custom runtime still lacks required shared-library closure, so Rusticl userspace cannot be treated as initialized.',
      blockers,
    };
  }

  if (inputs.vendorAbiMismatch === true) {
    return {
      classification: 'abi_mismatch_before_opencl',
      ready: false,
      summary:
        'The execution path fails at C++ ABI linkage before OpenCL route selection can even be evaluated.',
      blockers,
    };
  }

  return {
    classification: 'unknown',
    ready: false,
    summary: 'Dependency-chain closure remains unresolved outside the current classified buckets.',
    blockers,
  };
}

export function classifySpirvCapabilityContract(inputs) {
  const blockers = [];
  if (inputs.genericPointerWarningSeen === true) blockers.push('generic_pointer_warning_seen');
  if (inputs.allowInvalidSpirvChangesProbe === true) blockers.push('allow_invalid_spirv_changes_probe');
  if (inputs.allowInvalidSpirvChangesFrontier === true) blockers.push('allow_invalid_spirv_changes_frontier');

  if (inputs.genericPointerWarningSeen !== true) {
    return {
      classification: 'clean',
      ready: true,
      summary: 'No SPIR-V capability warning is present on the active route.',
      blockers: [],
    };
  }

  if (
    inputs.genericPointerWarningSeen === true &&
    inputs.allowInvalidSpirvChangesProbe !== true &&
    inputs.allowInvalidSpirvChangesFrontier !== true
  ) {
    return {
      classification: 'warning_only_generic_pointer',
      ready: true,
      summary:
        'GenericPointer is still warned by spirv_to_nir, but the allow_invalid_spirv replay does not change probe or frontier outcome, so this remains a warning-class lane rather than a proven blocker.',
      blockers,
    };
  }

  return {
    classification: 'warning_changes_runtime_outcome',
    ready: false,
    summary:
      'The GenericPointer warning changes runtime behavior when invalid SPIR-V is tolerated, so this lane must be treated as an active blocker.',
    blockers,
  };
}
