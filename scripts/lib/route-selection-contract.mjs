export function classifyLoaderNamespaceContract(inputs) {
  const blockers = [];
  const vendorStderr = `${inputs.vendorProbeStderr || ''}`;
  const vendorAbiMismatch =
    vendorStderr.includes('CANNOT LINK EXECUTABLE') &&
    vendorStderr.includes('cannot locate symbol');
  const customReady =
    (inputs.customRunAsPlatformCount ?? 0) > 0 &&
    (inputs.customRunAsDeviceCount ?? 0) > 0 &&
    inputs.customRunAsProbeStatus === 0 &&
    inputs.customLinkerUsesStagedOpenCl === true;

  if (vendorAbiMismatch) blockers.push('vendor_abi_mismatch');
  if (inputs.runAsStageStatus !== 0) blockers.push('run_as_stage_failed');
  if (inputs.customLinkerUsesStagedOpenCl === false) blockers.push('staged_loader_not_selected');
  if (inputs.renderNodeAccessible === false) blockers.push('render_node_policy_gate');
  if (inputs.kgslAccessible === true && inputs.renderNodeAccessible === false) {
    blockers.push('kgsl_access_without_render_access');
  }

  if (customReady) {
    if (vendorAbiMismatch) {
      return {
        classification: 'vendor_abi_mismatch',
        ready: true,
        summary:
          'The staged custom loader path is live, while the vendor route is ABI-broken for this executable shape.',
        blockers,
      };
    }
    if (inputs.renderNodeAccessible === false) {
      return {
        classification: 'render_node_policy_gate',
        ready: true,
        summary:
          'The staged custom loader path is live and enumerates devices, but render-node policy remains a lower runtime gate below loader/namespace closure.',
        blockers,
      };
    }
    return {
      classification: 'ready',
      ready: true,
      summary: 'The staged custom loader and namespace route is live on the active device.',
      blockers,
    };
  }

  if (
    inputs.runAsStageStatus === 0 &&
    (inputs.customRunAsPlatformCount ?? 0) === 0 &&
    inputs.customRunAsProbeStatus === 0
  ) {
    return {
      classification: 'namespace_or_icd_blocked',
      ready: false,
      summary:
        'The staged runtime launches, but platform enumeration does not materialize, indicating unresolved loader/ICD/namespace closure on the active route.',
      blockers,
    };
  }

  if (inputs.runAsStageStatus !== 0) {
    return {
      classification: 'dependency_chain_incomplete',
      ready: false,
      summary:
        'The staged runtime was not installed cleanly into the run-as context, so loader and dependency closure is incomplete.',
      blockers,
    };
  }

  return {
    classification: 'unknown',
    ready: false,
    summary: 'Loader/namespace closure remains unresolved outside the current classified buckets.',
    blockers,
  };
}

export function classifyRouteSelectionContract(inputs) {
  const blockers = [];
  if (inputs.vendorLoader === 'vendor_abi_mismatch') blockers.push('vendor_abi_mismatch');
  if (
    inputs.customLoader !== 'ready' &&
    inputs.customLoader !== 'vendor_abi_mismatch' &&
    inputs.customLoader !== 'render_node_policy_gate'
  ) {
    blockers.push(`custom_loader_${inputs.customLoader}`);
  }

  const customReady =
    (inputs.customRunAsPlatformCount ?? 0) > 0 && (inputs.customRunAsDeviceCount ?? 0) > 0;

  if (customReady && inputs.vendorLoader === 'vendor_abi_mismatch') {
    return {
      classification: 'custom_preferred_vendor_abi_blocked',
      ready: true,
      summary:
        'The custom staged route is the only honest executable path for this workload shape; the vendor route is ABI-blocked for the current CLI binary.',
      blockers,
    };
  }

  if (customReady && inputs.vendorProbeUsable !== true) {
    return {
      classification: 'custom_ready_vendor_diagnostic_only',
      ready: true,
      summary:
        'The custom staged route is live and should remain primary; the vendor route can only be treated as a diagnostic comparison lane.',
      blockers,
    };
  }

  if (!customReady && inputs.vendorProbeUsable === true) {
    return {
      classification: 'vendor_only_custom_blocked',
      ready: false,
      summary:
        'Only the vendor route is currently usable; the custom route remains blocked below loader or runtime closure.',
      blockers,
    };
  }

  if (customReady && inputs.vendorProbeUsable === true) {
    return {
      classification: 'custom_and_vendor_divergent',
      ready: true,
      summary:
        'Both routes are live, but they must remain capability-separated and cannot be collapsed into one truth surface.',
      blockers,
    };
  }

  return {
    classification: 'unknown',
    ready: false,
    summary: 'Route selection remains unresolved outside the current classified buckets.',
    blockers,
  };
}
