export type SystemSvmSubstrateContractClass =
  | 'ready'
  | 'vm_only_foreign_donor_present'
  | 'vm_only_no_system_cap'
  | 'unknown';

export type CommandBufferSubstrateContractClass =
  | 'ready'
  | 'headers_only_subsystem_absent'
  | 'source_subsystem_present_runtime_disabled'
  | 'subsystem_absent_no_headers'
  | 'unknown';

export interface SystemSvmSubstrateContractInputs {
  runtimeFineSystemReady?: boolean | null;
  runtimeFineBufferOnly?: boolean | null;
  runtimeCoarseOnly?: boolean | null;
  freedrenoVmHooksPresent?: boolean | null;
  rusticlGateBoundToSystemCap?: boolean | null;
  foreignDriverSystemSvmDonorPresent?: boolean | null;
}

export interface CommandBufferSubstrateContractInputs {
  runtimeReady?: boolean | null;
  headerDeclared?: boolean | null;
  rusticlIcdEntrypointsPresent?: boolean | null;
  rusticlApiSymbolsPresent?: boolean | null;
  rusticlCoreSubsystemPresent?: boolean | null;
}

export interface RemainingSubstrateVerdict<TClass extends string> {
  classification: TClass;
  ready: boolean;
  summary: string;
  blockers: string[];
}

export function classifySystemSvmSubstrateContract(
  inputs: SystemSvmSubstrateContractInputs,
): RemainingSubstrateVerdict<SystemSvmSubstrateContractClass> {
  const blockers: string[] = [];
  if (inputs.runtimeFineBufferOnly === true) blockers.push('runtime_fine_buffer_only');
  if (inputs.runtimeCoarseOnly === true) blockers.push('runtime_coarse_only');
  if (inputs.freedrenoVmHooksPresent !== true) blockers.push('freedreno_vm_hooks_missing');
  if (inputs.rusticlGateBoundToSystemCap !== true) blockers.push('rusticl_gate_not_confirmed');
  if (inputs.foreignDriverSystemSvmDonorPresent !== true) blockers.push('foreign_system_svm_donor_missing');

  if (inputs.runtimeFineSystemReady === true) {
    return {
      classification: 'ready',
      ready: true,
      summary: 'Fine-grain/system SVM is already live on the active route.',
      blockers: [],
    };
  }

  if (
    (inputs.runtimeCoarseOnly === true || inputs.runtimeFineBufferOnly === true) &&
    inputs.freedrenoVmHooksPresent === true &&
    inputs.rusticlGateBoundToSystemCap === true &&
    inputs.foreignDriverSystemSvmDonorPresent === true
  ) {
    return {
      classification: 'vm_only_foreign_donor_present',
      ready: false,
      summary:
        inputs.runtimeFineBufferOnly === true
          ? 'The active route now exposes fine-grain buffer SVM on top of Freedreno VM hooks, but full system SVM is still gated on pipe-screen system_svm; only foreign Mesa drivers currently advertise the donor capability.'
          : 'The active route already has VM-backed coarse SVM and VM hooks, but fine/system SVM is still gated on pipe-screen system_svm; only foreign Mesa drivers currently advertise the donor capability.',
      blockers,
    };
  }

  if (
    (inputs.runtimeCoarseOnly === true || inputs.runtimeFineBufferOnly === true) &&
    inputs.freedrenoVmHooksPresent === true
  ) {
    return {
      classification: 'vm_only_no_system_cap',
      ready: false,
      summary:
        inputs.runtimeFineBufferOnly === true
          ? 'The active route has fine-grain buffer SVM plus Freedreno VM hooks, but no honest system_svm capability substrate is exposed to Rusticl.'
          : 'The active route has VM-backed coarse SVM with Freedreno VM hooks, but no honest system_svm capability substrate is exposed to Rusticl.',
      blockers,
    };
  }

  return {
    classification: 'unknown',
    ready: false,
    summary: 'The system/fine SVM substrate remains unresolved outside the current classified buckets.',
    blockers,
  };
}

export function classifyCommandBufferSubstrateContract(
  inputs: CommandBufferSubstrateContractInputs,
): RemainingSubstrateVerdict<CommandBufferSubstrateContractClass> {
  const blockers: string[] = [];
  if (inputs.headerDeclared !== true) blockers.push('opencl_header_declaration_missing');
  if (inputs.rusticlIcdEntrypointsPresent !== true) blockers.push('rusticl_icd_entrypoints_missing');
  if (inputs.rusticlApiSymbolsPresent !== true) blockers.push('rusticl_api_surface_missing');
  if (inputs.rusticlCoreSubsystemPresent !== true) blockers.push('rusticl_core_subsystem_missing');

  if (inputs.runtimeReady === true) {
    return {
      classification: 'ready',
      ready: true,
      summary: 'The command-buffer subsystem is live on the active route.',
      blockers: [],
    };
  }

  if (
    inputs.headerDeclared === true &&
    inputs.rusticlIcdEntrypointsPresent === false &&
    inputs.rusticlApiSymbolsPresent === false &&
    inputs.rusticlCoreSubsystemPresent === false
  ) {
    return {
      classification: 'headers_only_subsystem_absent',
      ready: false,
      summary:
        'Mesa ships the OpenCL command-buffer declarations, but Rusticl still lacks ICD exports, API entrypoints, and a core command-buffer subsystem.',
      blockers,
    };
  }

  if (
    inputs.headerDeclared === true &&
    inputs.rusticlIcdEntrypointsPresent === true &&
    inputs.rusticlApiSymbolsPresent === true &&
    inputs.rusticlCoreSubsystemPresent === true
  ) {
    return {
      classification: 'source_subsystem_present_runtime_disabled',
      ready: false,
      summary:
        'Rusticl now has command-buffer headers, ICD entrypoints, API wiring, and a core subsystem in-tree, but the active route still does not advertise or execute the runtime surface.',
      blockers,
    };
  }

  if (
    inputs.headerDeclared === false &&
    inputs.rusticlIcdEntrypointsPresent === false &&
    inputs.rusticlApiSymbolsPresent === false &&
    inputs.rusticlCoreSubsystemPresent === false
  ) {
    return {
      classification: 'subsystem_absent_no_headers',
      ready: false,
      summary:
        'The command-buffer subsystem is absent end-to-end, including the expected declaration surface.',
      blockers,
    };
  }

  return {
    classification: 'unknown',
    ready: false,
    summary: 'The command-buffer substrate remains unresolved outside the current classified buckets.',
    blockers,
  };
}
