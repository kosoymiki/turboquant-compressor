export type SyncobjSubstrateContractClass =
  | 'ready'
  | 'kgsl_syncobj_donor_present'
  | 'kernel_syncobj_missing_no_donor'
  | 'unknown';

export interface SyncobjSubstrateContractInputs {
  freedrenoHasSyncobj?: boolean | null;
  freedrenoFenceSignal?: boolean | null;
  freedrenoSemaphoreCreate?: boolean | null;
  turnipKgslSyncobjSourcePresent?: boolean | null;
  turnipKgslSyncobjExportPresent?: boolean | null;
  turnipKgslSyncobjImportPresent?: boolean | null;
  turnipKgslSyncobjWaitPresent?: boolean | null;
}

export interface SyncobjSubstrateVerdict {
  classification: SyncobjSubstrateContractClass;
  ready: boolean;
  summary: string;
  blockers: string[];
}

export function classifySyncobjSubstrateContract(
  inputs: SyncobjSubstrateContractInputs,
): SyncobjSubstrateVerdict {
  const blockers: string[] = [];
  if (inputs.freedrenoHasSyncobj === false) blockers.push('drm_cap_syncobj_missing');
  if (inputs.freedrenoFenceSignal === false) blockers.push('fence_signal_missing');
  if (inputs.freedrenoSemaphoreCreate === false) blockers.push('semaphore_create_missing');
  if (inputs.turnipKgslSyncobjSourcePresent !== true) blockers.push('turnip_kgsl_syncobj_source_missing');
  if (inputs.turnipKgslSyncobjExportPresent !== true) blockers.push('turnip_kgsl_syncobj_export_missing');
  if (inputs.turnipKgslSyncobjImportPresent !== true) blockers.push('turnip_kgsl_syncobj_import_missing');
  if (inputs.turnipKgslSyncobjWaitPresent !== true) blockers.push('turnip_kgsl_syncobj_wait_missing');

  if (
    inputs.freedrenoHasSyncobj === true &&
    inputs.freedrenoFenceSignal === true &&
    inputs.freedrenoSemaphoreCreate === true
  ) {
    return {
      classification: 'ready',
      ready: true,
      summary: 'The active Gallium/Freedreno route already exposes a reusable syncobj-backed semaphore substrate.',
      blockers: [],
    };
  }

  if (
    inputs.freedrenoHasSyncobj === false &&
    inputs.turnipKgslSyncobjSourcePresent === true &&
    inputs.turnipKgslSyncobjExportPresent === true &&
    inputs.turnipKgslSyncobjImportPresent === true &&
    inputs.turnipKgslSyncobjWaitPresent === true
  ) {
    return {
      classification: 'kgsl_syncobj_donor_present',
      ready: false,
      summary:
        'The active Gallium route lacks DRM syncobj, but the same Mesa tree already contains a KGSL-native syncobj substrate in Turnip that can serve as the honest donor path for a Gallium/Rusticl port.',
      blockers,
    };
  }

  if (inputs.freedrenoHasSyncobj === false) {
    return {
      classification: 'kernel_syncobj_missing_no_donor',
      ready: false,
      summary:
        'The active Gallium route lacks DRM syncobj and no full KGSL-native reusable semaphore donor was detected in-tree.',
      blockers,
    };
  }

  return {
    classification: 'unknown',
    ready: false,
    summary: 'The lower syncobj substrate remains unresolved outside the current classified buckets.',
    blockers,
  };
}
