export type InitializeMemoryContractClass =
  | 'ready'
  | 'runtime_surface_missing'
  | 'context_fallback_only'
  | 'unknown';

export type CommandBufferContractClass =
  | 'ready'
  | 'runtime_surface_missing'
  | 'capability_bits_missing'
  | 'unknown';

export type SvmContractClass =
  | 'ready'
  | 'fine_buffer_only'
  | 'coarse_only'
  | 'no_svm'
  | 'unknown';

export interface InitializeMemoryContractInputs {
  probeHasInitializeMemory?: boolean | null;
  smokeHasInitializeMemory?: boolean | null;
  initContext?: boolean | null;
}

export interface CommandBufferContractInputs {
  probeHasCommandBuffer?: boolean | null;
  smokeHasCommandBuffer?: boolean | null;
  commandBufferCapabilities?: number | null;
  probeHasMutableDispatch?: boolean | null;
}

export interface SvmContractInputs {
  probeHasSvm?: boolean | null;
  probeHasSvmCoarse?: boolean | null;
  probeHasSvmFine?: boolean | null;
  probeHasSvmAtomics?: boolean | null;
  smokeHasSvm?: boolean | null;
  smokeHasSvmCoarse?: boolean | null;
  smokeHasSvmFine?: boolean | null;
  smokeHasSvmAtomics?: boolean | null;
  smokeSystemRawPointerOk?: boolean | null;
  smokeSystemAtomicOk?: boolean | null;
}

export interface FrontierContractVerdict<TClass extends string> {
  classification: TClass;
  ready: boolean;
  summary: string;
  blockers: string[];
}

export function classifyInitializeMemoryContract(
  inputs: InitializeMemoryContractInputs,
): FrontierContractVerdict<InitializeMemoryContractClass> {
  const blockers: string[] = [];
  if (inputs.probeHasInitializeMemory === false) blockers.push('extension_not_advertised');
  if (inputs.smokeHasInitializeMemory === false) blockers.push('runtime_surface_disabled');
  if (inputs.initContext === true && inputs.smokeHasInitializeMemory === false) {
    blockers.push('context_fell_back_without_initialize_memory');
  }

  if (inputs.probeHasInitializeMemory === true && inputs.smokeHasInitializeMemory === true) {
    return {
      classification: 'ready',
      ready: true,
      summary: 'Initialize-memory extension is advertised and live context initialization retains the surface.',
      blockers: [],
    };
  }

  if (inputs.initContext === true && inputs.probeHasInitializeMemory === false) {
    return {
      classification: 'runtime_surface_missing',
      ready: false,
      summary:
        'OpenCL context creation succeeds, but cl_khr_initialize_memory is not advertised on the active route, so TQ can only fall back to a plain context.',
      blockers,
    };
  }

  if (inputs.initContext === true && inputs.smokeHasInitializeMemory === false) {
    return {
      classification: 'context_fallback_only',
      ready: false,
      summary:
        'The runtime can create a context, but not one that preserves initialize-memory semantics on the active route.',
      blockers,
    };
  }

  return {
    classification: 'unknown',
    ready: false,
    summary: 'Initialize-memory closure remains blocked outside the current classified buckets.',
    blockers,
  };
}

export function classifyCommandBufferContract(
  inputs: CommandBufferContractInputs,
): FrontierContractVerdict<CommandBufferContractClass> {
  const blockers: string[] = [];
  if (inputs.probeHasCommandBuffer === false) blockers.push('extension_not_advertised');
  if (inputs.smokeHasCommandBuffer === false) blockers.push('runtime_surface_disabled');
  if (inputs.probeHasMutableDispatch === false) blockers.push('mutable_dispatch_missing');

  if (inputs.probeHasCommandBuffer === true && inputs.smokeHasCommandBuffer === true) {
    return {
      classification: 'ready',
      ready: true,
      summary:
        (inputs.commandBufferCapabilities ?? 0) !== 0
          ? 'Command-buffer extension and optional capability bits are live on the active route.'
          : 'Base command-buffer recording and replay are live on the active route; optional command-buffer capability bits remain zero because no layered optional command-buffer capabilities are advertised.',
      blockers: [],
    };
  }

  if (inputs.probeHasCommandBuffer === true && (inputs.commandBufferCapabilities ?? 0) === 0) {
    return {
      classification: 'capability_bits_missing',
      ready: false,
      summary:
        'Command-buffer extension is visible, but the driver reports zero command-buffer capability bits, so the route is not honestly usable.',
      blockers,
    };
  }

  return {
    classification: 'runtime_surface_missing',
    ready: false,
    summary:
      'The active route does not advertise a usable command-buffer surface, so TQ correctly keeps the lane disabled.',
    blockers,
  };
}

export function classifySvmContract(
  inputs: SvmContractInputs,
): FrontierContractVerdict<SvmContractClass> {
  const blockers: string[] = [];
  if (inputs.probeHasSvm !== true && inputs.smokeHasSvm !== true) blockers.push('svm_absent');
  if (inputs.probeHasSvmCoarse === true || inputs.smokeHasSvmCoarse === true) {
    blockers.push('coarse_only_visible');
  }
  if (inputs.probeHasSvmFine === false || inputs.smokeHasSvmFine === false) {
    blockers.push('fine_grain_missing');
  }
  if (inputs.probeHasSvmAtomics === false || inputs.smokeHasSvmAtomics === false) {
    blockers.push('system_or_atomics_missing');
  }
  if (inputs.smokeSystemRawPointerOk === false) blockers.push('raw_system_pointer_smoke_failed');
  if (inputs.smokeSystemAtomicOk === false) blockers.push('system_atomic_smoke_failed');

  if (
    inputs.probeHasSvm === true &&
    inputs.probeHasSvmCoarse === true &&
    inputs.probeHasSvmFine === true &&
    inputs.probeHasSvmAtomics === true &&
    inputs.smokeHasSvmFine === true &&
    inputs.smokeHasSvmAtomics === true &&
    inputs.smokeSystemRawPointerOk === true &&
    inputs.smokeSystemAtomicOk === true
  ) {
    return {
      classification: 'ready',
      ready: true,
      summary:
        'Fine-grain/system SVM with atomics is fully live on the active route, including raw system-pointer and atomic smoke validation.',
      blockers: [],
    };
  }

  if (
    (inputs.probeHasSvmFine === true || inputs.smokeHasSvmFine === true) &&
    (inputs.probeHasSvmAtomics !== true || inputs.smokeHasSvmAtomics !== true)
  ) {
    return {
      classification: 'fine_buffer_only',
      ready: false,
      summary:
        'The active route now exposes fine-grain buffer SVM, but still does not expose fine-grain system SVM with atomics.',
      blockers,
    };
  }

  if (inputs.probeHasSvm === true || inputs.smokeHasSvm === true) {
    return {
      classification: 'coarse_only',
      ready: false,
      summary:
        'The active route exposes VM-backed coarse-grain SVM only; fine-grain/system SVM and atomics remain below the current driver contract.',
      blockers,
    };
  }

  return {
    classification: 'no_svm',
    ready: false,
    summary: 'No SVM surface is visible on the active route.',
    blockers,
  };
}
