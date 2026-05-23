import {
  classifyCommandBufferContract,
  classifyInitializeMemoryContract,
  classifySvmContract,
} from './frontier_contracts.js';

describe('frontier contracts', () => {
  test('classifies initialize_memory runtime surface absence', () => {
    const verdict = classifyInitializeMemoryContract({
      probeHasInitializeMemory: false,
      smokeHasInitializeMemory: false,
      initContext: true,
    });

    expect(verdict.classification).toBe('runtime_surface_missing');
    expect(verdict.ready).toBe(false);
    expect(verdict.blockers).toContain('extension_not_advertised');
  });

  test('classifies command_buffer runtime surface absence', () => {
    const verdict = classifyCommandBufferContract({
      probeHasCommandBuffer: false,
      smokeHasCommandBuffer: false,
      commandBufferCapabilities: 0,
      probeHasMutableDispatch: false,
    });

    expect(verdict.classification).toBe('runtime_surface_missing');
    expect(verdict.ready).toBe(false);
    expect(verdict.blockers).toContain('extension_not_advertised');
  });

  test('classifies base command_buffer route as ready even with zero optional capability bits', () => {
    const verdict = classifyCommandBufferContract({
      probeHasCommandBuffer: true,
      smokeHasCommandBuffer: true,
      commandBufferCapabilities: 0,
      probeHasMutableDispatch: false,
    });

    expect(verdict.classification).toBe('ready');
    expect(verdict.ready).toBe(true);
  });

  test('classifies fine-buffer-only svm contract', () => {
    const verdict = classifySvmContract({
      probeHasSvm: true,
      probeHasSvmCoarse: true,
      probeHasSvmFine: true,
      probeHasSvmAtomics: false,
      smokeHasSvm: true,
      smokeHasSvmCoarse: true,
      smokeHasSvmFine: true,
      smokeHasSvmAtomics: false,
    });

    expect(verdict.classification).toBe('fine_buffer_only');
    expect(verdict.ready).toBe(false);
    expect(verdict.blockers).toContain('system_or_atomics_missing');
  });

  test('classifies full system svm with atomics only when live smoke passes', () => {
    const verdict = classifySvmContract({
      probeHasSvm: true,
      probeHasSvmCoarse: true,
      probeHasSvmFine: true,
      probeHasSvmAtomics: true,
      smokeHasSvm: true,
      smokeHasSvmCoarse: true,
      smokeHasSvmFine: true,
      smokeHasSvmAtomics: true,
      smokeSystemRawPointerOk: true,
      smokeSystemAtomicOk: true,
    });

    expect(verdict.classification).toBe('ready');
    expect(verdict.ready).toBe(true);
  });
});
