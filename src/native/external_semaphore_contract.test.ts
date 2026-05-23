import {
  classifyExternalSemaphoreContract,
  parseExternalSemaphoreTrace,
} from './external_semaphore_contract.js';

describe('external semaphore contract', () => {
  test('classifies missing syncobj kernel contract from trace-backed inputs', () => {
    const trace = parseExternalSemaphoreTrace(`
failed to open /dev/dri/renderD128: Permission denied
freedreno-mesa: forensic layer=fd-screen screen_init gpu=FD725 has_syncobj=0 dev_version=5
freedreno-mesa: forensic layer=fd-screen screen_callbacks fence_signal=0 fence_get_fd=1 semaphore_create=0
rusticl: external_semaphore create_semaphore rejected: no semaphore_create path fence_signal=false semaphore_create=false
    `);
    const verdict = classifyExternalSemaphoreContract({
      ...trace,
      probeHasExternalSemaphore: false,
      smokeHasExternalSemaphore: false,
      createdExportable: false,
      exportedSyncFd: false,
      renderNodeAccessible: false,
      kgslAccessible: true,
    });

    expect(trace.freedrenoHasSyncobj).toBe(false);
    expect(trace.freedrenoFenceGetFd).toBe(true);
    expect(trace.deviceVersion).toBe(5);
    expect(verdict.classification).toBe('kernel_syncobj_missing');
    expect(verdict.reusableExternalSemaphorePossible).toBe(false);
    expect(verdict.blockers).toContain('drm_cap_syncobj_missing');
  });

  test('classifies fully ready reusable external semaphores', () => {
    const verdict = classifyExternalSemaphoreContract({
      freedrenoHasSyncobj: true,
      freedrenoFenceSignal: true,
      freedrenoFenceGetFd: true,
      freedrenoSemaphoreCreate: true,
      probeHasExternalSemaphore: true,
      smokeHasExternalSemaphore: true,
      createdExportable: true,
      signaled: true,
      exportedSyncFd: true,
      importedSyncFd: true,
      waitedImported: true,
      renderNodeAccessible: true,
      kgslAccessible: true,
      deviceVersion: 10,
    });

    expect(verdict.classification).toBe('ready');
    expect(verdict.reusableExternalSemaphorePossible).toBe(true);
    expect(verdict.blockers).toEqual([]);
  });
});
