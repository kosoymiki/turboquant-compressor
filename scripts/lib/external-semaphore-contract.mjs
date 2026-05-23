export function parseExternalSemaphoreTrace(text) {
  const source = `${text || ''}`;
  const parseBool01 = (key) => {
    const match = source.match(new RegExp(`${key}=(0|1)`));
    if (!match) return null;
    return match[1] === '1';
  };
  const deviceVersionMatch = source.match(/dev_version=(\d+)/);
  return {
    freedrenoHasSyncobj: parseBool01('has_syncobj'),
    freedrenoFenceSignal: parseBool01('fence_signal'),
    freedrenoFenceGetFd: parseBool01('fence_get_fd'),
    freedrenoSemaphoreCreate: parseBool01('semaphore_create'),
    renderNodePermissionDenied:
      source.includes('/dev/dri/renderD128: Permission denied') ||
      source.includes('failed to open /dev/dri/renderD128: Permission denied'),
    semaphoreRejectedNoPath: source.includes(
      'create_semaphore rejected: no semaphore_create path',
    ),
    deviceVersion: deviceVersionMatch ? Number(deviceVersionMatch[1]) : null,
  };
}

export function classifyExternalSemaphoreContract(inputs) {
  const blockers = [];
  const hasSyncobj = inputs.freedrenoHasSyncobj === true;
  const hasFenceSignal = inputs.freedrenoFenceSignal === true;
  const hasSemaphoreCreate = inputs.freedrenoSemaphoreCreate === true;
  const hasFenceGetFd = inputs.freedrenoFenceGetFd === true;
  const exportWorks =
    inputs.createdExportable === true &&
    inputs.signaled === true &&
    inputs.exportedSyncFd === true &&
    inputs.importedSyncFd === true &&
    inputs.waitedImported === true;

  if (inputs.renderNodeAccessible === false) {
    blockers.push('render_node_access_blocked');
  }
  if (inputs.kgslAccessible === true && inputs.renderNodeAccessible === false) {
    blockers.push('kgsl_access_without_render_access');
  }
  if (inputs.freedrenoHasSyncobj === false) {
    blockers.push('drm_cap_syncobj_missing');
  }
  if (inputs.freedrenoFenceSignal === false) {
    blockers.push('fence_signal_missing');
  }
  if (inputs.freedrenoSemaphoreCreate === false) {
    blockers.push('semaphore_create_missing');
  }
  if (inputs.probeHasExternalSemaphore === false) {
    blockers.push('extension_not_advertised');
  }
  if (inputs.smokeHasExternalSemaphore === false) {
    blockers.push('tq_runtime_surface_disabled');
  }
  if (!exportWorks) {
    blockers.push('live_export_not_working');
  }

  if (
    hasSyncobj &&
    hasFenceSignal &&
    hasSemaphoreCreate &&
    hasFenceGetFd &&
    inputs.probeHasExternalSemaphore === true &&
    inputs.smokeHasExternalSemaphore === true &&
    exportWorks
  ) {
    return {
      classification: 'ready',
      reusableExternalSemaphorePossible: true,
      summary: 'Syncobj-backed reusable external semaphores are fully available on the active route.',
      blockers: [],
    };
  }

  if (!hasSyncobj && inputs.kgslAccessible === true) {
    return {
      classification: 'kernel_syncobj_missing',
      reusableExternalSemaphorePossible: false,
      summary:
        'The active route can reach KGSL/MSM devices, but DRM syncobj is not exposed, so reusable OpenCL external semaphores cannot be honestly surfaced.',
      blockers,
    };
  }

  if (inputs.renderNodeAccessible === false && inputs.kgslAccessible === true) {
    return {
      classification: 'render_node_access_blocked',
      reusableExternalSemaphorePossible: false,
      summary:
        'Userspace can reach KGSL, but the render node is blocked on the active route, so the DRM-side reusable semaphore substrate is unavailable.',
      blockers,
    };
  }

  if (inputs.renderNodeAccessible === false && inputs.kgslAccessible !== true) {
    return {
      classification: 'userspace_route_missing',
      reusableExternalSemaphorePossible: false,
      summary:
        'The active route does not have sufficient device-node access to prove or expose reusable external semaphores.',
      blockers,
    };
  }

  if (!hasSyncobj && inputs.deviceVersion != null && inputs.deviceVersion <= 5) {
    return {
      classification: 'kgsl_only_route',
      reusableExternalSemaphorePossible: false,
      summary:
        'The active device contract remains below the syncobj era exposed by Freedreno DRM userspace; reusable external semaphores are blocked by the lower kernel/uAPI route.',
      blockers,
    };
  }

  return {
    classification: 'unknown',
    reusableExternalSemaphorePossible: false,
    summary:
      'External semaphore closure is still blocked, but the remaining gate does not fit a single classified kernel-contract bucket.',
    blockers,
  };
}
