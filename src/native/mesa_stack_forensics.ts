export type MesaForensicLayer =
  | 'tq_loader'
  | 'tq_probe'
  | 'rusticl_platform'
  | 'rusticl_device'
  | 'rusticl_context'
  | 'rusticl_memory'
  | 'rusticl_semaphore'
  | 'pipe_screen'
  | 'pipe_resource'
  | 'pipe_fence'
  | 'pipe_context'
  | 'freedreno_import'
  | 'freedreno_rd'
  | 'turnip_vk'
  | 'android_loader';

export interface MesaForensicProfile {
  env: Record<string, string>;
  layers: MesaForensicLayer[];
  closureTargets: string[];
  commands: string[];
}

export function createMesaForensicProfile(driverRoot: string): MesaForensicProfile {
  return {
    env: {
      TQ_OPENCL_TRACE: '1',
      TQ_DRIVER_TRACE: '1',
      TQ_MESA_FORENSICS: '1',
      RUSTICL_DEBUG: 'memory,forensics',
      FD_MESA_DEBUG: 'flush',
      FD_RD_DUMP: 'enable,full',
      FD_RD_DUMP_PATH: '/data/local/tmp',
      TU_DEBUG: 'flushall,rd',
      TU_PERFETTO: '1',
      OCL_ICD_VENDORS: `${driverRoot}/layer1-compute`,
      VK_ICD_FILENAMES: `${driverRoot}/layer2-vulkan/freedreno_icd.aarch64.json`,
      VK_DRIVER_FILES: `${driverRoot}/layer2-vulkan/freedreno_icd.aarch64.json`,
    },
    layers: [
      'tq_loader',
      'tq_probe',
      'rusticl_platform',
      'rusticl_device',
      'rusticl_context',
      'rusticl_memory',
      'rusticl_semaphore',
      'pipe_screen',
      'pipe_resource',
      'pipe_fence',
      'pipe_context',
      'freedreno_import',
      'freedreno_rd',
      'turnip_vk',
      'android_loader',
    ],
    closureTargets: [
      'initialize_memory',
      'external_memory',
      'external_semaphore',
      'system_svm',
      'command_buffer',
      'loader_namespace',
      'staged_dependency_chain',
      'turnip_submission_trace',
      'freedreno_rd_dump',
    ],
    commands: [
      'probe',
      'frontier-smoke',
      'adb shell logcat -d | grep -i -E "rusticl|OpenCL|libRusticl|linker|libLLVM|libxml2|libicu"',
      'adb shell /system/bin/linker64 --list <target>',
    ],
  };
}
