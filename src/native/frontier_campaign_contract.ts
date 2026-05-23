export type FrontierCampaignKind =
  | 'mesa_generic_pointer'
  | 'lower_render_node_drm_policy'
  | 'mesa_external_semaphore_syncobj'
  | 'mesa_initialize_memory'
  | 'mesa_command_buffer'
  | 'mesa_system_svm'
  | 'truth_locked_release';

export interface FrontierCampaignInputs {
  spirvCapabilityClassification?: string | null;
  renderNodePolicyClassification?: string | null;
  externalSemaphoreClassification?: string | null;
  initializeMemoryClassification?: string | null;
  commandBufferClassification?: string | null;
  svmClassification?: string | null;
  routeSelectionClassification?: string | null;
  stagedDependencyChainClassification?: string | null;
}

export interface FrontierCampaignPlan {
  primary: FrontierCampaignKind;
  secondary: FrontierCampaignKind | null;
  rationale: string;
  blockers: string[];
}

export function chooseFrontierCampaignPlan(
  inputs: FrontierCampaignInputs,
): FrontierCampaignPlan {
  const blockers: string[] = [];
  const renderRouteReady =
    inputs.renderNodePolicyClassification === 'ready' ||
    inputs.renderNodePolicyClassification === 'kgsl_route_operational';
  if (inputs.spirvCapabilityClassification && inputs.spirvCapabilityClassification !== 'clean') {
    blockers.push(`spirv:${inputs.spirvCapabilityClassification}`);
  }
  if (inputs.renderNodePolicyClassification && !renderRouteReady) {
    blockers.push(`render:${inputs.renderNodePolicyClassification}`);
  }
  if (inputs.externalSemaphoreClassification && inputs.externalSemaphoreClassification !== 'ready') {
    blockers.push(`semaphore:${inputs.externalSemaphoreClassification}`);
  }
  if (inputs.initializeMemoryClassification && inputs.initializeMemoryClassification !== 'ready') {
    blockers.push(`init_mem:${inputs.initializeMemoryClassification}`);
  }
  if (inputs.commandBufferClassification && inputs.commandBufferClassification !== 'ready') {
    blockers.push(`cmd_buf:${inputs.commandBufferClassification}`);
  }
  if (inputs.svmClassification && inputs.svmClassification !== 'ready') {
    blockers.push(`svm:${inputs.svmClassification}`);
  }

  if (inputs.spirvCapabilityClassification === 'warning_changes_runtime_outcome') {
    return {
      primary: 'mesa_generic_pointer',
      secondary:
        inputs.renderNodePolicyClassification === 'render_node_policy_gate'
          ? 'lower_render_node_drm_policy'
          : null,
      rationale:
        'GenericPointer already changes live probe behavior on the active route, so it is the highest-yield userspace campaign before dropping below Mesa/TQ.',
      blockers,
    };
  }

  if (inputs.renderNodePolicyClassification === 'render_node_policy_gate') {
    return {
      primary: 'lower_render_node_drm_policy',
      secondary:
        inputs.externalSemaphoreClassification === 'kernel_syncobj_missing'
          ? 'mesa_external_semaphore_syncobj'
          : null,
      rationale:
        'The remaining blocker is below loader closure and sits on render-node policy, so the next honest gain is a lower DRM/KGSL route campaign.',
      blockers,
    };
  }

  if (inputs.externalSemaphoreClassification === 'kernel_syncobj_missing') {
    return {
      primary: 'mesa_external_semaphore_syncobj',
      secondary: null,
      rationale:
        'The semaphore lane is source-complete in Rusticl/TQ and now isolated to the lower syncobj substrate.',
      blockers,
    };
  }

  if (inputs.initializeMemoryClassification === 'runtime_surface_missing') {
    return {
      primary: 'mesa_initialize_memory',
      secondary:
        inputs.commandBufferClassification === 'runtime_surface_missing'
          ? 'mesa_command_buffer'
          : null,
      rationale:
        'Initialize-memory remains a pure runtime-surface gap and can be pursued without crossing into lower kernel policy first.',
      blockers,
    };
  }

  if (inputs.commandBufferClassification === 'runtime_surface_missing') {
    return {
      primary: 'mesa_command_buffer',
      secondary:
        inputs.svmClassification === 'coarse_only'
          ? 'mesa_system_svm'
          : null,
      rationale:
        'Command-buffer remains absent as a Rusticl runtime surface and should be treated as its own implementation campaign.',
      blockers,
    };
  }

  if (inputs.svmClassification === 'coarse_only' || inputs.svmClassification === 'fine_buffer_only') {
    return {
      primary: 'mesa_system_svm',
      secondary: null,
      rationale:
        inputs.svmClassification === 'fine_buffer_only'
          ? 'Fine-grain buffer SVM is now exposed, so the remaining gain is the deeper Mesa/driver capability campaign for full system SVM with atomics.'
          : 'Only coarse-grain SVM is exposed, so the remaining gain is a deeper Mesa/driver capability campaign for fine/system SVM.',
      blockers,
    };
  }

  return {
    primary: 'truth_locked_release',
    secondary: null,
    rationale:
      'No unresolved classified frontier tail currently dominates the next campaign order, so the stack is ready for a truth-locked release posture.',
    blockers,
  };
}
