import { chooseFrontierCampaignPlan } from './frontier_campaign_contract.js';

describe('frontier campaign contract', () => {
  test('prioritizes generic pointer before render-node policy', () => {
    const plan = chooseFrontierCampaignPlan({
      spirvCapabilityClassification: 'warning_changes_runtime_outcome',
      renderNodePolicyClassification: 'render_node_policy_gate',
      externalSemaphoreClassification: 'kernel_syncobj_missing',
      initializeMemoryClassification: 'runtime_surface_missing',
      commandBufferClassification: 'runtime_surface_missing',
      svmClassification: 'coarse_only',
    });

    expect(plan.primary).toBe('mesa_generic_pointer');
    expect(plan.secondary).toBe('lower_render_node_drm_policy');
  });

  test('falls back to truth-locked release when no blocker remains', () => {
    const plan = chooseFrontierCampaignPlan({
      spirvCapabilityClassification: 'clean',
      renderNodePolicyClassification: 'ready',
      externalSemaphoreClassification: 'ready',
      initializeMemoryClassification: 'ready',
      commandBufferClassification: 'ready',
      svmClassification: 'ready',
    });

    expect(plan.primary).toBe('truth_locked_release');
  });

  test('prioritizes mesa_system_svm after fine-buffer-only progress', () => {
    const plan = chooseFrontierCampaignPlan({
      spirvCapabilityClassification: 'clean',
      renderNodePolicyClassification: 'kgsl_route_operational',
      externalSemaphoreClassification: 'ready',
      initializeMemoryClassification: 'ready',
      commandBufferClassification: 'ready',
      svmClassification: 'fine_buffer_only',
    });

    expect(plan.primary).toBe('mesa_system_svm');
    expect(plan.secondary).toBe(null);
  });

  test('treats kgsl-operational render policy as non-primary and moves to semaphore substrate', () => {
    const plan = chooseFrontierCampaignPlan({
      spirvCapabilityClassification: 'clean',
      renderNodePolicyClassification: 'kgsl_route_operational',
      externalSemaphoreClassification: 'kernel_syncobj_missing',
      initializeMemoryClassification: 'runtime_surface_missing',
      commandBufferClassification: 'runtime_surface_missing',
      svmClassification: 'coarse_only',
    });

    expect(plan.primary).toBe('mesa_external_semaphore_syncobj');
    expect(plan.secondary).toBe(null);
  });
});
