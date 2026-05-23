import { chooseP1P3CampaignPlan } from './p1_p3_campaign_contract.js';

describe('chooseP1P3CampaignPlan', () => {
  it('orders unresolved clusters by fixed engineering priority', () => {
    const plan = chooseP1P3CampaignPlan([
      { id: 'p3_distributed_privacy', unresolvedCount: 10, specOnlyCount: 0, partialCount: 0, donorCount: 8 },
      { id: 'p2_precision_training_runtime', unresolvedCount: 4, specOnlyCount: 4, partialCount: 0, donorCount: 5 },
      { id: 'p2_paged_kv_prefix_cache', unresolvedCount: 3, specOnlyCount: 3, partialCount: 0, donorCount: 5 },
      { id: 'p1_runtime_memory', unresolvedCount: 8, specOnlyCount: 3, partialCount: 5, donorCount: 10 },
      { id: 'p1_kernel_primitives', unresolvedCount: 12, specOnlyCount: 4, partialCount: 8, donorCount: 12 },
    ]);

    expect(plan.queue.map((entry) => entry.id)).toEqual([
      'p1_kernel_primitives',
      'p1_runtime_memory',
      'p2_paged_kv_prefix_cache',
      'p2_precision_training_runtime',
      'p3_distributed_privacy',
    ]);
  });

  it('returns an empty queue when no unresolved cluster remains', () => {
    const plan = chooseP1P3CampaignPlan([
      { id: 'p1_kernel_primitives', unresolvedCount: 0, specOnlyCount: 0, partialCount: 0, donorCount: 12 },
    ]);

    expect(plan.queue).toEqual([]);
  });
});
