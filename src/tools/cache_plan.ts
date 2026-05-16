import { z } from 'zod';
import type { CachePlan } from '../cost/types.js';
import { buildCachePlan } from '../cost/recommendations.js';

const CachePlanInputSchema = z.object({
  blocks: z.array(
    z.object({
      label: z.string(),
      text: z.string(),
      volatilityHint: z.string().optional(),
    })
  ),
  target: z.string().default('claude_code_context_hygiene'),
});

export function turboquantCachePlan(rawInput: unknown): CachePlan {
  const input = CachePlanInputSchema.parse(rawInput);
  return buildCachePlan(input.blocks, input.target);
}