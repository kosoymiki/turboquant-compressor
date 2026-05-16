import { z } from 'zod';
import { lintCacheStability } from '../cost/context_lint.js';
const PromptCacheLintInputSchema = z.object({
    blocks: z.array(z.object({
        label: z.string(),
        text: z.string(),
    })),
});
export function turboquantPromptCacheLint(rawInput) {
    const input = PromptCacheLintInputSchema.parse(rawInput);
    return lintCacheStability(input.blocks);
}
