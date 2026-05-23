export const PUBLIC_ALGORITHM_LEVEL = 'LEVEL_1_PUBLIC';
export const EXPERIMENTAL_QJL_LEVEL = 'LEVEL_1_EXPERIMENTAL_QJL';

export function publicLevelWarning(codebookType: 'uniform' | 'turboquant_beta'): string {
  const quantizer = codebookType === 'turboquant_beta'
    ? 'TurboQuant Beta Lloyd-Max scalar quantization'
    : 'fixed uniform quantization';
  return `${PUBLIC_ALGORITHM_LEVEL}: Public path uses ${quantizer} without QJL correction.`;
}
