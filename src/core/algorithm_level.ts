/**
 * TurboQuant v4.5.0 — Production Algorithm Levels
 *
 * LEVEL_1_PUBLIC: TurboQuant Beta Lloyd-Max scalar quantization + QJL residual correction
 *   - Beta-distributed coordinate PDF after random Hadamard rotation
 *   - Lloyd-Max optimal codebook trained on Beta(d/2, d/2) distribution
 *   - QJL residual sketch for unbiased dot-product estimation (Zandieh et al., 2024)
 *   - Paper-faithful implementation: asymmetric QJL (sign-bit on keys), structured JL via FWHT
 *
 * References:
 *   TurboQuant theory: arXiv:2504.19874
 *   QJL: Zandieh et al., "Query-Aware JPEG-Like Compression", ICML 2024
 */

export const PUBLIC_ALGORITHM_LEVEL = 'LEVEL_1_PUBLIC';
export const EXPERIMENTAL_QJL_LEVEL = 'LEVEL_1_EXPERIMENTAL_QJL';

/**
 * Algorithm level descriptions for MCP tool responses and documentation.
 */
export const ALGORITHM_DESCRIPTIONS: Record<string, {
  name: string;
  description: string;
  components: string[];
  paperReference?: string;
}> = {
  'LEVEL_1_PUBLIC': {
    name: 'TurboQuant Beta Lloyd-Max + QJL',
    description: 'Production TurboQuant: Beta PDF coordinate model, Lloyd-Max optimal quantization, QJL residual correction',
    components: [
      'Random Hadamard rotation (FWHT-based sign flip)',
      'Beta(d/2, d/2) coordinate PDF after rotation',
      'Lloyd-Max codebook optimized for Beta distribution',
      'QJL residual sketch for unbiased dot-product estimation',
      'FWHT-based structured JL projection O(d log d)'
    ],
    paperReference: 'arXiv:2504.19874'
  }
};

export function algorithmLevelDescription(level: string = PUBLIC_ALGORITHM_LEVEL): string {
  const desc = ALGORITHM_DESCRIPTIONS[level];
  if (!desc) return level;
  return `${desc.name}: ${desc.description}`;
}
