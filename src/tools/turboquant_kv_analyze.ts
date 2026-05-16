/**
 * MCP tool: turboquant_kv_analyze — KV cache compression analysis.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

export interface KVAnalyzeInput {
  /** Action: 'estimate_savings' | 'recommend_config' */
  action: 'estimate_savings' | 'recommend_config';
  /** Model head dimension */
  headDim: number;
  /** Number of KV heads */
  numKvHeads?: number;
  /** Number of layers */
  numLayers?: number;
  /** Sequence length */
  seqLen?: number;
  /** Key bits (1-6) */
  keyBits?: number;
  /** Value bits (2 or 4) */
  valueBits?: number;
  /** Buffer size (recent unquantized tokens) */
  bufferSize?: number;
  /** Target memory budget in MB */
  memoryBudgetMb?: number;
}

export interface KVAnalyzeOutput {
  action: string;
  estimate?: {
    baselineMemoryMb: number;
    compressedMemoryMb: number;
    savingsPercent: number;
    compressionRatio: number;
    keyBitsUsed: number;
    valueBitsUsed: number;
    bufferSize: number;
  };
  recommendation?: {
    keyBits: number;
    valueBits: number;
    bufferSize: number;
    estimatedMsePerCoord: number;
    maxSeqLen: number;
    memoryMb: number;
  };
}

export function turboquantKvAnalyze(input: KVAnalyzeInput): KVAnalyzeOutput {
  const headDim = input.headDim;
  const numKvHeads = input.numKvHeads ?? 8;
  const numLayers = input.numLayers ?? 32;
  const seqLen = input.seqLen ?? 4096;
  const keyBits = input.keyBits ?? 4;
  const valueBits = input.valueBits ?? 2;
  const bufferSize = input.bufferSize ?? 128;

  const result: KVAnalyzeOutput = { action: input.action };

  if (input.action === 'estimate_savings') {
    // Baseline: FP16 KV cache
    const bytesPerElement = 2; // float16
    const baselineBytes = numLayers * numKvHeads * seqLen * headDim * 2 * bytesPerElement; // *2 for K+V

    // Compressed: quantized history + FP16 buffer
    const historyLen = Math.max(0, seqLen - bufferSize);

    // Key storage: bits per coord * headDim / 8 + 4 bytes norm per token
    const keyBytesPerToken = Math.ceil(headDim * keyBits / 8) + 4;
    // Value storage: packed + scales + zeros
    const groupSize = 32;
    const nGroups = headDim / groupSize;
    const valPackedPerToken = Math.ceil(headDim * valueBits / 8);
    const valMetaPerToken = nGroups * 4 * 2; // scales + zeros as float32
    const valueBytesPerToken = valPackedPerToken + valMetaPerToken;

    const compressedHistoryBytes = numLayers * numKvHeads * historyLen * (keyBytesPerToken + valueBytesPerToken);
    const bufferBytes = numLayers * numKvHeads * bufferSize * headDim * 2 * bytesPerElement;
    const compressedBytes = compressedHistoryBytes + bufferBytes;

    const baselineMb = baselineBytes / (1024 * 1024);
    const compressedMb = compressedBytes / (1024 * 1024);

    result.estimate = {
      baselineMemoryMb: Math.round(baselineMb * 100) / 100,
      compressedMemoryMb: Math.round(compressedMb * 100) / 100,
      savingsPercent: Math.round((1 - compressedMb / baselineMb) * 10000) / 100,
      compressionRatio: Math.round(baselineMb / compressedMb * 100) / 100,
      keyBitsUsed: keyBits,
      valueBitsUsed: valueBits,
      bufferSize,
    };
  } else if (input.action === 'recommend_config') {
    const budgetMb = input.memoryBudgetMb ?? 1024;
    const budgetBytes = budgetMb * 1024 * 1024;

    // Try configurations from most aggressive to least
    const configs = [
      { kb: 2, vb: 2 as const, buf: 64 },
      { kb: 3, vb: 2 as const, buf: 128 },
      { kb: 4, vb: 2 as const, buf: 128 },
      { kb: 4, vb: 4 as const, buf: 128 },
      { kb: 4, vb: 4 as const, buf: 256 },
    ];

    // Approximate MSE per coord for different bit widths (from paper)
    const mseTable: Record<number, number> = { 1: 0.05, 2: 0.005, 3: 0.0005, 4: 0.00007, 5: 0.00001, 6: 0.000002 };

    let best = configs[configs.length - 1]!;
    let bestSeqLen = 0;

    for (const cfg of configs) {
      const groupSize = 32;
      const nGroups = headDim / groupSize;
      const keyBytesPerToken = Math.ceil(headDim * cfg.kb / 8) + 4;
      const valPackedPerToken = Math.ceil(headDim * cfg.vb / 8);
      const valMetaPerToken = nGroups * 4 * 2;
      const compressedPerToken = numLayers * numKvHeads * (keyBytesPerToken + valPackedPerToken + valMetaPerToken);
      const bufferFixed = numLayers * numKvHeads * cfg.buf * headDim * 2 * 2;

      const availableForHistory = budgetBytes - bufferFixed;
      if (availableForHistory <= 0) continue;

      const maxHistoryTokens = Math.floor(availableForHistory / compressedPerToken);
      const maxSeq = maxHistoryTokens + cfg.buf;

      if (maxSeq > bestSeqLen) {
        bestSeqLen = maxSeq;
        best = cfg;
      }
    }

    result.recommendation = {
      keyBits: best.kb,
      valueBits: best.vb,
      bufferSize: best.buf,
      estimatedMsePerCoord: mseTable[best.kb] ?? 0.00007,
      maxSeqLen: bestSeqLen,
      memoryMb: budgetMb,
    };
  }

  return result;
}

export const turboquantKvAnalyzeSchema = {
  name: 'turboquant_kv_analyze',
  description: 'Analyze KV cache compression savings and recommend TurboQuant configuration for a given model architecture and memory budget.',
  inputSchema: {
    type: 'object' as const,
    properties: {
      action: {
        type: 'string' as const,
        enum: ['estimate_savings', 'recommend_config'],
        description: 'Action to perform',
      },
      headDim: { type: 'number' as const, description: 'Model head dimension' },
      numKvHeads: { type: 'number' as const, description: 'Number of KV heads (default 8)' },
      numLayers: { type: 'number' as const, description: 'Number of layers (default 32)' },
      seqLen: { type: 'number' as const, description: 'Sequence length (default 4096)' },
      keyBits: { type: 'number' as const, description: 'Key bits 1-6 (default 4)' },
      valueBits: { type: 'number' as const, description: 'Value bits 2 or 4 (default 2)' },
      bufferSize: { type: 'number' as const, description: 'Recent unquantized tokens (default 128)' },
      memoryBudgetMb: { type: 'number' as const, description: 'Target memory budget in MB (for recommend_config)' },
    },
    required: ['action', 'headDim'],
  },
};
