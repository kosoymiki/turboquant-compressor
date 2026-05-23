import { computeLloydMaxBetaCodebook } from './beta_sphere.js';
import { UniformSymmetricCodebook } from './quantizer.js';
import { RotationEngine, FWHT_SIGN } from './rotation.js';
import { dequantizeValues, quantizeValues } from './value_quant.js';

export interface CalibrationObserverStats {
  min: number;
  max: number;
  absMax: number;
  mean: number;
  variance: number;
  p01: number;
  p99: number;
}

export interface CalibrationFakeQuantSummary {
  policy: 'affine_per_tensor' | 'symmetric_per_tensor' | 'affine_per_group';
  bits: number;
  mse: number;
  maxAbsError: number;
  cosineSimilarity: number;
  clippedFraction: number;
}

export interface CalibrationStepSizeSummary {
  policy: 'lsq_compatible_symmetric' | 'affine_minmax' | 'group_minmax';
  learnable: boolean;
  initialStepSize: number;
  zeroPoint: number;
  qmin: number;
  qmax: number;
  gradientScale: number;
  groupCount?: number;
  minGroupStep?: number;
  maxGroupStep?: number;
}

export interface CalibrationSurfaceSummary {
  name: string;
  role: 'activation' | 'weight' | 'kv_value';
  sampleCount: number;
  elementCount: number;
  observer: CalibrationObserverStats;
  fakeQuant: CalibrationFakeQuantSummary;
  stepSize: CalibrationStepSizeSummary;
}

export interface PrecisionCalibrationRuntimeState {
  timestamp: string;
  module: 'precision_calibration_runtime';
  calibrationProfile: 'bounded_synthetic_runtime_contract';
  observerReady: boolean;
  fakeQuantPolicyReady: boolean;
  stepSizeContractReady: boolean;
  calibrationArtifactReady: boolean;
  surfaceCount: number;
  observerStrategy: string;
  fakeQuantNoisePolicy: string;
  surfaces: CalibrationSurfaceSummary[];
  activationStepSizeMean: number;
  weightStepSizeMean: number;
  kvGroupStepSizeMean: number;
  betaCodebookRange: {
    dimension: number;
    bits: number;
    centroidMin: number;
    centroidMax: number;
    msePerCoord: number;
  };
  anchors: string[];
  residualFrontier: string[];
}

function makeDeterministicVector(length: number, phase: number, amplitude: number): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const x = i + 1;
    out[i] =
      amplitude * (
        Math.sin(x * (phase + 0.173)) * 0.62 +
        Math.cos(x * (phase + 0.097)) * 0.28 +
        Math.sin(x * 0.11 + phase * 0.5) * 0.1
      );
  }
  return out;
}

function concatVectors(vectors: Float32Array[]): Float32Array {
  const total = vectors.reduce((sum, vector) => sum + vector.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const vector of vectors) {
    out.set(vector, offset);
    offset += vector.length;
  }
  return out;
}

function normalize(vector: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < vector.length; i++) {
    norm += vector[i]! * vector[i]!;
  }
  const scale = norm > 1e-12 ? 1 / Math.sqrt(norm) : 0;
  const out = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i++) {
    out[i] = vector[i]! * scale;
  }
  return out;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.round((sorted.length - 1) * p)));
  return sorted[idx]!;
}

function summarizeObserver(values: Float32Array): CalibrationObserverStats {
  let min = Infinity;
  let max = -Infinity;
  let mean = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    min = Math.min(min, v);
    max = Math.max(max, v);
    mean += v;
  }
  mean /= values.length || 1;

  let variance = 0;
  for (let i = 0; i < values.length; i++) {
    const delta = values[i]! - mean;
    variance += delta * delta;
  }
  variance /= values.length || 1;

  const sorted = Array.from(values).sort((a, b) => a - b);
  return {
    min,
    max,
    absMax: Math.max(Math.abs(min), Math.abs(max)),
    mean,
    variance,
    p01: percentile(sorted, 0.01),
    p99: percentile(sorted, 0.99),
  };
}

function computeErrorMetrics(reference: Float32Array, reconstructed: Float32Array) {
  let mse = 0;
  let maxAbsError = 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < reference.length; i++) {
    const a = reference[i]!;
    const b = reconstructed[i]!;
    const err = a - b;
    mse += err * err;
    maxAbsError = Math.max(maxAbsError, Math.abs(err));
    dot += a * b;
    normA += a * a;
    normB += b * b;
  }
  mse /= reference.length || 1;
  const cosineSimilarity = dot / ((Math.sqrt(normA) * Math.sqrt(normB)) || 1);
  return { mse, maxAbsError, cosineSimilarity };
}

function fakeQuantizeSymmetric(values: Float32Array, bits: 2 | 3 | 4 | 8) {
  const qmax = (1 << bits) / 2 - 1;
  const observer = summarizeObserver(values);
  const scale = Math.max(observer.absMax / Math.max(qmax, 1), 1e-10);
  const normalized = new Float32Array(values.length);
  let clipped = 0;
  for (let i = 0; i < values.length; i++) {
    const scaled = values[i]! / scale;
    const normalizedValue = scaled / Math.max(qmax, 1);
    if (normalizedValue > 1 || normalizedValue < -1) clipped++;
    normalized[i] = Math.max(-1, Math.min(1, normalizedValue));
  }

  const codebook = new UniformSymmetricCodebook(bits);
  const { indices } = codebook.quantize(normalized);
  const dequantizedNormalized = codebook.dequantize(indices, 1.0, values.length);
  const reconstructed = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    reconstructed[i] = dequantizedNormalized[i]! * scale * Math.max(qmax, 1);
  }

  return {
    reconstructed,
    clippedFraction: clipped / Math.max(values.length, 1),
    stepSize: {
      policy: 'lsq_compatible_symmetric' as const,
      learnable: true,
      initialStepSize: scale,
      zeroPoint: 0,
      qmin: -qmax,
      qmax,
      gradientScale: 1 / Math.sqrt(values.length * Math.max(qmax, 1)),
    },
  };
}

function fakeQuantizeAffine(values: Float32Array, bits: 2 | 3 | 4 | 8) {
  const observer = summarizeObserver(values);
  const qmin = 0;
  const qmax = (1 << bits) - 1;
  const scale = Math.max((observer.max - observer.min) / Math.max(qmax - qmin, 1), 1e-10);
  const zeroPoint = Math.max(qmin, Math.min(qmax, Math.round(qmin - observer.min / scale)));
  const reconstructed = new Float32Array(values.length);
  let clipped = 0;
  for (let i = 0; i < values.length; i++) {
    const q = Math.round(values[i]! / scale + zeroPoint);
    const qClamped = Math.max(qmin, Math.min(qmax, q));
    if (q !== qClamped) clipped++;
    reconstructed[i] = (qClamped - zeroPoint) * scale;
  }
  return {
    reconstructed,
    clippedFraction: clipped / Math.max(values.length, 1),
    stepSize: {
      policy: 'affine_minmax' as const,
      learnable: false,
      initialStepSize: scale,
      zeroPoint,
      qmin,
      qmax,
      gradientScale: 0,
    },
  };
}

function summarizeActivationSurface(): CalibrationSurfaceSummary {
  const dimension = 64;
  const rotation = new RotationEngine(dimension, 7, FWHT_SIGN);
  const batches: Float32Array[] = [];
  for (let i = 0; i < 6; i++) {
    const normalized = normalize(makeDeterministicVector(dimension, 0.17 + i * 0.09, 1.0));
    batches.push(rotation.rotate(normalized));
  }
  const values = concatVectors(batches);
  const observer = summarizeObserver(values);
  const quantized = fakeQuantizeAffine(values, 4);
  const error = computeErrorMetrics(values, quantized.reconstructed);
  return {
    name: 'rotated_activation_surface',
    role: 'activation',
    sampleCount: batches.length,
    elementCount: values.length,
    observer,
    fakeQuant: {
      policy: 'affine_per_tensor',
      bits: 4,
      mse: error.mse,
      maxAbsError: error.maxAbsError,
      cosineSimilarity: error.cosineSimilarity,
      clippedFraction: quantized.clippedFraction,
    },
    stepSize: quantized.stepSize,
  };
}

function summarizeWeightSurface(): CalibrationSurfaceSummary {
  const blocks: Float32Array[] = [];
  for (let i = 0; i < 4; i++) {
    blocks.push(makeDeterministicVector(64, 0.41 + i * 0.13, 0.85));
  }
  const values = concatVectors(blocks);
  const observer = summarizeObserver(values);
  const quantized = fakeQuantizeSymmetric(values, 4);
  const error = computeErrorMetrics(values, quantized.reconstructed);
  return {
    name: 'weight_block_surface',
    role: 'weight',
    sampleCount: blocks.length,
    elementCount: values.length,
    observer,
    fakeQuant: {
      policy: 'symmetric_per_tensor',
      bits: 4,
      mse: error.mse,
      maxAbsError: error.maxAbsError,
      cosineSimilarity: error.cosineSimilarity,
      clippedFraction: quantized.clippedFraction,
    },
    stepSize: quantized.stepSize,
  };
}

function summarizeKvValueSurface(): CalibrationSurfaceSummary {
  const numTokens = 4;
  const headDim = 32;
  const values = new Float32Array(numTokens * headDim);
  for (let token = 0; token < numTokens; token++) {
    const vector = makeDeterministicVector(headDim, 0.23 + token * 0.19, 0.95);
    values.set(vector, token * headDim);
  }

  const quantized = quantizeValues(values, numTokens, headDim, 4, 8);
  const reconstructed = dequantizeValues(quantized);
  const observer = summarizeObserver(values);
  const error = computeErrorMetrics(values, reconstructed);

  let stepSum = 0;
  let stepMin = Infinity;
  let stepMax = -Infinity;
  for (let i = 0; i < quantized.scales.length; i++) {
    const scale = quantized.scales[i]!;
    stepSum += scale;
    stepMin = Math.min(stepMin, scale);
    stepMax = Math.max(stepMax, scale);
  }

  return {
    name: 'kv_value_group_surface',
    role: 'kv_value',
    sampleCount: numTokens,
    elementCount: values.length,
    observer,
    fakeQuant: {
      policy: 'affine_per_group',
      bits: 4,
      mse: error.mse,
      maxAbsError: error.maxAbsError,
      cosineSimilarity: error.cosineSimilarity,
      clippedFraction: 0,
    },
    stepSize: {
      policy: 'group_minmax',
      learnable: false,
      initialStepSize: stepSum / Math.max(quantized.scales.length, 1),
      zeroPoint: 0,
      qmin: 0,
      qmax: (1 << quantized.bits) - 1,
      gradientScale: 0,
      groupCount: quantized.scales.length,
      minGroupStep: stepMin,
      maxGroupStep: stepMax,
    },
  };
}

export function buildPrecisionCalibrationRuntimeState(): PrecisionCalibrationRuntimeState {
  const activationSurface = summarizeActivationSurface();
  const weightSurface = summarizeWeightSurface();
  const kvSurface = summarizeKvValueSurface();
  const surfaces = [activationSurface, weightSurface, kvSurface];
  const betaCodebook = computeLloydMaxBetaCodebook(64, 4);

  const observerReady = surfaces.every((surface) =>
    Number.isFinite(surface.observer.min) &&
    Number.isFinite(surface.observer.max) &&
    surface.elementCount > 0,
  );
  const fakeQuantPolicyReady = surfaces.every((surface) =>
    Number.isFinite(surface.fakeQuant.mse) &&
    surface.fakeQuant.cosineSimilarity > 0.9,
  );
  const stepSizeContractReady = surfaces.every((surface) =>
    Number.isFinite(surface.stepSize.initialStepSize) &&
    surface.stepSize.initialStepSize > 0,
  );

  return {
    timestamp: new Date().toISOString(),
    module: 'precision_calibration_runtime',
    calibrationProfile: 'bounded_synthetic_runtime_contract',
    observerReady,
    fakeQuantPolicyReady,
    stepSizeContractReady,
    calibrationArtifactReady: observerReady && fakeQuantPolicyReady && stepSizeContractReady,
    surfaceCount: surfaces.length,
    observerStrategy: 'minmax_with_percentile_snapshots',
    fakeQuantNoisePolicy: 'activation_affine_weight_symmetric_kv_group_affine',
    surfaces,
    activationStepSizeMean: activationSurface.stepSize.initialStepSize,
    weightStepSizeMean: weightSurface.stepSize.initialStepSize,
    kvGroupStepSizeMean: kvSurface.stepSize.initialStepSize,
    betaCodebookRange: {
      dimension: 64,
      bits: 4,
      centroidMin: Math.min(...betaCodebook.centroids),
      centroidMax: Math.max(...betaCodebook.centroids),
      msePerCoord: betaCodebook.msePerCoord,
    },
    anchors: [
      'src/tools/turboquant_quantize.ts',
      'src/core/quantizer.ts',
      'src/core/value_quant.ts',
      'src/core/codebook.ts',
    ],
    residualFrontier: [
      'hybrid_cpu_gpu_training_runtime',
      'full_qat_training_loop',
      'dataset_backed_distillation',
      'on_device_distillation',
    ],
  };
}
