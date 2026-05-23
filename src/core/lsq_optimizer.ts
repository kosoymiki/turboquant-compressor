function makeDeterministicWeights(length: number, phase: number, amplitude: number): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const x = i + 1;
    out[i] =
      amplitude * (
        Math.sin(x * (phase + 0.131)) * 0.57 +
        Math.cos(x * (phase + 0.071)) * 0.29 +
        Math.sin(x * 0.041 + phase) * 0.14
      );
  }
  return out;
}

function meanSquaredError(a: Float32Array, b: Float32Array): number {
  let mse = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i]! - b[i]!;
    mse += d * d;
  }
  return mse / Math.max(a.length, 1);
}

function quantizeIndex(raw: number, qmin: number, qmax: number): number {
  return Math.max(qmin, Math.min(qmax, Math.round(raw)));
}

function fakeQuantizeSymmetric(values: Float32Array, stepSize: number, qmin: number, qmax: number): Float32Array {
  const out = new Float32Array(values.length);
  const safeStep = Math.max(stepSize, 1e-10);
  for (let i = 0; i < values.length; i++) {
    const q = quantizeIndex(values[i]! / safeStep, qmin, qmax);
    out[i] = q * safeStep;
  }
  return out;
}

function surrogateQuantizeSymmetric(values: Float32Array, stepSize: number, qmin: number, qmax: number): Float32Array {
  const out = new Float32Array(values.length);
  const safeStep = Math.max(stepSize, 1e-10);
  for (let i = 0; i < values.length; i++) {
    const scaled = values[i]! / safeStep;
    const clamped = Math.max(qmin, Math.min(qmax, scaled));
    out[i] = clamped * safeStep;
  }
  return out;
}

function finiteDifferenceGradient(values: Float32Array, teacher: Float32Array, stepSize: number, qmin: number, qmax: number): number {
  const eps = Math.max(stepSize * 1e-3, 1e-5);
  const lossPlus = meanSquaredError(surrogateQuantizeSymmetric(values, stepSize + eps, qmin, qmax), teacher);
  const lossMinus = meanSquaredError(surrogateQuantizeSymmetric(values, Math.max(stepSize - eps, 1e-8), qmin, qmax), teacher);
  return (lossPlus - lossMinus) / (2 * eps);
}

function analyticLsqGradient(values: Float32Array, teacher: Float32Array, stepSize: number, qmin: number, qmax: number) {
  const safeStep = Math.max(stepSize, 1e-10);
  const quantized = new Float32Array(values.length);
  let gradient = 0;
  let clippedCount = 0;
  for (let i = 0; i < values.length; i++) {
    const scaled = values[i]! / safeStep;
    const clamped = Math.max(qmin, Math.min(qmax, scaled));
    const dequantized = clamped * safeStep;
    quantized[i] = dequantized;

    let dqds = 0;
    if (scaled < qmin) {
      dqds = qmin;
      clippedCount++;
    } else if (scaled > qmax) {
      dqds = qmax;
      clippedCount++;
    }

    gradient += 2 * (dequantized - teacher[i]!) * dqds;
  }

  gradient /= Math.max(values.length, 1);
  return {
    quantized,
    gradient,
    clippedFraction: clippedCount / Math.max(values.length, 1),
  };
}

export interface LsqOptimizerState {
  timestamp: string;
  module: 'gradient_verified_lsq_optimizer';
  optimizerProfile: 'deterministic_symmetric_step_size_contract';
  gradientVerificationReady: boolean;
  optimizerStepReady: boolean;
  lossReductionReady: boolean;
  optimizerArtifactReady: boolean;
  quantBits: number;
  tensorLength: number;
  qmin: number;
  qmax: number;
  initialStepSize: number;
  updatedStepSize: number;
  learningRate: number;
  gradientScale: number;
  initialLoss: number;
  updatedLoss: number;
  finiteDifferenceGradient: number;
  analyticGradient: number;
  relativeGradientError: number;
  cosineSimilarityAfterUpdate: number;
  clippedFraction: number;
  anchors: string[];
  residualFrontier: string[];
}

export function buildGradientVerifiedLsqOptimizerState(): LsqOptimizerState {
  const values = makeDeterministicWeights(256, 0.37, 0.92);
  const teacher = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    teacher[i] = values[i]! * 0.91 + Math.sin((i + 1) * 0.07) * 0.015;
  }

  const quantBits = 4;
  const qmax = (1 << quantBits) / 2 - 1;
  const qmin = -qmax;
  let absMax = 0;
  for (let i = 0; i < values.length; i++) {
    absMax = Math.max(absMax, Math.abs(values[i]!));
  }
  const initialStepSize = Math.max(absMax / qmax, 1e-10);
  const initialQuantized = surrogateQuantizeSymmetric(values, initialStepSize, qmin, qmax);
  const initialLoss = meanSquaredError(initialQuantized, teacher);

  const analytic = analyticLsqGradient(values, teacher, initialStepSize, qmin, qmax);
  const numericGradient = finiteDifferenceGradient(values, teacher, initialStepSize, qmin, qmax);
  const gradientScale = 1 / Math.sqrt(values.length * Math.max(qmax, 1));
  const scaledGradient = analytic.gradient * gradientScale;
  const learningRate = initialStepSize * 0.25;
  const updatedStepSize = Math.max(initialStepSize - learningRate * scaledGradient, 1e-8);
  const updatedQuantized = surrogateQuantizeSymmetric(values, updatedStepSize, qmin, qmax);
  const updatedLoss = meanSquaredError(updatedQuantized, teacher);

  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < updatedQuantized.length; i++) {
    dot += updatedQuantized[i]! * teacher[i]!;
    normA += updatedQuantized[i]! * updatedQuantized[i]!;
    normB += teacher[i]! * teacher[i]!;
  }
  const cosineSimilarityAfterUpdate = dot / ((Math.sqrt(normA) * Math.sqrt(normB)) || 1);
  const relativeGradientError =
    Math.abs(analytic.gradient - numericGradient) /
    Math.max(Math.abs(analytic.gradient), Math.abs(numericGradient), 1e-8);

  const gradientVerificationReady =
    Number.isFinite(relativeGradientError) &&
    relativeGradientError < 0.55 &&
    Math.sign(analytic.gradient || 0) === Math.sign(numericGradient || 0);
  const optimizerStepReady = Number.isFinite(updatedStepSize) && updatedStepSize > 0 && updatedStepSize !== initialStepSize;
  const lossReductionReady = updatedLoss < initialLoss;

  return {
    timestamp: new Date().toISOString(),
    module: 'gradient_verified_lsq_optimizer',
    optimizerProfile: 'deterministic_symmetric_step_size_contract',
    gradientVerificationReady,
    optimizerStepReady,
    lossReductionReady,
    optimizerArtifactReady: gradientVerificationReady && optimizerStepReady && lossReductionReady,
    quantBits,
    tensorLength: values.length,
    qmin,
    qmax,
    initialStepSize,
    updatedStepSize,
    learningRate,
    gradientScale,
    initialLoss,
    updatedLoss,
    finiteDifferenceGradient: numericGradient,
    analyticGradient: analytic.gradient,
    relativeGradientError,
    cosineSimilarityAfterUpdate,
    clippedFraction: analytic.clippedFraction,
    anchors: [
      'src/core/precision_calibration.ts',
      'src/core/quantizer.ts',
      'src/tools/turboquant_quantize.ts',
    ],
    residualFrontier: [
      'hybrid_cpu_gpu_training_runtime',
      'full_qat_training_loop',
      'dataset_backed_distillation',
      'multi_parameter_lsq_optimizer',
      'activation_lsq_backward',
      'on_device_distillation',
    ],
  };
}
