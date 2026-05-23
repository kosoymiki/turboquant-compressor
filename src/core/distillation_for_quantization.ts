function makeDeterministicLogits(rows: number, cols: number, phase: number, amplitude: number): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = (r + 1) * (c + 1);
      out[r * cols + c] =
        amplitude * (
          Math.sin(x * (phase + 0.031)) * 0.58 +
          Math.cos((c + 1) * (phase + 0.083)) * 0.24 +
          Math.sin((r + 1) * 0.17 + c * 0.13) * 0.18
        );
    }
  }
  return out;
}

function softmaxRow(logits: Float32Array, offset: number, width: number, temperature: number): Float32Array {
  const out = new Float32Array(width);
  let max = -Infinity;
  for (let i = 0; i < width; i++) {
    max = Math.max(max, logits[offset + i]! / temperature);
  }
  let sum = 0;
  for (let i = 0; i < width; i++) {
    const value = Math.exp(logits[offset + i]! / temperature - max);
    out[i] = value;
    sum += value;
  }
  const inv = sum > 0 ? 1 / sum : 0;
  for (let i = 0; i < width; i++) out[i] = out[i]! * inv;
  return out;
}

function klDivergence(p: Float32Array, q: Float32Array): number {
  let total = 0;
  for (let i = 0; i < p.length; i++) {
    const pi = Math.max(p[i]!, 1e-8);
    const qi = Math.max(q[i]!, 1e-8);
    total += pi * Math.log(pi / qi);
  }
  return total;
}

function mse(a: Float32Array, b: Float32Array): number {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i]! - b[i]!;
    total += d * d;
  }
  return total / Math.max(a.length, 1);
}

function quantizeSymmetricTensor(values: Float32Array, bits: number) {
  const qmax = (1 << bits) / 2 - 1;
  let absMax = 0;
  for (let i = 0; i < values.length; i++) absMax = Math.max(absMax, Math.abs(values[i]!));
  const step = Math.max(absMax / Math.max(qmax, 1), 1e-10);
  const quantized = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const q = Math.max(-qmax, Math.min(qmax, Math.round(values[i]! / step)));
    quantized[i] = q * step;
  }
  return { quantized, step };
}

export interface DistillationForQuantizationState {
  timestamp: string;
  module: 'distillation_for_quantization_contract';
  distillationProfile: 'deterministic_teacher_student_quantized_logits';
  teacherStudentPairReady: boolean;
  softenedTargetReady: boolean;
  distillationLossReady: boolean;
  studentUpdateReady: boolean;
  contractArtifactReady: boolean;
  batchSize: number;
  vocabSize: number;
  temperature: number;
  quantBits: number;
  quantStepSize: number;
  teacherEntropyMean: number;
  initialKlMean: number;
  updatedKlMean: number;
  initialAnchorMse: number;
  updatedAnchorMse: number;
  initialCompositeLoss: number;
  updatedCompositeLoss: number;
  updateNorm: number;
  anchors: string[];
  residualFrontier: string[];
}

export function buildDistillationForQuantizationState(): DistillationForQuantizationState {
  const batchSize = 5;
  const vocabSize = 16;
  const temperature = 1.75;
  const quantBits = 4;

  const teacherLogits = makeDeterministicLogits(batchSize, vocabSize, 0.19, 1.0);
  const studentBase = makeDeterministicLogits(batchSize, vocabSize, 0.27, 0.88);
  const initialStudentRaw = new Float32Array(studentBase.length);
  for (let i = 0; i < studentBase.length; i++) {
    initialStudentRaw[i] = studentBase[i]! * 0.93 + teacherLogits[i]! * 0.07;
  }
  const initialQuant = quantizeSymmetricTensor(initialStudentRaw, quantBits);
  const initialStudent = initialQuant.quantized;

  const alpha = 0.7;
  const beta = 0.3;
  let teacherEntropySum = 0;
  let initialKlMean = 0;
  let updatedKlMean = 0;

  const gradient = new Float32Array(initialStudentRaw.length);
  for (let row = 0; row < batchSize; row++) {
    const offset = row * vocabSize;
    const teacherProb = softmaxRow(teacherLogits, offset, vocabSize, temperature);
    const studentProb = softmaxRow(initialStudent, offset, vocabSize, temperature);

    let entropy = 0;
    for (let i = 0; i < vocabSize; i++) {
      const p = Math.max(teacherProb[i]!, 1e-8);
      entropy += -p * Math.log(p);
      gradient[offset + i] = alpha * (studentProb[i]! - teacherProb[i]!) * temperature;
    }
    teacherEntropySum += entropy;
    initialKlMean += klDivergence(teacherProb, studentProb);
  }
  initialKlMean /= batchSize;

  for (let i = 0; i < gradient.length; i++) {
    gradient[i] = gradient[i]! + beta * 2 * (initialStudent[i]! - teacherLogits[i]!) / gradient.length;
  }

  const learningRate = 0.85;
  const updatedStudentRaw = new Float32Array(initialStudentRaw.length);
  let updateNorm = 0;
  for (let i = 0; i < updatedStudentRaw.length; i++) {
    const delta = learningRate * gradient[i]!;
    updatedStudentRaw[i] = initialStudentRaw[i]! - delta;
    updateNorm += delta * delta;
  }
  updateNorm = Math.sqrt(updateNorm);

  const updatedQuant = quantizeSymmetricTensor(updatedStudentRaw, quantBits);
  const updatedStudent = updatedQuant.quantized;

  for (let row = 0; row < batchSize; row++) {
    const offset = row * vocabSize;
    const teacherProb = softmaxRow(teacherLogits, offset, vocabSize, temperature);
    const studentProb = softmaxRow(updatedStudent, offset, vocabSize, temperature);
    updatedKlMean += klDivergence(teacherProb, studentProb);
  }
  updatedKlMean /= batchSize;

  const initialAnchorMse = mse(initialStudent, teacherLogits);
  const updatedAnchorMse = mse(updatedStudent, teacherLogits);
  const initialCompositeLoss = alpha * initialKlMean * (temperature * temperature) + beta * initialAnchorMse;
  const updatedCompositeLoss = alpha * updatedKlMean * (temperature * temperature) + beta * updatedAnchorMse;

  return {
    timestamp: new Date().toISOString(),
    module: 'distillation_for_quantization_contract',
    distillationProfile: 'deterministic_teacher_student_quantized_logits',
    teacherStudentPairReady: teacherLogits.length === initialStudent.length && teacherLogits.length > 0,
    softenedTargetReady: temperature > 1 && Number.isFinite(teacherEntropySum / batchSize),
    distillationLossReady: Number.isFinite(initialCompositeLoss) && Number.isFinite(updatedCompositeLoss),
    studentUpdateReady: updatedCompositeLoss < initialCompositeLoss && updateNorm > 0,
    contractArtifactReady:
      teacherLogits.length === initialStudent.length &&
      temperature > 1 &&
      Number.isFinite(initialCompositeLoss) &&
      Number.isFinite(updatedCompositeLoss) &&
      updatedCompositeLoss < initialCompositeLoss &&
      updateNorm > 0,
    batchSize,
    vocabSize,
    temperature,
    quantBits,
    quantStepSize: updatedQuant.step,
    teacherEntropyMean: teacherEntropySum / batchSize,
    initialKlMean,
    updatedKlMean,
    initialAnchorMse,
    updatedAnchorMse,
    initialCompositeLoss,
    updatedCompositeLoss,
    updateNorm,
    anchors: [
      'src/core/precision_calibration.ts',
      'src/core/lsq_optimizer.ts',
      'src/core/hybrid_score.ts',
      'src/tools/turboquant_quantize.ts',
    ],
    residualFrontier: [
      'hybrid_cpu_gpu_training_runtime',
      'full_qat_training_loop',
      'dataset_backed_distillation',
      'multi_step_student_training_loop',
      'on_device_teacher_runtime',
    ],
  };
}
