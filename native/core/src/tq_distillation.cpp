/**
 * TurboQuant Core — Knowledge Distillation Implementation
 * Teacher-student quantization distillation
 */

#include "tq_distillation.h"
#include "tq_core.h"
#include <cmath>
#include <cstring>
#include <cstdlib>

// ── Make deterministic teacher logits ─────────────────────────────────────────
void tq_distillation_make_teacher_logits(
    float* out, uint32_t batch_size, uint32_t vocab_size,
    float phase, float amplitude
) {
    if (!out) return;
    for (uint32_t r = 0; r < batch_size; r++) {
        for (uint32_t c = 0; c < vocab_size; c++) {
            float x = (float)((r + 1) * (c + 1));
            out[r * vocab_size + c] = amplitude * (
                sinf(x * (phase + 0.031f)) * 0.58f +
                cosf((c + 1.0f) * (phase + 0.083f)) * 0.24f +
                sinf((r + 1.0f) * 0.17f + c * 0.13f) * 0.18f
            );
        }
    }
}

// ── Make student base ─────────────────────────────────────────────────────────
void tq_distillation_make_student_base(
    float* out, uint32_t batch_size, uint32_t vocab_size,
    float phase, float amplitude
) {
    if (!out) return;
    for (uint32_t r = 0; r < batch_size; r++) {
        for (uint32_t c = 0; c < vocab_size; c++) {
            float x = (float)((r + 1) * (c + 1));
            out[r * vocab_size + c] = amplitude * (
                sinf(x * (phase + 0.131f)) * 0.57f +
                cosf(x * (phase + 0.071f)) * 0.29f +
                sinf(x * 0.041f + phase) * 0.14f
            );
        }
    }
}

// ── Softmax row ───────────────────────────────────────────────────────────────
void tq_distillation_softmax_row(
    const float* logits, uint32_t offset, uint32_t width,
    float temperature, float* out
) {
    if (!logits || !out) return;
    float max_val = -1e30f;
    for (uint32_t i = 0; i < width; i++) {
        float v = logits[offset + i] / temperature;
        if (v > max_val) max_val = v;
    }

    float sum = 0;
    for (uint32_t i = 0; i < width; i++) {
        out[i] = expf(logits[offset + i] / temperature - max_val);
        sum += out[i];
    }

    float inv = sum > 0 ? 1.0f / sum : 0;
    for (uint32_t i = 0; i < width; i++) {
        out[i] *= inv;
    }
}

// ── KL divergence ───────────────────────────────────────────────────────────
float tq_distillation_kl_divergence(const float* p, const float* q, uint32_t size) {
    if (!p || !q || size == 0) return 0.0f;
    double total = 0.0;
    for (uint32_t i = 0; i < size; i++) {
        float pi = fmaxf(p[i], 1e-8f);
        float qi = fmaxf(q[i], 1e-8f);
        total += pi * logf(pi / qi);
    }
    return (float)total;
}

// ── MSE ─────────────────────────────────────────────────────────────────────
float tq_distillation_mse(const float* a, const float* b, uint32_t size) {
    if (!a || !b || size == 0) return 0.0f;
    double total = 0.0;
    for (uint32_t i = 0; i < size; i++) {
        float d = a[i] - b[i];
        total += d * d;
    }
    return (float)(total / size);
}

// ── Quantize symmetric ────────────────────────────────────────────────────────
void tq_distillation_quantize_symmetric(
    const float* values, uint32_t count, uint8_t bits,
    float* out, float* step_size_out
) {
    if (!values || !out) return;
    int qmax = (1 << (bits - 1)) - 1;
    float abs_max = 0;
    for (uint32_t i = 0; i < count; i++) {
        float a = values[i];
        if (a < 0) a = -a;
        if (a > abs_max) abs_max = a;
    }
    float step = fmaxf(abs_max / (float)qmax, 1e-10f);
    if (step_size_out) *step_size_out = step;

    for (uint32_t i = 0; i < count; i++) {
        float q = values[i] / step;
        int qi = (int)(q + ((q >= 0) ? 0.5f : -0.5f));
        if (qi < -qmax) qi = -qmax;
        if (qi > qmax) qi = qmax;
        out[i] = (float)qi * step;
    }
}

// ── Full distillation run ────────────────────────────────────────────────────
int tq_distillation_run(
    uint32_t batch_size, uint32_t vocab_size,
    uint8_t bits, float temperature,
    float alpha, float beta, float learning_rate,
    tq_distillation_result* result
) {
    if (!result) return TQ_ERR_NULL;
    memset(result, 0, sizeof(tq_distillation_result));

    result->temperature = temperature > 0 ? temperature : 1.75f;
    result->quant_bits = bits > 0 ? bits : 4;
    result->alpha = alpha > 0 ? alpha : 0.7f;
    result->beta = beta > 0 ? beta : 0.3f;
    result->learning_rate = learning_rate > 0 ? learning_rate : 0.85f;

    uint32_t total_size = batch_size * vocab_size;
    float* teacher_logits = (float*)malloc(total_size * sizeof(float));
    float* student_base = (float*)malloc(total_size * sizeof(float));
    float* initial_student_raw = (float*)malloc(total_size * sizeof(float));
    float* initial_student = (float*)malloc(total_size * sizeof(float));
    float* updated_student_raw = (float*)malloc(total_size * sizeof(float));
    float* updated_student = (float*)malloc(total_size * sizeof(float));
    float* gradient = (float*)malloc(total_size * sizeof(float));

    if (!teacher_logits || !student_base || !initial_student_raw || !initial_student ||
        !updated_student_raw || !updated_student || !gradient) {
        free(teacher_logits); free(student_base); free(initial_student_raw);
        free(initial_student); free(updated_student_raw); free(updated_student);
        free(gradient);
        return TQ_ERR_ALLOC;
    }

    // Generate logits
    tq_distillation_make_teacher_logits(teacher_logits, batch_size, vocab_size, 0.19f, 1.0f);
    tq_distillation_make_student_base(student_base, batch_size, vocab_size, 0.27f, 0.88f);

    // Mix initial student
    for (uint32_t i = 0; i < total_size; i++) {
        initial_student_raw[i] = student_base[i] * 0.93f + teacher_logits[i] * 0.07f;
    }

    // Initial quantization
    tq_distillation_quantize_symmetric(initial_student_raw, total_size, result->quant_bits, initial_student, nullptr);

    // Compute losses
    float teacher_entropy_sum = 0;
    float initial_kl_sum = 0;

    for (uint32_t r = 0; r < batch_size; r++) {
        float* teacher_prob = (float*)malloc(vocab_size * sizeof(float));
        float* student_prob = (float*)malloc(vocab_size * sizeof(float));

        tq_distillation_softmax_row(teacher_logits, r * vocab_size, vocab_size, result->temperature, teacher_prob);
        tq_distillation_softmax_row(initial_student, r * vocab_size, vocab_size, result->temperature, student_prob);

        for (uint32_t i = 0; i < vocab_size; i++) {
            float p = fmaxf(teacher_prob[i], 1e-8f);
            teacher_entropy_sum -= p * logf(p);
            gradient[r * vocab_size + i] = result->alpha * (student_prob[i] - teacher_prob[i]) * result->temperature;
        }
        initial_kl_sum += tq_distillation_kl_divergence(teacher_prob, student_prob, vocab_size);

        free(teacher_prob); free(student_prob);
    }

    result->teacher_entropy_mean = teacher_entropy_sum / batch_size;
    result->initial_kl_mean = initial_kl_sum / batch_size;

    // Gradient update
    for (uint32_t i = 0; i < total_size; i++) {
        gradient[i] += result->beta * 2.0f * (initial_student[i] - teacher_logits[i]) / total_size;
    }

    // Apply update
    float update_norm = 0;
    for (uint32_t i = 0; i < total_size; i++) {
        float delta = result->learning_rate * gradient[i];
        updated_student_raw[i] = initial_student_raw[i] - delta;
        update_norm += delta * delta;
    }
    result->update_norm = sqrtf(update_norm);

    // Updated quantization
    tq_distillation_quantize_symmetric(updated_student_raw, total_size, result->quant_bits, updated_student, &result->quant_step_size);

    // Compute updated KL
    float updated_kl_sum = 0;
    for (uint32_t r = 0; r < batch_size; r++) {
        float* teacher_prob = (float*)malloc(vocab_size * sizeof(float));
        float* student_prob = (float*)malloc(vocab_size * sizeof(float));

        tq_distillation_softmax_row(teacher_logits, r * vocab_size, vocab_size, result->temperature, teacher_prob);
        tq_distillation_softmax_row(updated_student, r * vocab_size, vocab_size, result->temperature, student_prob);

        updated_kl_sum += tq_distillation_kl_divergence(teacher_prob, student_prob, vocab_size);
        free(teacher_prob); free(student_prob);
    }
    result->updated_kl_mean = updated_kl_sum / batch_size;

    // Anchor MSE
    result->initial_anchor_mse = tq_distillation_mse(initial_student, teacher_logits, total_size);
    result->updated_anchor_mse = tq_distillation_mse(updated_student, teacher_logits, total_size);

    // Composite loss
    float t_sq = result->temperature * result->temperature;
    result->initial_composite_loss = result->alpha * result->initial_kl_mean * t_sq + result->beta * result->initial_anchor_mse;
    result->updated_composite_loss = result->alpha * result->updated_kl_mean * t_sq + result->beta * result->updated_anchor_mse;

    free(teacher_logits); free(student_base); free(initial_student_raw);
    free(initial_student); free(updated_student_raw); free(updated_student);
    free(gradient);

    return TQ_OK;
}