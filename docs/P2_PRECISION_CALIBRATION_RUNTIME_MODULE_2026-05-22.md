# P2 Precision Calibration Runtime Module 2026-05-22

This document defines the first honest bounded execution module inside `p2_precision_training_runtime`: `precision_calibration_runtime`.

It is intentionally narrower than full low-precision training. The current repo does not justify pretending that on-device distillation, full QAT training loops, or broad hybrid CPU+GPU training runtime are already the next shippable unit.

## Scope

- observer-style activation and weight range capture
- fake-quant policy shape for forward-path error injection
- learnable or policy-driven step-size bookkeeping compatible with `LSQ`-style calibration
- calibration artifact and runtime evidence contract for quantized kernel policy selection
- explicit boundary with existing live inference kernels rather than a separate opaque training stack

## Non-Goals

- full low-precision training loop closure
- gradient-verified `LSQ` optimizer implementation
- on-device distillation
- data-free or zero-shot quantization claims
- broad `USM` / hybrid CPU+GPU training runtime closure

## Why This Is The First Module

- the local repo currently has almost no honest shipped training-runtime surface under `135-157`
- observer/fake-quant/step-size calibration is the smallest technically coherent module that can connect future precision work to existing inference/runtime evidence
- this matches real donor structure better than jumping straight to distillation or full low-precision training

## External Donor Basis

- PyTorch quantization support / observer and fake-quant APIs
  - https://docs.pytorch.org/docs/stable/quantization-support
- PyTorch QAT for LLMs overview
  - https://docs.pytorch.org/blog/quantization-aware-training/
- Learned Step Size Quantization
  - https://arxiv.org/abs/1902.08153
- Intel Neural Compressor distillation-for-quantization docs
  - https://intel.github.io/neural-compressor/latest/docs/source/distillation_quantization.html

## Closure Direction

This module is now closed as a bounded artifact/runtime contract, not as a full training runtime.

## Current Closure Evidence

- deterministic calibration artifact written at `forensics/precision-calibration-runtime-state.json`
- canonical contract written at `forensics/precision/precision-calibration-runtime-contract.json`
- observer state over three explicit `TQ` surfaces:
  - rotated activation surface
  - weight block surface
  - grouped KV-value surface
- fake-quant policy shape represented explicitly:
  - activation affine per-tensor
  - weight symmetric per-tensor
  - KV values affine per-group
- explicit step-size contract:
  - `LSQ`-compatible learnable symmetric step size on weights
  - min/max affine activation step size
  - grouped min/max step-size distribution on KV values

## Closure Boundary

This module is considered honestly closed because repo evidence now shows:

- bounded calibration state, not hand-written static quant params alone
- observer or equivalent range-capture artifacts tied to real `TQ` tensor surfaces
- fake-quant or quantization-noise policy represented in runtime contracts
- step-size policy that is explicit enough to map to future `LSQ`-style or adaptive-precision work

What is still not claimed:

- full low-precision training loop closure
- gradient-verified `LSQ` optimizer updates
- on-device distillation
- broad hybrid `CPU+GPU` training runtime
