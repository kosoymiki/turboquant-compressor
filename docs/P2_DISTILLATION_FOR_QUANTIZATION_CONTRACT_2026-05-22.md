# P2 Distillation For Quantization Contract 2026-05-22

This document defines the next bounded execution module inside `p2_precision_training_runtime`: `distillation_for_quantization_contract`.

The honest claim here is not full on-device distillation. It is only that repo evidence now contains a deterministic teacher-student contract with softened targets, explicit distillation loss, a quantized student path, and a bounded update that reduces composite loss.

## Scope

- deterministic teacher and quantized student logits
- temperature-softened target distribution
- explicit distillation loss with KL term plus logit-anchor MSE term
- single bounded student update
- canonical artifact and contract for later training-lane composition

## Non-Goals

- dataset-backed distillation
- repeated student training loop
- on-device teacher inference runtime
- hybrid `CPU+GPU` training runtime
- full low-precision QAT closure

## Current Closure Evidence

- deterministic artifact written at `forensics/distillation-for-quantization-state.json`
- canonical contract written at `forensics/precision/distillation-for-quantization-contract.json`
- softened teacher targets with explicit temperature
- quantized student path with post-update composite loss reduction

## Closure Boundary

This module is considered honestly closed because repo evidence now shows:

- an explicit teacher-student pair
- a real distillation objective rather than a vague placeholder
- a quantized student update that improves the bounded objective

What is still not claimed:

- real dataset-backed distillation
- multi-step student optimization
- full on-device distillation runtime
- broad hybrid `CPU+GPU` training runtime
