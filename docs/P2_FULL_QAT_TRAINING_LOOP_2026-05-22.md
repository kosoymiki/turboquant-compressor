# P2 Full QAT Training Loop 2026-05-22

This document defines the next bounded execution module inside `p2_precision_training_runtime`: `full_qat_training_loop`.

The honest claim here is not a dataset-backed trainer. It is only that repo evidence now contains a bounded two-stage QAT loop contract that composes the already-closed calibration, LSQ optimizer, distillation, and hybrid CPU/WASM/GPU routing modules into one improving loop artifact.

## Scope

- loop composition across calibration, optimizer, distillation, and hybrid routing stages
- explicit stage order and route admission
- aggregate loss improvement across the bounded loop
- canonical loop artifact and contract

## Non-Goals

- dataset-backed training
- long-running multi-epoch optimization
- live GPU backward pass
- multi-device gradient synchronization

## Current Closure Evidence

- deterministic artifact written at `forensics/full-qat-training-loop-state.json`
- canonical contract written at `forensics/precision/full-qat-training-loop-contract.json`
- explicit stage order across four already-closed bounded modules
- aggregate post-loop loss lower than aggregate pre-loop loss

## Closure Boundary

This module is considered honestly closed because repo evidence now shows:

- a full bounded QAT loop exists, not just isolated pieces
- the loop runs through the expected stages in order
- the aggregate loop objective improves

What is still not claimed:

- dataset-backed distillation
- long-running or convergent training
- full live GPU backward/runtime training kernels
- distributed or multi-device training
