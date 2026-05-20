# Optimization Order

This document defines the permanent engineering order for performance work in TurboQuant.

## Rule

Optimization work must follow this sequence:

1. Profile `scripts/open-test-local.mjs` and `src/tools/search.ts` for:
   - score distortion
   - ranking loss
2. Introduce a stronger reproducible baseline vectorizer alongside token-hash.
3. Resolve the experimental QJL path:
   - either implement a real estimator/search-correction path
   - or keep it explicitly research-only
4. Compare:
   - uniform quantization
   - Lloyd-Max quantization

   on the same open corpus with the same gates and metrics.
5. Only then proceed to native/OpenCL optimization.

## Why This Order Exists

- A faster weak retrieval path is still a weak retrieval path.
- Current public evidence is strongest for `LEVEL_0_TURBOQUANT_INSPIRED_MVP`, not for a paper-faithful TurboQuant path.
- Native acceleration is valuable only after the retrieval stack is numerically defensible and reproducibly benchmarked.

## Enforcement

- README, audit docs, and future optimization plans must respect this order.
- New performance claims must state which step of this order they belong to.
- Any attempt to skip directly to native/OpenCL tuning must be treated as policy drift unless retrieval-quality proof is already in place.
