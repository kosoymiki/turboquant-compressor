# QJL Roadmap

QJL is not implemented in LEVEL_0.

TurboQuant uses QJL on the residual after MSE quantization to build an unbiased inner-product estimator. This project currently does not implement that estimator.

Current behavior:

- `includeQJL` enables an experimental residual sketch serialization path.
- The public benchmark path keeps `include_qjl = false`.
- Search does not apply QJL correction, even when an experimental residual payload exists.
- No unbiased-estimator or paper-faithfulness claim is allowed for the current implementation.

Implementation requirements for LEVEL_1:

1. Store residual sketch with explicit qjlOffset/qjlLength.
2. Define projection count and seed derivation.
3. Implement estimator formula.
4. Add synthetic unbiasedness tests.
5. Add ranking recall tests.
6. Compare QJL-on-K, QJL-on-V, and QJL-on-both for KV-cache contexts.
