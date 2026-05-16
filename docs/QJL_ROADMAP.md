# QJL Roadmap

QJL is not implemented in LEVEL_0.

TurboQuant uses QJL on the residual after MSE quantization to build an unbiased inner-product estimator. This project currently does not implement that estimator.

Current behavior:

- `includeQJL` is accepted for forward compatibility.
- No residual sketch is stored.
- `include_qjl` in output must remain false until actual sketch exists.
- Search does not apply QJL correction.

Implementation requirements for LEVEL_1:

1. Store residual sketch with explicit qjlOffset/qjlLength.
2. Define projection count and seed derivation.
3. Implement estimator formula.
4. Add synthetic unbiasedness tests.
5. Add ranking recall tests.
6. Compare QJL-on-K, QJL-on-V, and QJL-on-both for KV-cache contexts.