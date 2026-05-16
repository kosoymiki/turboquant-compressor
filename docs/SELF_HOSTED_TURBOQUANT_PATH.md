# Self-Hosted TurboQuant Path

TurboQuant is relevant when you control the inference runtime.

## Applicable Scenarios

- Self-hosted vLLM
- SGLang
- Custom inference engine
- Local vector search/indexing experiments

## Not Applicable

- Anthropic-hosted Claude Code backend
- Closed Claude inference stack
- Direct Anthropic token billing

## Before Claiming QJL

To claim QJL (Quantized Johnson-Lindenstrauss) capability:

- qjlOffset/qjlLength > 0
- Correction term applied during search/inference
- Estimator documented
- TypeScript unbiasedness test
- Python oracle
- Ranking recall benchmark
- Variance reported

## LEVEL_0 Status

Current implementation:
- No QJL payload stored
- No correction term applied
- Deterministic token-hash vectors only
- Not a substitute for semantic embeddings