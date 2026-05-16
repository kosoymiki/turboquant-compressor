# License Strategy — turboquant-compressor

## Current License: GPL-3.0-or-later

turboquant-compressor is now licensed under **GPL-3.0-or-later**.

## Rationale

This project incorporates donor code and scientific implementations from:

- **0xSero/turboquant** — GPL-licensed TurboQuant KV-cache quantization implementation
- **TurboQuant paper** — scientific research on Beta-distributed coordinates and Lloyd-Max quantization
- **vLLM/Triton** — GPU runtime integration

GPL-3.0-or-later allows:

- Full incorporation of donor GPL code with proper attribution
- Derivative works and redistribution
- Scientific collaboration and research
- Clear license compatibility with donor projects

## Attribution

Donor 0xSero/turboquant is acknowledged in:

- `NOTICE` file
- `docs/DONOR_EVIDENCE_FINAL.md`
- `docs/DONOR_FILE_TO_FILE_SCAN.md`

## Donor Files

The following donor files have been incorporated or ported:

| Donor File | Local Target | License Status |
|---|---|---|
| codebook.py | src/core/beta_sphere.ts, src/core/codebook.ts | GPL-3.0 |
| rotation.py | src/core/rotation.ts | GPL-3.0 |
| quantizer.py | src/core/turboquant_mse.ts, src/core/turboquant_prod.ts | GPL-3.0 |
| kv_cache.py | src/kv/value_quant.ts, src/kv/compressed_kv_cache.ts | GPL-3.0 |
| score.py | src/kv/hybrid_score.ts | GPL-3.0 |
| triton_kernels.py | kernels/triton/* | GPL-3.0 |
| integration/vllm.py | python/turboquant_ext/integration/vllm.py | GPL-3.0 |

## Compliance

This project complies with GPL-3.0-or-later by:

1. Including full license text in LICENSE file
2. Including attribution in NOTICE file
3. Documenting donor code in DONOR_EVIDENCE_FINAL.md
4. Maintaining file-to-file scan in DONOR_FILE_TO_FILE_SCAN.md
5. Providing corresponding source for all distributions

## Future Changes

This license may be updated to a later version of GPL or dual-licensed
for specific subpackages if needed for scientific collaboration.
