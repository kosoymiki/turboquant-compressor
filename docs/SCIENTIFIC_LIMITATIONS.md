# Scientific Limitations

## Overview

This document describes the scientific limitations of TurboQuant Compressor and its relationship to the original TurboQuant paper.

## Algorithm Level

The public benchmarked implementation is labeled as `LEVEL_0_TURBOQUANT_INSPIRED_MVP`, indicating:

- **Inspired by** TurboQuant paper concepts
- **Not a faithful implementation** of the paper
- **Proof of concept** for mobile deployment
- **Known deviations** from paper methodology

## TurboQuant Paper Deviations

### 1. Random Rotation

**Paper Claim:**
- Random rotation using Hadamard matrix
- Beta-distributed coordinates after rotation

**Our Implementation:**
- FWHT-based sign-flip rotation
- Deterministic sign pattern using Mulberry32 PRNG
- No guarantee of Beta distribution

**Impact:**
- Rotation properties may differ from paper
- Compression quality may vary

### 2. Scalar Quantization

**Paper Claim:**
- Optimal scalar quantizer for Beta distribution
- Lloyd-Max algorithm for codebook training

**Our Implementation:**
- Public path uses a precomputed TurboQuant Beta Lloyd-Max scalar codebook for 2/3/4-bit payloads
- 8-bit mode falls back to uniform symmetric quantization
- Public path does not perform corpus-trained or per-dataset codebook adaptation

**Impact:**
- Quantization error is lower than the old fixed-uniform baseline on the current public path, but still lacks dataset-specific optimality guarantees
- No claim of paper-faithful training on deployment data

### 3. QJL Residual Stage

**Paper Claim:**
- Johnson-Lindenstrauss projection for residual
- 1-bit quantization with unbiased estimator

**Our Implementation:**
- Public search path does not implement QJL correction
- Experimental residual sketch serialization exists behind `includeQJL`
- No unbiased estimator guarantee

**Impact:**
- Cannot achieve paper's compression ratios
- No theoretical guarantees on approximation quality

### 4. Norm Preservation

**Paper Claim:**
- Rotation preserves L2 norm (orthonormal)

**Our Implementation:**
- Uses normalized FWHT
- Norm is explicitly calculated and stored
- Decoding uses stored norms

**Impact:**
- Norms are preserved but not implicitly
- Additional metadata required

## Known Issues

### 1. Fixed Public Codebook

The shipped 2/3/4-bit path uses a deterministic precomputed Beta Lloyd-Max codebook. This means:

- Inputs are still clipped to the supported scalar range
- The codebook is dimension-aware but not dataset-trained at runtime
- Deployment data may still be mismatched to the public codebook

### 2. No Corpus-Specific Training

Unlike a fully trained deployment pipeline:

- No per-corpus iterative optimization
- No online centroid adaptation
- No convergence claim on user data

### 3. No Residual Quantization

Without a search-time QJL estimator:

- Cannot achieve sub-bit precision
- No unbiased inner-product estimation
- Limited compression ratios

## Research Context

### Related Papers

1. **TurboQuant** (arXiv:2504.19874)
   - Original paper describing the method
   - Claims near-lossless compression

2. **PolarQuant** (arXiv:2502.00527)
   - Polar transformation for KV cache
   - Different approach to quantization

3. **RaBitQ** (arXiv:2604.19528)
   - Revisits TurboQuant claims
   - Questions reproducibility of results

### Critical Assessment

Recent research (RaBitQ) suggests:

- TurboQuant may not show consistent advantages
- Reproducibility issues in original claims
- Need for careful validation

## Recommendations

### For Production Use

1. **Validate on your data**
   - Test compression quality on representative dataset
   - Measure reconstruction error
   - Compare with other methods

2. **Set expectations**
   - Not suitable for precision-critical applications
   - Best for approximate nearest neighbor search
   - Monitor quality metrics

### For Research

1. **Implement paper faithfully**
   - Add Lloyd-Max codebook training
   - Implement QJL residual stage
   - Validate against paper claims

2. **Compare methods**
   - Implement RaBitQ for comparison
   - Test on standard benchmarks
   - Publish reproducible results

## Disclaimer

This implementation is provided as-is for educational and experimental purposes. It does not make claims of paper-faithfulness or optimality. Users should validate the implementation for their specific use case.
