"""
Fidelity oracle for compression quality assessment.
Computes reconstruction error metrics.
"""

import numpy as np
from typing import Dict, List
try:
    from .reference_quantizer import uniform_symmetric_quantize
except ImportError:
    from reference_quantizer import uniform_symmetric_quantize

def compute_mse(original: List[float], reconstructed: List[float]) -> float:
    """Mean squared error between original and reconstructed vectors."""
    orig = np.array(original)
    recon = np.array(reconstructed)
    return float(np.mean((orig - recon) ** 2))

def compute_rmse(original: List[float], reconstructed: List[float]) -> float:
    """Root mean squared error."""
    return float(np.sqrt(compute_mse(original, reconstructed)))

def compute_max_error(original: List[float], reconstructed: List[float]) -> float:
    """Maximum absolute error."""
    orig = np.array(original)
    recon = np.array(reconstructed)
    return float(np.max(np.abs(orig - recon)))

def compute_snr(original: List[float], reconstructed: List[float]) -> float:
    """Signal-to-noise ratio in dB."""
    orig = np.array(original)
    recon = np.array(reconstructed)
    signal_power = np.mean(orig ** 2)
    noise_power = np.mean((orig - recon) ** 2)
    if noise_power == 0:
        return float('inf')
    return float(10 * np.log10(signal_power / noise_power))

def assess_fidelity(
    original_vectors: List[List[float]],
    reconstructed_vectors: List[List[float]],
    bits: int
) -> Dict:
    """Comprehensive fidelity assessment."""
    if len(original_vectors) != len(reconstructed_vectors):
        raise ValueError("Vector count mismatch")

    all_mse = []
    all_rmse = []
    all_max = []
    all_snr = []

    for orig, recon in zip(original_vectors, reconstructed_vectors):
        all_mse.append(compute_mse(orig, recon))
        all_rmse.append(compute_rmse(orig, recon))
        all_max.append(compute_max_error(orig, recon))
        all_snr.append(compute_snr(orig, recon))

    return {
        'bits_per_value': bits,
        'mse_mean': float(np.mean(all_mse)),
        'mse_std': float(np.std(all_mse)),
        'rmse_mean': float(np.mean(all_rmse)),
        'max_error_max': float(np.max(all_max)),
        'snr_mean_db': float(np.mean(all_snr)),
        'snr_min_db': float(np.min(all_snr)),
        'vectors_tested': len(original_vectors)
    }

if __name__ == "__main__":
    test_vectors = [
        [0.1, -0.3, 0.5, -0.7],
        [0.9, 0.8, -0.2, 0.1]
    ]

    for bits in [2, 4, 8]:
        reconstructed = []
        for vec in test_vectors:
            _, recon = uniform_symmetric_quantize(vec, bits)
            reconstructed.append(recon)

        metrics = assess_fidelity(test_vectors, reconstructed, bits)
        print(f"\n{bits}-bit quantization:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")