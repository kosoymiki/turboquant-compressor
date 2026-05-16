"""
Reference quantizer implementation for validation.
Implements uniform symmetric scalar quantization.
"""

import numpy as np
from typing import List, Tuple

def uniform_symmetric_quantize(values: List[float], bits: int) -> Tuple[List[int], List[float]]:
    """Quantize values using uniform symmetric codebook."""
    if bits not in [2, 3, 4, 8]:
        raise ValueError(f"Unsupported bits per value: {bits}")

    num_levels = 2 ** bits
    half_levels = num_levels // 2

    values_arr = np.array(values)
    max_val = np.max(np.abs(values_arr))

    if max_val == 0:
        return [0] * len(values), [0.0] * len(values)

    scale = (half_levels - 0.5) / max_val
    scaled = values_arr * scale

    indices = np.floor(scaled + half_levels).astype(int)
    indices = np.clip(indices, 0, num_levels - 1)

    reconstruction = (indices - half_levels + 0.5) / scale
    reconstruction = np.where(np.abs(values_arr) < 1e-10, 0.0, reconstruction)

    return indices.tolist(), reconstruction.tolist()

def uniform_symmetric_dequantize(indices: List[int], bits: int, max_val: float) -> List[float]:
    """Dequantize indices back to values."""
    num_levels = 2 ** bits
    half_levels = num_levels // 2

    indices_arr = np.array(indices)
    scale = (half_levels - 0.5) / max_val if max_val > 0 else 1.0

    reconstruction = (indices_arr - half_levels + 0.5) / scale
    return reconstruction.tolist()

if __name__ == "__main__":
    test_values = [0.1, -0.3, 0.5, -0.7, 0.9]
    indices, reconstructed = uniform_symmetric_quantize(test_values, 4)
    print(f"Original: {test_values}")
    print(f"Indices: {indices}")
    print(f"Reconstructed: {reconstructed}")