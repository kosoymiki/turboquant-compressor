"""
TurboQuant Python reference — Beta PDF, Lloyd-Max codebook, rotation, QJL.
Paper-faithful implementation for cross-language validation against TS core.

SPDX-License-Identifier: GPL-3.0-or-later
"""

import math
import numpy as np
from typing import List, Tuple, TypedDict


class LloydMaxResult(TypedDict):
    centroids: List[float]
    boundaries: List[float]
    mse_per_coord: float
    mse_total: float
    dimension: int
    bits: int


# --- Beta PDF on sphere ---

def log_gamma(x: float) -> float:
    """Lanczos approximation to log(Gamma(x))."""
    if x <= 0:
        raise ValueError(f"log_gamma: x must be positive, got {x}")
    g = 7
    p = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    ]
    if x < 0.5:
        return math.log(math.pi / (math.sin(math.pi * x))) - log_gamma(1 - x)
    x -= 1
    a = p[0]
    for i in range(1, g + 2):
        a += p[i] / (x + i)
    t = x + g + 0.5
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def sphere_coordinate_beta_pdf(x: float, dimension: int) -> float:
    """PDF of a single coordinate of a uniform random point on S^{d-1}."""
    if dimension <= 2:
        raise ValueError(f"Dimension d={dimension} too small, need d >= 3")
    if x < -1 or x > 1:
        return 0.0
    log_const = log_gamma(dimension / 2) - 0.5 * math.log(math.pi) - log_gamma((dimension - 1) / 2)
    exponent = (dimension - 3) / 2
    log_val = log_const + exponent * math.log(max(1e-15, 1 - x * x))
    return math.exp(log_val)


# --- Adaptive Simpson integration ---

def integrate_adaptive_simpson(f, lo: float, hi: float, eps: float = 1e-10, max_depth: int = 50) -> float:
    """Adaptive Simpson's rule."""
    def simpson(a, b):
        c = (a + b) / 2
        return (b - a) / 6 * (f(a) + 4 * f(c) + f(b))

    def adaptive(a, b, whole, depth):
        c = (a + b) / 2
        left = simpson(a, c)
        right = simpson(c, b)
        delta = left + right - whole
        if depth >= max_depth or abs(delta) < 15 * eps:
            return left + right + delta / 15
        return adaptive(a, c, left, depth + 1) + adaptive(c, b, right, depth + 1)

    whole = simpson(lo, hi)
    return adaptive(lo, hi, whole, 0)


def conditional_mean_beta_sphere(lo: float, hi: float, dimension: int) -> float:
    """E[X | lo < X < hi] under the Beta PDF on [-1, 1]."""
    num = integrate_adaptive_simpson(lambda x: x * sphere_coordinate_beta_pdf(x, dimension), lo, hi)
    den = integrate_adaptive_simpson(lambda x: sphere_coordinate_beta_pdf(x, dimension), lo, hi)
    if den < 1e-30:
        return (lo + hi) / 2
    return num / den


# --- Lloyd-Max codebook ---

def compute_lloyd_max_beta_codebook(
    dimension: int,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-12
) -> LloydMaxResult:
    """
    Compute optimal Lloyd-Max codebook for Beta distribution on sphere.
    Returns dict with centroids, boundaries, mse_per_coord, mse_total.
    """
    n_clusters = 2 ** bits

    # Initialize centroids via quantile midpoints
    x_grid = np.linspace(-1, 1, 10000)
    pdf_vals = np.array([sphere_coordinate_beta_pdf(x, dimension) for x in x_grid])
    cdf_vals = np.cumsum(pdf_vals)
    cdf_vals /= cdf_vals[-1]

    centroids = []
    for i in range(n_clusters):
        q_mid = (i + 0.5) / n_clusters
        idx = np.searchsorted(cdf_vals, q_mid)
        centroids.append(float(x_grid[min(idx, len(x_grid) - 1)]))

    # Lloyd-Max iterations
    prev_cost = float('inf')
    cost = 0.0

    for _ in range(max_iter):
        # Boundaries = midpoints
        boundaries = [-1.0]
        for i in range(n_clusters - 1):
            boundaries.append((centroids[i] + centroids[i + 1]) / 2)
        boundaries.append(1.0)

        # Update centroids
        new_centroids = []
        for i in range(n_clusters):
            new_centroids.append(conditional_mean_beta_sphere(boundaries[i], boundaries[i + 1], dimension))

        # Compute MSE
        cost = 0.0
        for i in range(n_clusters):
            lo, hi, c = boundaries[i], boundaries[i + 1], new_centroids[i]
            cost += integrate_adaptive_simpson(
                lambda x, c=c: (x - c) ** 2 * sphere_coordinate_beta_pdf(x, dimension), lo, hi
            )

        centroids = new_centroids
        if abs(prev_cost - cost) < tol:
            break
        prev_cost = cost

    # Final boundaries
    final_boundaries = [-1.0]
    for i in range(n_clusters - 1):
        final_boundaries.append((centroids[i] + centroids[i + 1]) / 2)
    final_boundaries.append(1.0)

    return {
        'centroids': centroids,
        'boundaries': final_boundaries,
        'mse_per_coord': cost,
        'mse_total': cost * dimension,
        'dimension': dimension,
        'bits': bits,
    }


# --- Rotation (FWHT + sign flip) ---

class Mulberry32:
    """Deterministic PRNG matching TS implementation."""
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def next_u32(self) -> int:
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t ^ ((t ^ (t >> 7)) * (t | 61))) & 0xFFFFFFFF
        return (t ^ (t >> 14)) & 0xFFFFFFFF

    def random_float(self) -> float:
        return self.next_u32() / 0x100000000


def fwht_in_place(arr: np.ndarray) -> None:
    """Fast Walsh-Hadamard Transform in-place."""
    n = len(arr)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = arr[j]
                y = arr[j + h]
                arr[j] = x + y
                arr[j + h] = x - y
        h *= 2


def normalized_fwht_in_place(arr: np.ndarray) -> None:
    """Normalized FWHT (divide by sqrt(n))."""
    fwht_in_place(arr)
    arr /= math.sqrt(len(arr))


def next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def rotate_fwht_sign(vector: np.ndarray, seed: int = 0) -> np.ndarray:
    """Rotate vector using FWHT + random sign flips."""
    d = len(vector)
    padded_d = next_power_of_two(d)
    padded = np.zeros(padded_d)
    padded[:d] = vector

    # Generate sign flip pattern
    prng = Mulberry32(seed)
    signs = np.array([1.0 if prng.random_float() < 0.5 else -1.0 for _ in range(padded_d)])

    padded *= signs
    normalized_fwht_in_place(padded)
    return padded


def inverse_rotate_fwht_sign(rotated: np.ndarray, original_dim: int, seed: int = 0) -> np.ndarray:
    """Inverse rotation."""
    padded_d = len(rotated)
    temp = rotated.copy()
    normalized_fwht_in_place(temp)

    prng = Mulberry32(seed)
    signs = np.array([1.0 if prng.random_float() < 0.5 else -1.0 for _ in range(padded_d)])
    temp *= signs
    return temp[:original_dim]


# --- Quantizer ---

def quantize_vector(vector: np.ndarray, codebook: LloydMaxResult, seed: int = 0) -> Tuple[np.ndarray, float]:
    """
    Quantize a vector using TurboQuant algorithm:
    1. Normalize
    2. Rotate (FWHT + sign)
    3. Quantize each coordinate with Lloyd-Max codebook
    Returns (indices, norm).
    """
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        return np.zeros(len(vector), dtype=np.int32), 0.0

    normalized = vector / norm
    rotated = rotate_fwht_sign(normalized, seed)

    centroids = codebook['centroids']
    boundaries = codebook['boundaries']
    n_clusters = len(centroids)
    d = len(rotated)

    indices = np.zeros(d, dtype=np.int32)
    for i in range(d):
        x = rotated[i]
        # Binary search in boundaries
        idx = n_clusters - 1
        for j in range(n_clusters):
            if x < boundaries[j + 1]:
                idx = j
                break
        indices[i] = idx

    return indices, norm


def dequantize_vector(indices: np.ndarray, norm: float, codebook: LloydMaxResult, original_dim: int, seed: int = 0) -> np.ndarray:
    """Dequantize: look up centroids, inverse rotate, rescale."""
    centroids = codebook['centroids']
    d = len(indices)
    rotated = np.array([centroids[idx] for idx in indices])

    unrotated = inverse_rotate_fwht_sign(rotated, original_dim, seed)
    return unrotated * norm


# --- Cross-validation entry point ---

if __name__ == "__main__":
    print("Computing Lloyd-Max codebook for d=128, bits=4...")
    cb = compute_lloyd_max_beta_codebook(128, 4)
    print(f"  MSE per coord: {cb['mse_per_coord']:.8f}")
    print(f"  MSE total:     {cb['mse_total']:.8f}")
    print(f"  Centroids[0:4]: {cb['centroids'][:4]}")
    print(f"  Boundaries[0:5]: {cb['boundaries'][:5]}")

    # Test roundtrip
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(128).astype(np.float64)
    vec /= np.linalg.norm(vec)

    indices, norm = quantize_vector(vec, cb, seed=42)
    reconstructed = dequantize_vector(indices, norm, cb, 128, seed=42)

    mse = float(np.mean((vec - reconstructed) ** 2))
    print(f"\n  Roundtrip MSE: {mse:.8f}")
    print(f"  Cosine sim:    {float(np.dot(vec, reconstructed) / (np.linalg.norm(vec) * np.linalg.norm(reconstructed))):.6f}")
