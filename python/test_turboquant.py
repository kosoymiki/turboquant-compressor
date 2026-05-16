"""
TurboQuant Python reference test suite.
Covers: Beta PDF, Lloyd-Max codebook, rotation roundtrip, quantizer, fidelity oracle.

Run: python -m pytest test_turboquant.py -v
"""

import math
import numpy as np
import pytest

from turboquant_reference import (
    sphere_coordinate_beta_pdf,
    integrate_adaptive_simpson,
    compute_lloyd_max_beta_codebook,
    rotate_fwht_sign,
    inverse_rotate_fwht_sign,
    quantize_vector,
    dequantize_vector,
)
from reference_quantizer import uniform_symmetric_quantize
from fidelity_oracle import compute_mse, compute_snr, assess_fidelity


# --- Beta PDF ---

class TestBetaPDF:
    def test_pdf_integrates_to_one(self):
        """PDF over [-1, 1] should integrate to ~1.0."""
        for d in [16, 64, 128]:
            integral = integrate_adaptive_simpson(
                lambda x: sphere_coordinate_beta_pdf(x, d), -1, 1
            )
            assert abs(integral - 1.0) < 1e-6, f"d={d}: integral={integral}"

    def test_pdf_symmetric(self):
        """PDF should be symmetric around 0."""
        for d in [32, 128]:
            assert abs(
                sphere_coordinate_beta_pdf(0.3, d) - sphere_coordinate_beta_pdf(-0.3, d)
            ) < 1e-12

    def test_pdf_zero_outside_range(self):
        assert sphere_coordinate_beta_pdf(1.5, 64) == 0.0
        assert sphere_coordinate_beta_pdf(-1.1, 64) == 0.0

    def test_pdf_raises_low_dimension(self):
        with pytest.raises(ValueError):
            sphere_coordinate_beta_pdf(0.0, 2)

    def test_pdf_concentration_increases_with_dimension(self):
        """Higher d → more concentrated around 0."""
        val_low_d = sphere_coordinate_beta_pdf(0.0, 16)
        val_high_d = sphere_coordinate_beta_pdf(0.0, 256)
        assert val_high_d > val_low_d


# --- Lloyd-Max Codebook ---

class TestLloydMax:
    def test_codebook_returns_correct_structure(self):
        cb = compute_lloyd_max_beta_codebook(64, 2, max_iter=50)
        assert len(cb['centroids']) == 4
        assert len(cb['boundaries']) == 5
        assert cb['boundaries'][0] == -1.0
        assert cb['boundaries'][-1] == 1.0

    def test_codebook_centroids_sorted(self):
        cb = compute_lloyd_max_beta_codebook(128, 3, max_iter=100)
        centroids = cb['centroids']
        for i in range(len(centroids) - 1):
            assert centroids[i] < centroids[i + 1]

    def test_codebook_mse_finite_positive(self):
        cb = compute_lloyd_max_beta_codebook(128, 4, max_iter=100)
        assert cb['mse_per_coord'] > 0
        assert math.isfinite(cb['mse_per_coord'])

    def test_codebook_mse_monotonic_with_bits(self):
        """More bits → lower MSE."""
        mse_2 = compute_lloyd_max_beta_codebook(64, 2, max_iter=100)['mse_per_coord']
        mse_3 = compute_lloyd_max_beta_codebook(64, 3, max_iter=100)['mse_per_coord']
        mse_4 = compute_lloyd_max_beta_codebook(64, 4, max_iter=100)['mse_per_coord']
        assert mse_2 > mse_3 > mse_4

    def test_codebook_cost_not_accumulated_bug(self):
        """Regression: cost must reset each iteration, not accumulate."""
        cb = compute_lloyd_max_beta_codebook(128, 3, max_iter=200)
        # If cost accumulated, mse_per_coord would be unreasonably large
        assert cb['mse_per_coord'] < 0.1

    def test_codebook_deterministic(self):
        cb1 = compute_lloyd_max_beta_codebook(64, 3, max_iter=100)
        cb2 = compute_lloyd_max_beta_codebook(64, 3, max_iter=100)
        assert cb1['centroids'] == cb2['centroids']
        assert cb1['mse_per_coord'] == cb2['mse_per_coord']


# --- Rotation ---

class TestRotation:
    def test_rotation_preserves_norm(self):
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(64)
        rotated = rotate_fwht_sign(vec, seed=7)
        # Padded to next power of 2, so norm of padded input = norm of original
        assert abs(np.linalg.norm(rotated) - np.linalg.norm(vec)) < 1e-10

    def test_rotation_roundtrip(self):
        rng = np.random.default_rng(99)
        vec = rng.standard_normal(128)
        rotated = rotate_fwht_sign(vec, seed=42)
        recovered = inverse_rotate_fwht_sign(rotated, 128, seed=42)
        np.testing.assert_allclose(recovered, vec, atol=1e-10)

    def test_rotation_deterministic(self):
        vec = np.ones(32)
        r1 = rotate_fwht_sign(vec, seed=123)
        r2 = rotate_fwht_sign(vec, seed=123)
        np.testing.assert_array_equal(r1, r2)

    def test_rotation_different_seeds_differ(self):
        vec = np.ones(64)
        r1 = rotate_fwht_sign(vec, seed=1)
        r2 = rotate_fwht_sign(vec, seed=2)
        assert not np.allclose(r1, r2)


# --- Quantizer roundtrip ---

class TestQuantizer:
    @pytest.fixture
    def codebook_128_3(self):
        return compute_lloyd_max_beta_codebook(128, 3, max_iter=100)

    def test_quantize_returns_valid_indices(self, codebook_128_3):
        rng = np.random.default_rng(7)
        vec = rng.standard_normal(128)
        indices, norm = quantize_vector(vec, codebook_128_3, seed=0)
        assert indices.min() >= 0
        assert indices.max() < 8  # 2^3
        assert norm > 0

    def test_dequantize_finite(self, codebook_128_3):
        rng = np.random.default_rng(7)
        vec = rng.standard_normal(128)
        indices, norm = quantize_vector(vec, codebook_128_3, seed=0)
        recon = dequantize_vector(indices, norm, codebook_128_3, 128, seed=0)
        assert np.all(np.isfinite(recon))

    def test_roundtrip_cosine_similarity(self, codebook_128_3):
        """Roundtrip should preserve direction reasonably well."""
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(128)
        vec /= np.linalg.norm(vec)
        indices, norm = quantize_vector(vec, codebook_128_3, seed=5)
        recon = dequantize_vector(indices, norm, codebook_128_3, 128, seed=5)
        cos_sim = np.dot(vec, recon) / (np.linalg.norm(vec) * np.linalg.norm(recon))
        assert cos_sim > 0.7  # 3-bit should preserve direction

    def test_zero_vector_handled(self, codebook_128_3):
        vec = np.zeros(128)
        indices, norm = quantize_vector(vec, codebook_128_3, seed=0)
        assert norm == 0.0
        assert np.all(indices == 0)


# --- Reference quantizer ---

class TestReferenceQuantizer:
    def test_uniform_symmetric_4bit(self):
        values = [0.5, -0.5, 0.0]
        indices, recon = uniform_symmetric_quantize(values, 4)
        assert len(indices) == 3
        assert len(recon) == 3
        assert all(0 <= i < 16 for i in indices)

    def test_uniform_symmetric_raises_invalid_bits(self):
        with pytest.raises(ValueError):
            uniform_symmetric_quantize([1.0], 5)

    def test_zero_vector(self):
        indices, recon = uniform_symmetric_quantize([0.0, 0.0], 4)
        assert indices == [0, 0]
        assert recon == [0.0, 0.0]


# --- Fidelity oracle ---

class TestFidelityOracle:
    def test_mse_zero_for_identical(self):
        assert compute_mse([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_snr_inf_for_perfect(self):
        assert compute_snr([1.0, 2.0], [1.0, 2.0]) == float('inf')

    def test_assess_fidelity_structure(self):
        orig = [[0.1, -0.3, 0.5]]
        recon = [[0.11, -0.29, 0.49]]
        result = assess_fidelity(orig, recon, 4)
        assert 'mse_mean' in result
        assert 'snr_mean_db' in result
        assert result['vectors_tested'] == 1

    def test_assess_fidelity_raises_mismatch(self):
        with pytest.raises(ValueError):
            assess_fidelity([[1.0]], [[1.0], [2.0]], 4)
