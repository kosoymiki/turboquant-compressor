/**
 * Scientific regression: codebook MSE must be stable, finite, monotonic.
 */
import { computeLloydMaxBetaCodebook } from '../../src/core/beta_sphere.js';
import { computeCodebookMse, TurboQuantCodebookQuantizer } from '../../src/core/codebook.js';

describe('codebook MSE regression', () => {
  it('d=64 bits=2,3,4 all produce finite MSE', () => {
    for (const bits of [2, 3, 4] as const) {
      const cb = computeLloydMaxBetaCodebook(64, bits, { maxIter: 50 });
      expect(Number.isFinite(cb.msePerCoord)).toBe(true);
      expect(cb.msePerCoord).toBeGreaterThan(0);
      expect(Number.isFinite(cb.mseTotal)).toBe(true);
    }
  });

  it('MSE strictly decreases as bits increases (d=64)', () => {
    const mse2 = computeLloydMaxBetaCodebook(64, 2, { maxIter: 50 }).msePerCoord;
    const mse3 = computeLloydMaxBetaCodebook(64, 3, { maxIter: 50 }).msePerCoord;
    const mse4 = computeLloydMaxBetaCodebook(64, 4, { maxIter: 50 }).msePerCoord;
    expect(mse3).toBeLessThan(mse2);
    expect(mse4).toBeLessThan(mse3);
  });

  it('d=128 b=4 MSE stable across two runs (deterministic)', () => {
    const a = computeLloydMaxBetaCodebook(128, 4, { maxIter: 50 });
    const b = computeLloydMaxBetaCodebook(128, 4, { maxIter: 50 });
    expect(a.msePerCoord).toBe(b.msePerCoord);
    expect(a.centroids).toEqual(b.centroids);
  });

  it('computeCodebookMse matches msePerCoord from computation', () => {
    const cb = computeLloydMaxBetaCodebook(64, 3, { maxIter: 50 });
    const mse = computeCodebookMse(cb);
    // Should be very close (same integration method)
    expect(Math.abs(mse - cb.msePerCoord)).toBeLessThan(1e-10);
  });

  it('no NaN or Infinity in centroids/boundaries', () => {
    const cb = computeLloydMaxBetaCodebook(128, 4, { maxIter: 50 });
    for (const c of cb.centroids) {
      expect(Number.isFinite(c)).toBe(true);
      expect(Number.isNaN(c)).toBe(false);
    }
    for (const b of cb.boundaries) {
      expect(Number.isFinite(b)).toBe(true);
      expect(Number.isNaN(b)).toBe(false);
    }
  });

  it('boundaries strictly increasing', () => {
    const cb = computeLloydMaxBetaCodebook(64, 4, { maxIter: 50 });
    for (let i = 1; i < cb.boundaries.length; i++) {
      expect(cb.boundaries[i]!).toBeGreaterThan(cb.boundaries[i - 1]!);
    }
  });

  it('quantizer roundtrip preserves index', () => {
    const cb = computeLloydMaxBetaCodebook(64, 3, { maxIter: 50 });
    const q = new TurboQuantCodebookQuantizer(cb);
    for (let i = 0; i < cb.centroids.length; i++) {
      // Quantizing a centroid should return itself
      expect(q.quantize(cb.centroids[i]!)).toBe(cb.centroids[i]!);
    }
  });
});
