import {
  sphereCoordinateBetaPdf,
  integrateAdaptiveSimpson,
  computeLloydMaxBetaCodebook
} from './beta_sphere.js';

describe('sphereCoordinateBetaPdf', () => {
  it('integrates approximately to 1', () => {
    const integral = integrateAdaptiveSimpson(
      (x) => sphereCoordinateBetaPdf(x, 64),
      -1, 1, { eps: 1e-8, maxDepth: 40 }
    );
    expect(integral).toBeGreaterThan(0.99);
    expect(integral).toBeLessThan(1.01);
  });

  it('has variance approximately 1/dimension', () => {
    const dimension = 64;
    const secondMoment = integrateAdaptiveSimpson(
      (x) => x * x * sphereCoordinateBetaPdf(x, dimension),
      -1, 1, { eps: 1e-8, maxDepth: 40 }
    );
    // E[X^2] ≈ 1/d for sphere coordinates
    expect(secondMoment).toBeGreaterThan(0.012);
    expect(secondMoment).toBeLessThan(0.020);
  });
});

describe('computeLloydMaxBetaCodebook', () => {
  it('produces sorted centroids and strictly increasing boundaries', () => {
    const cb = computeLloydMaxBetaCodebook(64, 3, { maxIter: 25, tol: 1e-8 });

    for (let i = 1; i < cb.centroids.length; i++) {
      expect(cb.centroids[i]!).toBeGreaterThanOrEqual(cb.centroids[i - 1]!);
    }
    for (let i = 1; i < cb.boundaries.length; i++) {
      expect(cb.boundaries[i]!).toBeGreaterThan(cb.boundaries[i - 1]!);
    }
  });

  it('MSE decreases as bit width increases', () => {
    const b2 = computeLloydMaxBetaCodebook(64, 2, { maxIter: 25, tol: 1e-8 });
    const b3 = computeLloydMaxBetaCodebook(64, 3, { maxIter: 25, tol: 1e-8 });
    expect(b3.mseTotal).toBeLessThan(b2.mseTotal);
  });

  it('msePerCoord is finite and positive after cost-reset fix', () => {
    const cb = computeLloydMaxBetaCodebook(128, 4, { maxIter: 50 });
    expect(Number.isFinite(cb.msePerCoord)).toBe(true);
    expect(cb.msePerCoord).toBeGreaterThan(0);
    // With correct cost reset, MSE should be small for 4-bit
    expect(cb.msePerCoord).toBeLessThan(0.001);
  });
});
