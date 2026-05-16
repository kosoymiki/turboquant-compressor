import { TurboQuantBetaCodebook, TurboQuantCodebookQuantizer, computeCodebookMse } from '../../src/core/codebook.js';
import { computeLloydMaxBetaCodebook } from '../../src/core/beta_sphere.js';

describe('TurboQuantCodebookQuantizer', () => {
  const DIM = 128;
  const BITS = 4 as const;

  test('quantize returns a centroid value', () => {
    const cb = computeLloydMaxBetaCodebook(DIM, BITS);
    const q = new TurboQuantCodebookQuantizer(cb);
    const result = q.quantize(0.0);
    expect(cb.centroids).toContain(result);
  });

  test('quantizeIndex returns valid index', () => {
    const cb = computeLloydMaxBetaCodebook(DIM, BITS);
    const q = new TurboQuantCodebookQuantizer(cb);
    const idx = q.quantizeIndex(0.0);
    expect(idx).toBeGreaterThanOrEqual(0);
    expect(idx).toBeLessThan(2 ** BITS);
  });

  test('dequantize(quantizeIndex(x)) == quantize(x)', () => {
    const cb = computeLloydMaxBetaCodebook(DIM, BITS);
    const q = new TurboQuantCodebookQuantizer(cb);
    const x = 0.3;
    expect(q.dequantize(q.quantizeIndex(x))).toBe(q.quantize(x));
  });

  test('boundary values are handled correctly', () => {
    const cb = computeLloydMaxBetaCodebook(DIM, BITS);
    const q = new TurboQuantCodebookQuantizer(cb);
    // Extreme values should map to first/last centroid
    const lo = q.quantize(-1.0);
    const hi = q.quantize(0.9999);
    expect(lo).toBe(cb.centroids[0]);
    expect(hi).toBe(cb.centroids[cb.centroids.length - 1]);
  });

  test('throws on invalid codebook shape', () => {
    expect(() => new TurboQuantCodebookQuantizer({
      centroids: [0.1, 0.2],
      boundaries: [-1, 0, 1],
      msePerCoord: 0,
      mseTotal: 0,
      dimension: 128,
      bits: 4, // expects 16 centroids for 4 bits
      algorithm: 'lloyd-max-beta',
      source: 'computed'
    })).toThrow(/Invariant violation/);
  });

  test('dequantize throws on out-of-bounds index', () => {
    const cb = computeLloydMaxBetaCodebook(DIM, BITS);
    const q = new TurboQuantCodebookQuantizer(cb);
    expect(() => q.dequantize(99)).toThrow(/Invariant violation/);
    expect(() => q.dequantize(-1)).toThrow(/Invariant violation/);
  });
});

describe('computeCodebookMse', () => {
  test('returns non-negative MSE', () => {
    const cb = computeLloydMaxBetaCodebook(128, 4);
    const mse = computeCodebookMse(cb);
    expect(mse).toBeGreaterThanOrEqual(0);
    expect(mse).toBeLessThan(1); // should be small for 4-bit
  });

  test('higher bits gives lower MSE', () => {
    const mse2 = computeCodebookMse(computeLloydMaxBetaCodebook(128, 2));
    const mse4 = computeCodebookMse(computeLloydMaxBetaCodebook(128, 4));
    expect(mse4).toBeLessThan(mse2);
  });
});

describe('TurboQuantBetaCodebook class', () => {
  test('compute + quantize + dequantize roundtrip', () => {
    const beta = new TurboQuantBetaCodebook(128, 4);
    beta.compute();
    const val = 0.1;
    const idx = beta.quantizeIndex(val);
    const reconstructed = beta.dequantize(idx);
    expect(reconstructed).toBe(beta.quantize(val));
  });

  test('getCodebook returns valid codebook after compute', () => {
    const beta = new TurboQuantBetaCodebook(64, 3);
    beta.compute();
    const cb = beta.getCodebook();
    expect(cb).not.toBeNull();
    expect(cb!.algorithm).toBe('lloyd-max-beta');
    expect(cb!.source).toBe('computed');
    expect(cb!.centroids.length).toBe(8);
    expect(cb!.boundaries.length).toBe(9);
  });

  test('throws if quantize called before compute', () => {
    const beta = new TurboQuantBetaCodebook(128, 4);
    expect(() => beta.quantize(0.5)).toThrow();
  });
});
