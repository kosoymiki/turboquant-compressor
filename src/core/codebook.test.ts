import { LloydMaxCodebook, BetaCodebook } from './codebook.js';
import { deterministicFloatArray } from './test-utils.js';

describe('LloydMaxCodebook', () => {
  test('train creates valid codebook', () => {
    const data = deterministicFloatArray(100, 101, -1, 1);

    const codebook = new LloydMaxCodebook(4);
    const result = codebook.train(data);

    expect(result.bitsPerValue).toBe(4);
    expect(result.entries.length).toBe(16);
    expect(result.mse).toBeGreaterThanOrEqual(0);
    expect(result.iterations).toBeGreaterThan(0);
  });

  test('encode and decode roundtrip', () => {
    const data = deterministicFloatArray(50, 102, -1, 1);

    const codebook = new LloydMaxCodebook(4);
    codebook.train(data);

    const encoded = codebook.encode(data);
    const decoded = codebook.decode(encoded);

    expect(decoded.length).toBe(data.length);
  });

  test('throws when encoding without training', () => {
    const codebook = new LloydMaxCodebook(4);
    const data = new Float32Array([0.5, 0.3, 0.7]);

    expect(() => codebook.encode(data)).toThrow();
  });

  test('throws when decoding without training', () => {
    const codebook = new LloydMaxCodebook(4);
    const packed = new Uint8Array([128]);

    expect(() => codebook.decode(packed)).toThrow();
  });

  test('different bits per value', () => {
    const data = deterministicFloatArray(50, 103, -1, 1);

    const codebook2 = new LloydMaxCodebook(2);
    const result2 = codebook2.train(data);
    expect(result2.entries.length).toBe(4);

    const codebook8 = new LloydMaxCodebook(8);
    const result8 = codebook8.train(data);
    expect(result8.entries.length).toBe(256);
  });

  test('getCodebook returns trained codebook', () => {
    const data = deterministicFloatArray(30, 104, -1, 1);

    const codebook = new LloydMaxCodebook(4);
    expect(codebook.getCodebook()).toBeNull();

    codebook.train(data);
    expect(codebook.getCodebook()).not.toBeNull();
  });

  test('trains distinct centroids on clustered data', () => {
    // Deterministic clustered data - two clear clusters
    const data = new Float32Array([-1, -0.95, -0.9, 0.9, 0.95, 1]);
    const cb = new LloydMaxCodebook(2);
    const model = cb.train(data, 20);

    expect(model.mse).toBeGreaterThanOrEqual(0);
    expect(model.entries.some(e => e.count > 0)).toBe(true);
    expect(new Set(model.entries.map(e => e.reconstruction.toFixed(3))).size).toBeGreaterThan(1);
  });
});

describe('BetaCodebook', () => {
  test('train creates valid codebook', () => {
    const data = deterministicFloatArray(100, 201, 0, 1);

    const codebook = new BetaCodebook(4);
    const result = codebook.train(data);

    expect(result.bitsPerValue).toBe(4);
    expect(result.entries.length).toBe(16);
    expect(result.mse).toBeGreaterThanOrEqual(0);
  });

  test('encode and decode roundtrip', () => {
    const data = deterministicFloatArray(50, 202, 0, 1);

    const codebook = new BetaCodebook(4);
    codebook.train(data);

    const encoded = codebook.encode(data);
    const decoded = codebook.decode(encoded);

    expect(decoded.length).toBe(data.length);
  });

  test('handles skewed data', () => {
    const data = new Float32Array(100);
    for (let i = 0; i < 100; i++) {
      // Skewed towards lower values
      const val = deterministicFloatArray(1, 203 + i, 0, 1)[0]!;
      data[i] = val * val;
    }

    const codebook = new BetaCodebook(4);
    const result = codebook.train(data);

    expect(result.mse).toBeGreaterThanOrEqual(0);
  });

  test('throws when encoding without training', () => {
    const codebook = new BetaCodebook(4);
    const data = new Float32Array([0.5, 0.3, 0.7]);

    expect(() => codebook.encode(data)).toThrow();
  });

  test('throws when decoding without training', () => {
    const codebook = new BetaCodebook(4);
    const packed = new Uint8Array([128]);

    expect(() => codebook.decode(packed)).toThrow();
  });

  test('getParameters returns Beta parameters', () => {
    const data = deterministicFloatArray(50, 204, 0, 1);

    const codebook = new BetaCodebook(4);
    expect(codebook.getParameters()).toBeNull();

    codebook.train(data);
    const params = codebook.getParameters();

    expect(params).not.toBeNull();
    expect(params!.alpha).toBeGreaterThan(0);
    expect(params!.beta).toBeGreaterThan(0);
  });

  test('getCodebook returns trained codebook', () => {
    const data = deterministicFloatArray(30, 205, 0, 1);

    const codebook = new BetaCodebook(4);
    expect(codebook.getCodebook()).toBeNull();

    codebook.train(data);
    expect(codebook.getCodebook()).not.toBeNull();
  });

  test('remains research-only and finite', () => {
    const data = new Float32Array([0.01, 0.1, 0.2, 0.8, 0.9, 0.99]);
    const cb = new BetaCodebook(4);
    const model = cb.train(data);
    expect(Number.isFinite(model.mse)).toBe(true);
    for (const e of model.entries) {
      expect(Number.isFinite(e.reconstruction)).toBe(true);
    }
  });
});