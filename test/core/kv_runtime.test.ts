import { quantizeValues, dequantizeValues, concatValueQuantized } from '../../src/core/value_quant.js';
import { CompressedKVCache } from '../../src/core/compressed_kv_cache.js';
import { computeHybridAttention } from '../../src/core/hybrid_score.js';

describe('value_quant', () => {
  test('quantize/dequantize roundtrip 2-bit', () => {
    const headDim = 64;
    const numTokens = 4;
    const values = new Float32Array(numTokens * headDim);
    for (let i = 0; i < values.length; i++) values[i] = Math.sin(i * 0.1);

    const vq = quantizeValues(values, numTokens, headDim, 2, 32);
    const restored = dequantizeValues(vq);

    expect(restored.length).toBe(values.length);
    // 2-bit quantization: expect reasonable MSE
    let mse = 0;
    for (let i = 0; i < values.length; i++) {
      mse += (values[i]! - restored[i]!) ** 2;
    }
    mse /= values.length;
    expect(mse).toBeLessThan(0.1);
  });

  test('quantize/dequantize roundtrip 4-bit', () => {
    const headDim = 64;
    const numTokens = 2;
    const values = new Float32Array(numTokens * headDim);
    for (let i = 0; i < values.length; i++) values[i] = Math.cos(i * 0.05);

    const vq = quantizeValues(values, numTokens, headDim, 4, 32);
    const restored = dequantizeValues(vq);

    let mse = 0;
    for (let i = 0; i < values.length; i++) {
      mse += (values[i]! - restored[i]!) ** 2;
    }
    mse /= values.length;
    expect(mse).toBeLessThan(0.01); // 4-bit should be more accurate
  });

  test('concat preserves data', () => {
    const headDim = 32;
    const a = quantizeValues(new Float32Array(2 * headDim).fill(0.5), 2, headDim, 2, 32);
    const b = quantizeValues(new Float32Array(3 * headDim).fill(-0.5), 3, headDim, 2, 32);
    const c = concatValueQuantized(a, b);
    expect(c.numTokens).toBe(5);
    expect(c.data.length).toBe(a.data.length + b.data.length);
  });
});

describe('CompressedKVCache', () => {
  const headDim = 64;

  function randomVec(d: number, seed: number): Float32Array {
    const v = new Float32Array(d);
    let s = seed;
    for (let i = 0; i < d; i++) {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      v[i] = (s / 0x7fffffff) * 2 - 1;
    }
    return v;
  }

  test('prefill within buffer size', () => {
    const cache = new CompressedKVCache({ headDim, bufferSize: 128 });
    const numTokens = 10;
    const keys = new Float32Array(numTokens * headDim);
    const values = new Float32Array(numTokens * headDim);
    for (let i = 0; i < keys.length; i++) {
      keys[i] = Math.sin(i);
      values[i] = Math.cos(i);
    }

    cache.prefill(keys, values, numTokens);
    expect(cache.getSeqLength()).toBe(numTokens);
    expect(cache.getKeyQuantized()).toBeNull();
    expect(cache.getBufferTokenCount()).toBe(numTokens);
  });

  test('prefill exceeding buffer triggers quantization', () => {
    const cache = new CompressedKVCache({ headDim, bufferSize: 4 });
    const numTokens = 10;
    const keys = new Float32Array(numTokens * headDim);
    const values = new Float32Array(numTokens * headDim);
    for (let i = 0; i < keys.length; i++) {
      keys[i] = Math.sin(i * 0.1);
      values[i] = Math.cos(i * 0.1);
    }

    cache.prefill(keys, values, numTokens);
    expect(cache.getSeqLength()).toBe(numTokens);
    expect(cache.getKeyQuantized()).not.toBeNull();
    expect(cache.getKeyQuantized()!.numTokens).toBe(6); // 10 - 4
    expect(cache.getBufferTokenCount()).toBe(4);
  });

  test('append flushes buffer when full', () => {
    const cache = new CompressedKVCache({ headDim, bufferSize: 4 });
    for (let t = 0; t < 6; t++) {
      cache.append(randomVec(headDim, t), randomVec(headDim, t + 100));
    }
    expect(cache.getSeqLength()).toBe(6);
    // After 6 appends with bufferSize=4, 2 should be flushed
    expect(cache.getKeyQuantized()).not.toBeNull();
    expect(cache.getKeyQuantized()!.numTokens).toBe(2);
    expect(cache.getBufferTokenCount()).toBe(4);
  });

  test('memoryBytes returns valid estimates', () => {
    const cache = new CompressedKVCache({ headDim, bufferSize: 4 });
    const numTokens = 10;
    const keys = new Float32Array(numTokens * headDim);
    const values = new Float32Array(numTokens * headDim);
    cache.prefill(keys, values, numTokens);

    const mem = cache.memoryBytes();
    expect(mem.total).toBeGreaterThan(0);
    expect(mem.quantizedKeys).toBeGreaterThan(0);
    expect(mem.buffer).toBeGreaterThan(0);
  });
});

describe('computeHybridAttention', () => {
  const headDim = 64;

  test('returns zero output for empty cache', () => {
    const cache = new CompressedKVCache({ headDim, bufferSize: 128 });
    const query = new Float32Array(headDim).fill(1 / Math.sqrt(headDim));
    const result = computeHybridAttention(query, cache);
    expect(result.totalTokens).toBe(0);
    expect(result.output.length).toBe(headDim);
  });

  test('attends over buffer-only cache', () => {
    const cache = new CompressedKVCache({ headDim, bufferSize: 128 });
    const numTokens = 5;
    const keys = new Float32Array(numTokens * headDim);
    const values = new Float32Array(numTokens * headDim);
    for (let i = 0; i < keys.length; i++) {
      keys[i] = Math.sin(i * 0.1);
      values[i] = Math.cos(i * 0.1);
    }
    cache.prefill(keys, values, numTokens);

    const query = new Float32Array(headDim);
    for (let i = 0; i < headDim; i++) query[i] = Math.sin(i * 0.1);

    const result = computeHybridAttention(query, cache);
    expect(result.totalTokens).toBe(numTokens);
    expect(result.bufferTokens).toBe(numTokens);
    expect(result.compressedTokens).toBe(0);
    expect(result.output.length).toBe(headDim);
  });

  test('attends over hybrid cache (compressed + buffer)', () => {
    const cache = new CompressedKVCache({ headDim, bufferSize: 4 });
    const numTokens = 20;
    const keys = new Float32Array(numTokens * headDim);
    const values = new Float32Array(numTokens * headDim);
    for (let i = 0; i < keys.length; i++) {
      keys[i] = Math.sin(i * 0.01);
      values[i] = Math.cos(i * 0.01);
    }
    cache.prefill(keys, values, numTokens);

    const query = new Float32Array(headDim).fill(0.1);
    const result = computeHybridAttention(query, cache);

    expect(result.totalTokens).toBe(numTokens);
    expect(result.compressedTokens).toBe(16);
    expect(result.bufferTokens).toBe(4);
    // Output should be a weighted average of values — non-zero
    let norm = 0;
    for (let i = 0; i < headDim; i++) norm += result.output[i]! ** 2;
    expect(Math.sqrt(norm)).toBeGreaterThan(0);
  });
});
