import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

function runScenario(script: string): unknown {
  const stdout = execFileSync('node', ['--input-type=module', '-e', script], {
    cwd: ROOT,
    encoding: 'utf8',
    timeout: 10000,
  });
  return JSON.parse(stdout);
}

describe('kv runtime', () => {
  test('value quantization roundtrips stay within expected error', () => {
    const result = runScenario(`
      import { quantizeValues, dequantizeValues, concatValueQuantized } from './dist/core/value_quant.js';
      const headDim = 64;
      const values2 = new Float32Array(4 * headDim);
      for (let i = 0; i < values2.length; i++) values2[i] = Math.sin(i * 0.1);
      const q2 = quantizeValues(values2, 4, headDim, 2, 32);
      const r2 = dequantizeValues(q2);
      let mse2 = 0;
      for (let i = 0; i < values2.length; i++) mse2 += (values2[i] - r2[i]) ** 2;
      mse2 /= values2.length;

      const values4 = new Float32Array(2 * headDim);
      for (let i = 0; i < values4.length; i++) values4[i] = Math.cos(i * 0.05);
      const q4 = quantizeValues(values4, 2, headDim, 4, 32);
      const r4 = dequantizeValues(q4);
      let mse4 = 0;
      for (let i = 0; i < values4.length; i++) mse4 += (values4[i] - r4[i]) ** 2;
      mse4 /= values4.length;

      const a = quantizeValues(new Float32Array(2 * 32).fill(0.5), 2, 32, 2, 32);
      const b = quantizeValues(new Float32Array(3 * 32).fill(-0.5), 3, 32, 2, 32);
      const c = concatValueQuantized(a, b);

      console.log(JSON.stringify({ mse2, mse4, concatTokens: c.numTokens, concatBytes: c.data.length }));
    `) as { mse2: number; mse4: number; concatTokens: number; concatBytes: number };

    expect(result.mse2).toBeLessThan(0.1);
    expect(result.mse4).toBeLessThan(0.01);
    expect(result.concatTokens).toBe(5);
    expect(result.concatBytes).toBeGreaterThan(0);
  });

  test('compressed kv cache prefill and append paths stay coherent', () => {
    const result = runScenario(`
      import { CompressedKVCache } from './dist/core/compressed_kv_cache.js';
      function randomVec(d, seed) {
        const v = new Float32Array(d);
        let s = seed;
        for (let i = 0; i < d; i++) {
          s = (s * 1103515245 + 12345) & 0x7fffffff;
          v[i] = (s / 0x7fffffff) * 2 - 1;
        }
        return v;
      }

      const headDim = 64;
      const cacheA = new CompressedKVCache({ headDim, bufferSize: 128 });
      const keysA = new Float32Array(10 * headDim);
      const valsA = new Float32Array(10 * headDim);
      for (let i = 0; i < keysA.length; i++) {
        keysA[i] = Math.sin(i);
        valsA[i] = Math.cos(i);
      }
      cacheA.prefill(keysA, valsA, 10);

      const cacheB = new CompressedKVCache({ headDim, bufferSize: 4 });
      const keysB = new Float32Array(10 * headDim);
      const valsB = new Float32Array(10 * headDim);
      for (let i = 0; i < keysB.length; i++) {
        keysB[i] = Math.sin(i * 0.1);
        valsB[i] = Math.cos(i * 0.1);
      }
      cacheB.prefill(keysB, valsB, 10);

      const cacheC = new CompressedKVCache({ headDim, bufferSize: 4 });
      for (let t = 0; t < 6; t++) {
        cacheC.append(randomVec(headDim, t), randomVec(headDim, t + 100));
      }

      console.log(JSON.stringify({
        prefillBufferOnly: {
          seqLen: cacheA.getSeqLength(),
          hasQuant: cacheA.getKeyQuantized() !== null,
          bufferTokens: cacheA.getBufferTokenCount(),
        },
        prefillHybrid: {
          seqLen: cacheB.getSeqLength(),
          quantTokens: cacheB.getKeyQuantized()?.numTokens ?? 0,
          bufferTokens: cacheB.getBufferTokenCount(),
          memTotal: cacheB.memoryBytes().total,
        },
        appendHybrid: {
          seqLen: cacheC.getSeqLength(),
          quantTokens: cacheC.getKeyQuantized()?.numTokens ?? 0,
          bufferTokens: cacheC.getBufferTokenCount(),
        }
      }));
    `) as {
      prefillBufferOnly: { seqLen: number; hasQuant: boolean; bufferTokens: number };
      prefillHybrid: { seqLen: number; quantTokens: number; bufferTokens: number; memTotal: number };
      appendHybrid: { seqLen: number; quantTokens: number; bufferTokens: number };
    };

    expect(result.prefillBufferOnly.seqLen).toBe(10);
    expect(result.prefillBufferOnly.hasQuant).toBe(false);
    expect(result.prefillBufferOnly.bufferTokens).toBe(10);
    expect(result.prefillHybrid.seqLen).toBe(10);
    expect(result.prefillHybrid.quantTokens).toBe(6);
    expect(result.prefillHybrid.bufferTokens).toBe(4);
    expect(result.prefillHybrid.memTotal).toBeGreaterThan(0);
    expect(result.appendHybrid.seqLen).toBe(6);
    expect(result.appendHybrid.quantTokens).toBe(2);
    expect(result.appendHybrid.bufferTokens).toBe(4);
  });

  test('hybrid attention returns stable non-zero output', () => {
    const result = runScenario(`
      import { CompressedKVCache } from './dist/core/compressed_kv_cache.js';
      import { computeHybridAttention } from './dist/core/hybrid_score.js';

      const headDim = 64;
      const emptyCache = new CompressedKVCache({ headDim, bufferSize: 128 });
      const emptyQuery = new Float32Array(headDim).fill(1 / Math.sqrt(headDim));
      const emptyResult = computeHybridAttention(emptyQuery, emptyCache);

      const bufferCache = new CompressedKVCache({ headDim, bufferSize: 128 });
      const numTokensA = 5;
      const keysA = new Float32Array(numTokensA * headDim);
      const valsA = new Float32Array(numTokensA * headDim);
      for (let i = 0; i < keysA.length; i++) {
        keysA[i] = Math.sin(i * 0.1);
        valsA[i] = Math.cos(i * 0.1);
      }
      bufferCache.prefill(keysA, valsA, numTokensA);
      const queryA = new Float32Array(headDim);
      for (let i = 0; i < headDim; i++) queryA[i] = Math.sin(i * 0.1);
      const bufferResult = computeHybridAttention(queryA, bufferCache);

      const hybridCache = new CompressedKVCache({ headDim, bufferSize: 4 });
      const numTokensB = 20;
      const keysB = new Float32Array(numTokensB * headDim);
      const valsB = new Float32Array(numTokensB * headDim);
      for (let i = 0; i < keysB.length; i++) {
        keysB[i] = Math.sin(i * 0.01);
        valsB[i] = Math.cos(i * 0.01);
      }
      hybridCache.prefill(keysB, valsB, numTokensB);
      const queryB = new Float32Array(headDim).fill(0.1);
      const hybridResult = computeHybridAttention(queryB, hybridCache);
      let norm = 0;
      for (let i = 0; i < headDim; i++) norm += hybridResult.output[i] ** 2;

      console.log(JSON.stringify({
        empty: { totalTokens: emptyResult.totalTokens, outputLength: emptyResult.output.length },
        bufferOnly: {
          totalTokens: bufferResult.totalTokens,
          compressedTokens: bufferResult.compressedTokens,
          bufferTokens: bufferResult.bufferTokens,
          outputLength: bufferResult.output.length,
        },
        hybrid: {
          totalTokens: hybridResult.totalTokens,
          compressedTokens: hybridResult.compressedTokens,
          bufferTokens: hybridResult.bufferTokens,
          norm: Math.sqrt(norm),
        }
      }));
    `) as {
      empty: { totalTokens: number; outputLength: number };
      bufferOnly: { totalTokens: number; compressedTokens: number; bufferTokens: number; outputLength: number };
      hybrid: { totalTokens: number; compressedTokens: number; bufferTokens: number; norm: number };
    };

    expect(result.empty.totalTokens).toBe(0);
    expect(result.empty.outputLength).toBe(64);
    expect(result.bufferOnly.totalTokens).toBe(5);
    expect(result.bufferOnly.compressedTokens).toBe(0);
    expect(result.bufferOnly.bufferTokens).toBe(5);
    expect(result.bufferOnly.outputLength).toBe(64);
    expect(result.hybrid.totalTokens).toBe(20);
    expect(result.hybrid.compressedTokens).toBe(16);
    expect(result.hybrid.bufferTokens).toBe(4);
    expect(result.hybrid.norm).toBeGreaterThan(0);
  });
});
