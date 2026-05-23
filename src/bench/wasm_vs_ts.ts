/**
 * Benchmark: WASM SIMD128 vs pure TS — FWHT + Lloyd-Max
 * Run: node --experimental-vm-modules dist/bench/wasm_vs_ts.js
 */
import { performance } from 'perf_hooks';
import { normalizedFwhtInPlace, isPowerOfTwo } from '../core/hadamard.js';
import { initWasmBackend, isWasmReady, wasmFwht, wasmDot } from '../native/wasm_backend.js';

const BENCH_LOG = process.env.TQ_BENCH_LOG === '1';

function fwhtPureTS(data: Float32Array): void {
  const n = data.length;
  let len = 1;
  while (len < n) {
    for (let i = 0; i < n; i += len * 2) {
      for (let j = 0; j < len; j++) {
        const u = data[i + j]!;
        const v = data[i + j + len]!;
        data[i + j] = u + v;
        data[i + j + len] = u - v;
      }
    }
    len *= 2;
  }
  const scale = 1 / Math.sqrt(n);
  for (let i = 0; i < n; i++) data[i]! *= scale;
}

function dotPureTS(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i]! * b[i]!;
  return sum;
}

function bench(name: string, fn: () => void, iters: number): number {
  // warmup
  for (let i = 0; i < 10; i++) fn();
  const start = performance.now();
  for (let i = 0; i < iters; i++) fn();
  const elapsed = performance.now() - start;
  const perIter = elapsed / iters;
  if (BENCH_LOG) console.log(`${name}: ${perIter.toFixed(3)}ms/iter (${iters} iters, ${elapsed.toFixed(1)}ms total)`);
  return perIter;
}

async function main() {
  const dims = [64, 128, 256];
  const iters = 1000;

  if (BENCH_LOG) console.log('=== TurboQuant WASM SIMD128 vs Pure TS Benchmark ===\n');

  const wasmOk = await initWasmBackend();
  if (BENCH_LOG) console.log(`WASM backend: ${wasmOk ? 'READY (SIMD128)' : 'UNAVAILABLE'}\n`);

  for (const dim of dims) {
    if (BENCH_LOG) console.log(`--- dim=${dim} ---`);
    const data = new Float32Array(dim);
    for (let i = 0; i < dim; i++) data[i] = Math.random() * 2 - 1;

    // FWHT benchmark
    const tsTime = bench(`  TS  FWHT(${dim})`, () => {
      const copy = new Float32Array(data);
      fwhtPureTS(copy);
    }, iters);

    let wasmTime = Infinity;
    if (wasmOk) {
      wasmTime = bench(`  WASM FWHT(${dim})`, () => {
        const copy = new Float32Array(data);
        wasmFwht(copy);
      }, iters);
      if (BENCH_LOG) console.log(`  Speedup: ${(tsTime / wasmTime).toFixed(2)}x`);
    }

    // Dot product benchmark
    const a = new Float32Array(dim);
    const b = new Float32Array(dim);
    for (let i = 0; i < dim; i++) { a[i] = Math.random(); b[i] = Math.random(); }

    const tsDotTime = bench(`  TS  dot(${dim})`, () => { dotPureTS(a, b); }, iters * 10);

    if (wasmOk) {
      const wasmDotTime = bench(`  WASM dot(${dim})`, () => { wasmDot(a, b); }, iters * 10);
      if (BENCH_LOG) console.log(`  Speedup: ${(tsDotTime / wasmDotTime).toFixed(2)}x`);
    }
    if (BENCH_LOG) console.log('');
  }
}

main().catch(console.error);
