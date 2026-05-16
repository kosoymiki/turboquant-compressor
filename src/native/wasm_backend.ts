/**
 * TurboQuant WASM backend integration for MCP server.
 * Offloads CPU-bound FWHT/Lloyd-Max/QJL to SIMD128 WASM core (37KB).
 * Falls back to pure TS if WASM unavailable.
 */
import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WASM_PATH = join(__dirname, '../../native/wasm/pkg/turboquant_wasm.wasm');

let wasmReady = false;
let wasmInstance: WebAssembly.Instance | null = null;
let wasmMemory: WebAssembly.Memory | null = null;

export async function initWasmBackend(): Promise<boolean> {
  if (wasmReady) return true;
  try {
    const buf = readFileSync(WASM_PATH);
    const mod = await WebAssembly.compile(buf);
    const mem = new WebAssembly.Memory({ initial: 64, maximum: 256 });
    const inst = await WebAssembly.instantiate(mod, { env: { memory: mem } });
    wasmInstance = inst;
    wasmMemory = (inst.exports.memory as WebAssembly.Memory) || mem;
    wasmReady = true;
    return true;
  } catch {
    wasmReady = false;
    return false;
  }
}

export function isWasmReady(): boolean {
  return wasmReady;
}

function getExports(): any {
  return wasmInstance?.exports;
}

/**
 * FWHT via WASM SIMD128
 * Returns null if WASM unavailable (caller falls back to TS)
 */
export function wasmFwht(data: Float32Array): Float32Array | null {
  if (!wasmReady) return null;
  const exports = getExports();
  if (!exports?.fwht) return null;

  const bytes = data.byteLength;
  const ptr = exports.__wbindgen_malloc(bytes, 4);
  const view = new Float32Array(wasmMemory!.buffer, ptr, data.length);
  view.set(data);
  exports.fwht(ptr, data.length);
  const result = new Float32Array(wasmMemory!.buffer, ptr, data.length).slice();
  exports.__wbindgen_free(ptr, bytes, 4);
  return result;
}

/**
 * SIMD dot product via WASM
 */
export function wasmDot(a: Float32Array, b: Float32Array): number | null {
  if (!wasmReady) return null;
  const exports = getExports();
  if (!exports?.simd_dot) return null;

  const aBytes = a.byteLength;
  const bBytes = b.byteLength;
  const ptrA = exports.__wbindgen_malloc(aBytes, 4);
  const ptrB = exports.__wbindgen_malloc(bBytes, 4);
  new Float32Array(wasmMemory!.buffer, ptrA, a.length).set(a);
  new Float32Array(wasmMemory!.buffer, ptrB, b.length).set(b);
  const result = exports.simd_dot(ptrA, a.length, ptrB, b.length);
  exports.__wbindgen_free(ptrA, aBytes, 4);
  exports.__wbindgen_free(ptrB, bBytes, 4);
  return result;
}

/**
 * Lloyd-Max quantize via WASM
 */
export function wasmLloydMaxQuantize(values: Float32Array, centroids: Float32Array, bits: number): Uint8Array | null {
  if (!wasmReady) return null;
  const exports = getExports();
  if (!exports?.lloyd_max_quantize) return null;

  const vBytes = values.byteLength;
  const cBytes = centroids.byteLength;
  const ptrV = exports.__wbindgen_malloc(vBytes, 4);
  const ptrC = exports.__wbindgen_malloc(cBytes, 4);
  new Float32Array(wasmMemory!.buffer, ptrV, values.length).set(values);
  new Float32Array(wasmMemory!.buffer, ptrC, centroids.length).set(centroids);

  const resultPtr = exports.lloyd_max_quantize(ptrV, values.length, ptrC, centroids.length, bits);
  const valsPerByte = Math.floor(8 / bits);
  const packedLen = Math.ceil(values.length / valsPerByte);
  const packed = new Uint8Array(wasmMemory!.buffer, resultPtr, packedLen).slice();

  exports.__wbindgen_free(ptrV, vBytes, 4);
  exports.__wbindgen_free(ptrC, cBytes, 4);
  return packed;
}

export default { initWasmBackend, isWasmReady, wasmFwht, wasmDot, wasmLloydMaxQuantize };
