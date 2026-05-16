/**
 * TurboQuant WASM SIMD128 backend — direct WebAssembly loader (ESM).
 * Bypasses wasm-bindgen CJS/ESM conflicts.
 * 17KB binary, f32x4 vectorized FWHT/Lloyd-Max/QJL.
 */
import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_PATH = join(__dir, 'wasm', 'pkg', 'turboquant_wasm_bg.wasm');

let exports: any = null;
let memory: WebAssembly.Memory;
let ready = false;

export async function initWasmBackend(): Promise<boolean> {
  if (ready) return true;
  try {
    const wasmBytes = readFileSync(WASM_PATH);

    // Heap for wasm-bindgen object references
    const heap: any[] = [undefined, null, true, false];
    let heap_next = heap.length;
    function addHeapObject(obj: any) { heap[heap_next] = obj; return heap_next++; }
    function getObject(idx: number) { return heap[idx]; }
    function dropObject(idx: number) { heap[idx] = undefined; }

    const importObj = {
      './turboquant_wasm_bg.js': {
        __wbg___wbindgen_copy_to_typed_array_787746aeb47818bc(ptr: number, len: number, dst: number) {
          const arr = getObject(dst) as Float32Array;
          arr.set(new Float32Array(memory.buffer, ptr, len / 4));
        },
        __wbindgen_object_drop_ref(idx: number) { dropObject(idx); },
      },
    };

    const mod = await WebAssembly.compile(wasmBytes);
    const inst = await WebAssembly.instantiate(mod, importObj);
    exports = inst.exports;
    memory = exports.memory as WebAssembly.Memory;
    ready = true;
    return true;
  } catch {
    ready = false;
    return false;
  }
}

export function isWasmReady(): boolean { return ready; }

function allocF32(data: Float32Array): [number, number] {
  const bytes = data.byteLength;
  const ptr = exports.__wbindgen_export(bytes, 4);
  new Float32Array(memory.buffer, ptr, data.length).set(data);
  return [ptr, data.length];
}

function freePtr(ptr: number, len: number) {
  exports.__wbindgen_export2?.(ptr, len * 4, 4);
}

export function wasmFwht(data: Float32Array): Float32Array | null {
  if (!ready) return null;
  const [ptr, len] = allocF32(data);
  exports.fwht(ptr, len);
  const result = new Float32Array(memory.buffer, ptr, len).slice();
  freePtr(ptr, len);
  return result;
}

export function wasmFwhtBatch(data: Float32Array, dim: number): Float32Array | null {
  if (!ready) return null;
  const [ptr, len] = allocF32(data);
  exports.fwht_batch(ptr, len, dim);
  const result = new Float32Array(memory.buffer, ptr, len).slice();
  freePtr(ptr, len);
  return result;
}

export function wasmDot(a: Float32Array, b: Float32Array): number | null {
  if (!ready) return null;
  const [ptrA, lenA] = allocF32(a);
  const [ptrB, lenB] = allocF32(b);
  const result = exports.simd_dot(ptrA, lenA, ptrB, lenB);
  freePtr(ptrA, lenA);
  freePtr(ptrB, lenB);
  return result;
}

export function wasmLloydMaxQuantize(values: Float32Array, centroids: Float32Array, bits: number): Uint8Array | null {
  if (!ready) return null;
  const [pv, lv] = allocF32(values);
  const [pc, lc] = allocF32(centroids);
  const retPtr = exports.lloyd_max_quantize(pv, lv, pc, lc, bits);
  const valsPerByte = Math.floor(8 / bits);
  const packedLen = Math.ceil(values.length / valsPerByte);
  const packed = new Uint8Array(memory.buffer, retPtr, packedLen).slice();
  freePtr(pv, lv);
  freePtr(pc, lc);
  return packed;
}

export function wasmLloydMaxDequantize(packed: Uint8Array, centroids: Float32Array, bits: number, count: number): Float32Array | null {
  if (!ready) return null;
  // Allocate and copy packed bytes
  const pBytes = packed.byteLength;
  const pp = exports.__wbindgen_export(pBytes, 1);
  new Uint8Array(memory.buffer, pp, pBytes).set(packed);
  const [pc, lc] = allocF32(centroids);
  const retPtr = exports.lloyd_max_dequantize(pp, pBytes, pc, lc, bits, count);
  const result = new Float32Array(memory.buffer, retPtr, count).slice();
  freePtr(pc, lc);
  return result;
}

export function wasmQjlSignProject(vector: Float32Array, randomMatrix: Float32Array, projDim: number): Uint8Array | null {
  if (!ready) return null;
  const [pv, lv] = allocF32(vector);
  const [pm, lm] = allocF32(randomMatrix);
  const retPtr = exports.qjl_sign_project(pv, lv, pm, lm, projDim);
  const packedLen = Math.ceil(projDim / 8);
  const bits = new Uint8Array(memory.buffer, retPtr, packedLen).slice();
  freePtr(pv, lv);
  freePtr(pm, lm);
  return bits;
}

export default { initWasmBackend, isWasmReady, wasmFwht, wasmFwhtBatch, wasmDot, wasmLloydMaxQuantize, wasmLloydMaxDequantize, wasmQjlSignProject };
