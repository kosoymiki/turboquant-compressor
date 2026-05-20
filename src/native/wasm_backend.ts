/**
 * TurboQuant WASM SIMD128 backend — direct WebAssembly loader (ESM).
 * Implements full wasm-bindgen heap/stack protocol for &mut [f32] and Vec<T> returns.
 */
import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_PATH_CANDIDATES = [
  join(__dir, 'wasm', 'pkg', 'turboquant_wasm_bg.wasm'),
  join(__dir, '..', '..', 'native', 'wasm', 'pkg', 'turboquant_wasm_bg.wasm'),
];

let wasm: any = null;
let ready = false;

// --- wasm-bindgen heap protocol (free-list) ---
const heap = new Array(1028).fill(undefined);
heap.push(undefined, null, true, false);
let heap_next = heap.length;

function addHeapObject(obj: any): number {
  if (heap_next === heap.length) heap.push(heap.length + 1);
  const idx = heap_next;
  heap_next = heap[idx];
  heap[idx] = obj;
  return idx;
}

function getObject(idx: number): any { return heap[idx]; }

function dropObject(idx: number) {
  if (idx < 1028) return;
  heap[idx] = heap_next;
  heap_next = idx;
}

function takeObject(idx: number): any {
  const ret = getObject(idx);
  dropObject(idx);
  return ret;
}

// --- Memory views (invalidated on grow) ---
let cachedU8: Uint8Array | null = null;
let cachedF32: Float32Array | null = null;
let cachedDV: DataView | null = null;

function getU8(): Uint8Array {
  if (!cachedU8 || cachedU8.byteLength === 0) cachedU8 = new Uint8Array(wasm.memory.buffer);
  return cachedU8;
}
function getF32(): Float32Array {
  if (!cachedF32 || cachedF32.byteLength === 0) cachedF32 = new Float32Array(wasm.memory.buffer);
  return cachedF32;
}
function getDV(): DataView {
  if (!cachedDV || cachedDV.buffer !== wasm.memory.buffer) cachedDV = new DataView(wasm.memory.buffer);
  return cachedDV;
}

// --- Pass arrays to WASM ---
let WASM_VECTOR_LEN = 0;

function passF32(data: Float32Array): number {
  const ptr = (wasm.__wbindgen_export(data.length * 4, 4) as number) >>> 0;
  getF32().set(data, ptr / 4);
  WASM_VECTOR_LEN = data.length;
  return ptr;
}

function passU8(data: Uint8Array): number {
  const ptr = (wasm.__wbindgen_export(data.length, 1) as number) >>> 0;
  getU8().set(data, ptr);
  WASM_VECTOR_LEN = data.length;
  return ptr;
}

// --- Init ---
export async function initWasmBackend(): Promise<boolean> {
  if (ready) return true;
  try {
    let wasmBytes: Uint8Array | null = null;
    for (const candidate of WASM_PATH_CANDIDATES) {
      try {
        wasmBytes = readFileSync(candidate);
        break;
      } catch {
        continue;
      }
    }
    if (!wasmBytes) {
      ready = false;
      return false;
    }

    const importObj = {
      './turboquant_wasm_bg.js': {
        __wbg___wbindgen_copy_to_typed_array_787746aeb47818bc(arg0: number, arg1: number, arg2: number) {
          // Copy raw bytes from WASM into the original typed array
          const obj = getObject(arg2);
          new Uint8Array(obj.buffer, obj.byteOffset, obj.byteLength)
            .set(getU8().subarray(arg0 >>> 0, (arg0 >>> 0) + arg1));
        },
        __wbindgen_object_drop_ref(arg0: number) {
          takeObject(arg0);
        },
      },
    };

    const wasmBuffer = new Uint8Array(
      wasmBytes.buffer,
      wasmBytes.byteOffset,
      wasmBytes.byteLength
    ) as unknown as BufferSource;
    const mod = await WebAssembly.compile(wasmBuffer);
    const inst = await WebAssembly.instantiate(mod, importObj);
    wasm = inst.exports;
    ready = true;
    return true;
  } catch {
    ready = false;
    return false;
  }
}

export function isWasmReady(): boolean { return ready; }

// --- Public API matching wasm-bindgen calling conventions ---

/** FWHT in-place on Float32Array (mutates original) */
export function wasmFwht(data: Float32Array): Float32Array | null {
  if (!ready) return null;
  const ptr = passF32(data);
  const len = WASM_VECTOR_LEN;
  // wasm.fwht(ptr, len, heapRef) — heapRef used to copy result back into original array
  wasm.fwht(ptr, len, addHeapObject(data));
  // After return, data has been mutated in-place via copy_to_typed_array
  return data;
}

/** Batch FWHT on concatenated Float32Array */
export function wasmFwhtBatch(data: Float32Array, dim: number): Float32Array | null {
  if (!ready) return null;
  const ptr = passF32(data);
  const len = WASM_VECTOR_LEN;
  wasm.fwht_batch(ptr, len, addHeapObject(data), dim);
  return data;
}

/** SIMD dot product */
export function wasmDot(a: Float32Array, b: Float32Array): number | null {
  if (!ready) return null;
  const ptrA = passF32(a);
  const lenA = WASM_VECTOR_LEN;
  const ptrB = passF32(b);
  const lenB = WASM_VECTOR_LEN;
  return wasm.simd_dot(ptrA, lenA, ptrB, lenB) as number;
}

/** Lloyd-Max quantize → packed Uint8Array */
export function wasmLloydMaxQuantize(values: Float32Array, centroids: Float32Array, bits: number): Uint8Array | null {
  if (!ready) return null;
  try {
    const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
    const ptr0 = passF32(values);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passF32(centroids);
    const len1 = WASM_VECTOR_LEN;
    wasm.lloyd_max_quantize(retptr, ptr0, len0, ptr1, len1, bits);
    const r0 = getDV().getInt32(retptr + 0, true);
    const r1 = getDV().getInt32(retptr + 4, true);
    const result = getU8().subarray(r0 >>> 0, (r0 >>> 0) + r1).slice();
    wasm.__wbindgen_export2(r0, r1, 1);
    return result;
  } finally {
    wasm.__wbindgen_add_to_stack_pointer(16);
  }
}

/** Lloyd-Max dequantize → Float32Array */
export function wasmLloydMaxDequantize(packed: Uint8Array, centroids: Float32Array, bits: number, count: number): Float32Array | null {
  if (!ready) return null;
  try {
    const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
    const ptr0 = passU8(packed);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passF32(centroids);
    const len1 = WASM_VECTOR_LEN;
    wasm.lloyd_max_dequantize(retptr, ptr0, len0, ptr1, len1, bits, count);
    const r0 = getDV().getInt32(retptr + 0, true);
    const r1 = getDV().getInt32(retptr + 4, true);
    const result = getF32().subarray(r0 / 4, r0 / 4 + r1).slice();
    wasm.__wbindgen_export2(r0, r1 * 4, 4);
    return result;
  } finally {
    wasm.__wbindgen_add_to_stack_pointer(16);
  }
}

/** QJL sign-bit projection → packed bits */
export function wasmQjlSignProject(vector: Float32Array, randomMatrix: Float32Array, projDim: number): Uint8Array | null {
  if (!ready) return null;
  try {
    const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
    const ptr0 = passF32(vector);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passF32(randomMatrix);
    const len1 = WASM_VECTOR_LEN;
    wasm.qjl_sign_project(retptr, ptr0, len0, ptr1, len1, projDim);
    const r0 = getDV().getInt32(retptr + 0, true);
    const r1 = getDV().getInt32(retptr + 4, true);
    const result = getU8().subarray(r0 >>> 0, (r0 >>> 0) + r1).slice();
    wasm.__wbindgen_export2(r0, r1, 1);
    return result;
  } finally {
    wasm.__wbindgen_add_to_stack_pointer(16);
  }
}

export default { initWasmBackend, isWasmReady, wasmFwht, wasmFwhtBatch, wasmDot, wasmLloydMaxQuantize, wasmLloydMaxDequantize, wasmQjlSignProject };
