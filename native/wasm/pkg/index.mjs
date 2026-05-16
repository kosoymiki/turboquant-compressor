/**
 * turboquant-wasm raw loader for Node.js
 * Works without wasm-bindgen CLI — manual memory management via WebAssembly API.
 * SIMD128 enabled, 37KB binary.
 */
import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WASM_PATH = join(__dirname, 'turboquant_wasm.wasm');

let instance = null;
let memory = null;

export async function init() {
  if (instance) return;
  const wasmBuffer = readFileSync(WASM_PATH);
  const { instance: inst } = await WebAssembly.instantiate(wasmBuffer, {
    env: { memory: new WebAssembly.Memory({ initial: 64, maximum: 256 }) }
  });
  instance = inst;
  memory = inst.exports.memory;
}

function allocF32(data) {
  const bytes = data.length * 4;
  const ptr = instance.exports.__wbindgen_malloc(bytes, 4);
  const view = new Float32Array(memory.buffer, ptr, data.length);
  view.set(data);
  return { ptr, len: data.length };
}

function readF32(ptr, len) {
  return new Float32Array(memory.buffer, ptr, len).slice();
}

function freePtr(ptr, len) {
  instance.exports.__wbindgen_free(ptr, len * 4, 4);
}

/**
 * FWHT in-place (power-of-2 dimension)
 */
export async function fwht(data) {
  await init();
  const { ptr, len } = allocF32(data);
  instance.exports.fwht(ptr, len);
  const result = readF32(ptr, len);
  freePtr(ptr, len);
  return result;
}

/**
 * SIMD dot product
 */
export async function simdDot(a, b) {
  await init();
  const pa = allocF32(a);
  const pb = allocF32(b);
  const result = instance.exports.simd_dot(pa.ptr, pa.len, pb.ptr, pb.len);
  freePtr(pa.ptr, pa.len);
  freePtr(pb.ptr, pb.len);
  return result;
}

/**
 * Lloyd-Max quantize
 */
export async function lloydMaxQuantize(values, centroids, bits) {
  await init();
  const pv = allocF32(values);
  const pc = allocF32(centroids);
  const resultPtr = instance.exports.lloyd_max_quantize(pv.ptr, pv.len, pc.ptr, pc.len, bits);
  const valsPerByte = Math.floor(8 / bits);
  const packedLen = Math.ceil(values.length / valsPerByte);
  const packed = new Uint8Array(memory.buffer, resultPtr, packedLen).slice();
  freePtr(pv.ptr, pv.len);
  freePtr(pc.ptr, pc.len);
  return packed;
}

/**
 * Lloyd-Max dequantize
 */
export async function lloydMaxDequantize(packed, centroids, bits, count) {
  await init();
  const pp = allocF32(new Float32Array(packed)); // treat as bytes via f32 hack
  const pc = allocF32(centroids);
  const resultPtr = instance.exports.lloyd_max_dequantize(pp.ptr, packed.length, pc.ptr, pc.len, bits, count);
  const result = readF32(resultPtr, count);
  freePtr(pp.ptr, pp.len);
  freePtr(pc.ptr, pc.len);
  return result;
}

/**
 * QJL 1-bit sign projection
 */
export async function qjlSignProject(vector, randomMatrix, projDim) {
  await init();
  const pv = allocF32(vector);
  const pm = allocF32(randomMatrix);
  const resultPtr = instance.exports.qjl_sign_project(pv.ptr, pv.len, pm.ptr, pm.len, projDim);
  const packedLen = Math.ceil(projDim / 8);
  const bits = new Uint8Array(memory.buffer, resultPtr, packedLen).slice();
  freePtr(pv.ptr, pv.len);
  freePtr(pm.ptr, pm.len);
  return bits;
}

/**
 * Batch FWHT (multiple vectors)
 */
export async function fwhtBatch(data, dim) {
  await init();
  const { ptr, len } = allocF32(data);
  instance.exports.fwht_batch(ptr, len, dim);
  const result = readF32(ptr, len);
  freePtr(ptr, len);
  return result;
}

export default { init, fwht, simdDot, lloydMaxQuantize, lloydMaxDequantize, qjlSignProject, fwhtBatch };
