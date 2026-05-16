/* @ts-self-types="./turboquant_wasm.d.ts" */

/**
 * Fast Walsh-Hadamard Transform (in-place, power-of-2)
 * Butterfly is data-dependent (not SIMD-able), normalize uses f32x4.
 * @param {Float32Array} data
 */
function fwht(data) {
    var ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_export);
    var len0 = WASM_VECTOR_LEN;
    wasm.fwht(ptr0, len0, addHeapObject(data));
}
exports.fwht = fwht;

/**
 * Batch FWHT: apply to N vectors of same dimension
 * @param {Float32Array} data
 * @param {number} dim
 */
function fwht_batch(data, dim) {
    var ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_export);
    var len0 = WASM_VECTOR_LEN;
    wasm.fwht_batch(ptr0, len0, addHeapObject(data), dim);
}
exports.fwht_batch = fwht_batch;

/**
 * Dequantize packed indices back to f32 using centroids
 * @param {Uint8Array} packed
 * @param {Float32Array} centroids
 * @param {number} bits
 * @param {number} count
 * @returns {Float32Array}
 */
function lloyd_max_dequantize(packed, centroids, bits, count) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passArray8ToWasm0(packed, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(centroids, wasm.__wbindgen_export);
        const len1 = WASM_VECTOR_LEN;
        wasm.lloyd_max_dequantize(retptr, ptr0, len0, ptr1, len1, bits, count);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v3 = getArrayF32FromWasm0(r0, r1).slice();
        wasm.__wbindgen_export2(r0, r1 * 4, 4);
        return v3;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}
exports.lloyd_max_dequantize = lloyd_max_dequantize;

/**
 * Lloyd-Max quantize: find nearest centroid index for each value
 * centroids: sorted array of centroid values
 * Returns packed indices (bits_per_value per element)
 * @param {Float32Array} values
 * @param {Float32Array} centroids
 * @param {number} bits
 * @returns {Uint8Array}
 */
function lloyd_max_quantize(values, centroids, bits) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passArrayF32ToWasm0(values, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(centroids, wasm.__wbindgen_export);
        const len1 = WASM_VECTOR_LEN;
        wasm.lloyd_max_quantize(retptr, ptr0, len0, ptr1, len1, bits);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v3 = getArrayU8FromWasm0(r0, r1).slice();
        wasm.__wbindgen_export2(r0, r1 * 1, 1);
        return v3;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}
exports.lloyd_max_quantize = lloyd_max_quantize;

/**
 * Bit-pack 4 values (2-bit each) into 1 byte — SIMD-friendly layout
 * @param {Uint8Array} indices
 * @returns {Uint8Array}
 */
function pack_2bit(indices) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passArray8ToWasm0(indices, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        wasm.pack_2bit(retptr, ptr0, len0);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v2 = getArrayU8FromWasm0(r0, r1).slice();
        wasm.__wbindgen_export2(r0, r1 * 1, 1);
        return v2;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}
exports.pack_2bit = pack_2bit;

/**
 * QJL 1-bit sign projection: dot product signs with random matrix row
 * @param {Float32Array} vector
 * @param {Float32Array} random_matrix
 * @param {number} proj_dim
 * @returns {Uint8Array}
 */
function qjl_sign_project(vector, random_matrix, proj_dim) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passArrayF32ToWasm0(vector, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(random_matrix, wasm.__wbindgen_export);
        const len1 = WASM_VECTOR_LEN;
        wasm.qjl_sign_project(retptr, ptr0, len0, ptr1, len1, proj_dim);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v3 = getArrayU8FromWasm0(r0, r1).slice();
        wasm.__wbindgen_export2(r0, r1 * 1, 1);
        return v3;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}
exports.qjl_sign_project = qjl_sign_project;

/**
 * Exposed SIMD dot product for JS
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @returns {number}
 */
function simd_dot(a, b) {
    const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.simd_dot(ptr0, len0, ptr1, len1);
    return ret;
}
exports.simd_dot = simd_dot;

/**
 * Unpack 2-bit values
 * @param {Uint8Array} packed
 * @param {number} count
 * @returns {Uint8Array}
 */
function unpack_2bit(packed, count) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passArray8ToWasm0(packed, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        wasm.unpack_2bit(retptr, ptr0, len0, count);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v2 = getArrayU8FromWasm0(r0, r1).slice();
        wasm.__wbindgen_export2(r0, r1 * 1, 1);
        return v2;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}
exports.unpack_2bit = unpack_2bit;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_copy_to_typed_array_787746aeb47818bc: function(arg0, arg1, arg2) {
            new Uint8Array(getObject(arg2).buffer, getObject(arg2).byteOffset, getObject(arg2).byteLength).set(getArrayU8FromWasm0(arg0, arg1));
        },
        __wbindgen_object_drop_ref: function(arg0) {
            takeObject(arg0);
        },
    };
    return {
        __proto__: null,
        "./turboquant_wasm_bg.js": import0,
    };
}

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function dropObject(idx) {
    if (idx < 1028) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getObject(idx) { return heap[idx]; }

let heap = new Array(1024).fill(undefined);
heap.push(undefined, null, true, false);

let heap_next = heap.length;

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

let WASM_VECTOR_LEN = 0;

const wasmPath = new URL("./turboquant_wasm_bg.wasm", import.meta.url).pathname;
const wasmBytes = require('fs').readFileSync(wasmPath);
const wasmModule = new WebAssembly.Module(wasmBytes);
let wasmInstance = new WebAssembly.Instance(wasmModule, __wbg_get_imports());
let wasm = wasmInstance.exports;
