/**
 * Binary format v3 with 8-byte aligned header for Rust/Zig/C/WASM compatibility.
 * Magic: 'TQMC' (TurboQuant Magic Constant)
 *
 * v3 Layout (80 bytes, all fields 8-byte aligned):
 *   0-3:   Magic (4 bytes)
 *   4-7:   Version (4 bytes)
 *   8-11:  dimensions (4 bytes)
 *  12-15:  paddedDimensions (4 bytes)
 *  16-19:  vectorCount (4 bytes)
 *  20-23:  bitsPerValue (1) + reserved (3)
 *  24-27:  rotationSeed (4 bytes)
 *  28-31:  flags (1) + reserved (3)
 *  32-35:  headerLength (4 bytes)
 *  36-39:  reserved (4)
 *  40-43:  normsOffset (4 bytes)
 *  44-47:  normsLength (4 bytes)
 *  48-51:  quantizedOffset (4 bytes)
 *  52-55:  quantizedLength (4 bytes)
 *  56-59:  qjlOffset (4 bytes)
 *  60-63:  qjlLength (4 bytes)
 *  64-67:  payloadCrc32 (4 bytes)
 *  68-79:  reserved (12 bytes)
 */
import { crc32, verifyCrc32 } from './crc32.js';
import { isPowerOfTwo, nextPowerOfTwo } from './hadamard.js';
const MAGIC = new Uint8Array([0x54, 0x51, 0x4D, 0x43]); // 'TQMC'
const VERSION_V2 = 2;
const VERSION_V3 = 3;
const HEADER_SIZE_V3 = 80;
export function encodeCompressedDatabase(vectors, dimensions, bitsPerValue, rotationSeed, norms, qjlSketch, version = VERSION_V3) {
    if (version === VERSION_V2) {
        return encodeV2(vectors, dimensions, bitsPerValue, rotationSeed, norms, qjlSketch);
    }
    return encodeV3(vectors, dimensions, bitsPerValue, rotationSeed, norms, qjlSketch);
}
function encodeV2(vectors, dimensions, bitsPerValue, rotationSeed, norms, qjlSketch) {
    const paddedDimensions = isPowerOfTwo(dimensions) ? dimensions : nextPowerOfTwo(dimensions);
    const vectorLength = Math.ceil(paddedDimensions * bitsPerValue / 8);
    const quantizedLength = vectors.length * vectorLength;
    const normsLength = norms.length * 4;
    const qjlLength = qjlSketch?.length ?? 0;
    const HEADER_SIZE = 72;
    const normsOffset = HEADER_SIZE;
    const quantizedOffset = normsOffset + normsLength;
    const qjlOffset = quantizedOffset + quantizedLength;
    const payloadLength = normsLength + quantizedLength + qjlLength;
    const header = new ArrayBuffer(HEADER_SIZE);
    const headerView = new DataView(header);
    headerView.setUint8(0, MAGIC[0]);
    headerView.setUint8(1, MAGIC[1]);
    headerView.setUint8(2, MAGIC[2]);
    headerView.setUint8(3, MAGIC[3]);
    headerView.setUint32(4, VERSION_V2, true);
    headerView.setUint32(8, dimensions, true);
    headerView.setUint32(12, paddedDimensions, true);
    headerView.setUint32(16, vectors.length, true);
    headerView.setUint8(20, bitsPerValue);
    headerView.setUint32(21, rotationSeed, true);
    headerView.setUint8(25, 0);
    headerView.setUint32(28, HEADER_SIZE, true);
    headerView.setUint32(32, normsOffset, true);
    headerView.setUint32(36, normsLength, true);
    headerView.setUint32(40, quantizedOffset, true);
    headerView.setUint32(44, quantizedLength, true);
    headerView.setUint32(48, qjlOffset, true);
    headerView.setUint32(52, qjlLength, true);
    headerView.setUint32(56, 0, true);
    const totalLength = HEADER_SIZE + payloadLength;
    const result = new Uint8Array(totalLength);
    result.set(new Uint8Array(header), 0);
    const resultView = new DataView(result.buffer, result.byteOffset, result.byteLength);
    for (let i = 0; i < norms.length; i++) {
        resultView.setFloat32(normsOffset + i * 4, norms[i], true);
    }
    let offset = quantizedOffset;
    for (const vector of vectors) {
        result.set(vector, offset);
        offset += vector.length;
    }
    if (qjlSketch) {
        result.set(qjlSketch, qjlOffset);
    }
    const payload = result.slice(normsOffset, qjlOffset + qjlLength);
    const payloadCrc = crc32(payload);
    resultView.setUint32(56, payloadCrc, true);
    return result;
}
function encodeV3(vectors, dimensions, bitsPerValue, rotationSeed, norms, qjlSketch) {
    const paddedDimensions = isPowerOfTwo(dimensions) ? dimensions : nextPowerOfTwo(dimensions);
    const vectorLength = Math.ceil(paddedDimensions * bitsPerValue / 8);
    const quantizedLength = vectors.length * vectorLength;
    const normsLength = norms.length * 4;
    const qjlLength = qjlSketch?.length ?? 0;
    const normsOffset = HEADER_SIZE_V3;
    const quantizedOffset = normsOffset + normsLength;
    const qjlOffset = quantizedOffset + quantizedLength;
    const payloadLength = normsLength + quantizedLength + qjlLength;
    const header = new ArrayBuffer(HEADER_SIZE_V3);
    const headerView = new DataView(header);
    // Magic (0-3) + Version (4-7)
    headerView.setUint8(0, MAGIC[0]);
    headerView.setUint8(1, MAGIC[1]);
    headerView.setUint8(2, MAGIC[2]);
    headerView.setUint8(3, MAGIC[3]);
    headerView.setUint32(4, VERSION_V3, true);
    // dimensions (8-11) + paddedDimensions (12-15) + vectorCount (16-19)
    headerView.setUint32(8, dimensions, true);
    headerView.setUint32(12, paddedDimensions, true);
    headerView.setUint32(16, vectors.length, true);
    // bitsPerValue (20) + reserved (21-23)
    headerView.setUint8(20, bitsPerValue);
    headerView.setUint8(21, 0);
    headerView.setUint8(22, 0);
    headerView.setUint8(23, 0);
    // rotationSeed (24-27)
    headerView.setUint32(24, rotationSeed, true);
    // flags (28) + reserved (29-31)
    headerView.setUint8(28, 0);
    headerView.setUint8(29, 0);
    headerView.setUint8(30, 0);
    headerView.setUint8(31, 0);
    // headerLength (32-35) + reserved (36-39)
    headerView.setUint32(32, HEADER_SIZE_V3, true);
    headerView.setUint32(36, 0, true);
    // Offsets (40-63)
    headerView.setUint32(40, normsOffset, true);
    headerView.setUint32(44, normsLength, true);
    headerView.setUint32(48, quantizedOffset, true);
    headerView.setUint32(52, quantizedLength, true);
    headerView.setUint32(56, qjlOffset, true);
    headerView.setUint32(60, qjlLength, true);
    // payloadCrc32 (64-67) + reserved (68-79)
    headerView.setUint32(64, 0, true);
    headerView.setUint32(68, 0, true);
    headerView.setUint32(72, 0, true);
    headerView.setUint32(76, 0, true);
    const totalLength = HEADER_SIZE_V3 + payloadLength;
    const result = new Uint8Array(totalLength);
    result.set(new Uint8Array(header), 0);
    const resultView = new DataView(result.buffer, result.byteOffset, result.byteLength);
    for (let i = 0; i < norms.length; i++) {
        resultView.setFloat32(normsOffset + i * 4, norms[i], true);
    }
    let offset = quantizedOffset;
    for (const vector of vectors) {
        result.set(vector, offset);
        offset += vector.length;
    }
    if (qjlSketch) {
        result.set(qjlSketch, qjlOffset);
    }
    const payload = result.slice(normsOffset, qjlOffset + qjlLength);
    const payloadCrc = crc32(payload);
    resultView.setUint32(64, payloadCrc, true);
    return result;
}
export function decodeCompressedDatabase(base64) {
    const binary = decodeBase64(base64);
    if (binary.length < 72) {
        throw new Error(`Data too short for header: ${binary.length} < 72`);
    }
    for (let i = 0; i < 4; i++) {
        if (binary[i] !== MAGIC[i]) {
            throw new Error('Invalid magic number');
        }
    }
    const binaryView = new DataView(binary.buffer, binary.byteOffset, binary.byteLength);
    const version = binaryView.getUint32(4, true);
    if (version === VERSION_V2) {
        return decodeV2(binary, binaryView);
    }
    else if (version === VERSION_V3) {
        return decodeV3(binary, binaryView);
    }
    throw new Error(`Unsupported version: ${version}, expected ${VERSION_V2} or ${VERSION_V3}`);
}
function decodeV2(binary, binaryView) {
    const HEADER_SIZE = 72;
    const dimensions = binaryView.getUint32(8, true);
    const paddedDimensions = binaryView.getUint32(12, true);
    const vectorCount = binaryView.getUint32(16, true);
    const bitsPerValue = binaryView.getUint8(20);
    const rotationSeed = binaryView.getUint32(21, true);
    const flags = binaryView.getUint8(25);
    const headerLength = binaryView.getUint32(28, true);
    const normsOffset = binaryView.getUint32(32, true);
    const normsLength = binaryView.getUint32(36, true);
    const quantizedOffset = binaryView.getUint32(40, true);
    const quantizedLength = binaryView.getUint32(44, true);
    const qjlOffset = binaryView.getUint32(48, true);
    const qjlLength = binaryView.getUint32(52, true);
    const storedPayloadCrc = binaryView.getUint32(56, true);
    if (headerLength !== HEADER_SIZE) {
        throw new Error(`Invalid headerLength: ${headerLength}, expected ${HEADER_SIZE}`);
    }
    if (vectorCount < 1 || vectorCount > 4096) {
        throw new Error(`Invalid vectorCount: ${vectorCount}`);
    }
    if (normsOffset !== HEADER_SIZE) {
        throw new Error(`Invalid normsOffset: ${normsOffset}, expected ${HEADER_SIZE}`);
    }
    if (quantizedOffset !== normsOffset + normsLength) {
        throw new Error(`Invalid quantizedOffset: ${quantizedOffset}, expected ${normsOffset + normsLength}`);
    }
    if (qjlOffset !== quantizedOffset + quantizedLength) {
        throw new Error(`Invalid qjlOffset: ${qjlOffset}, expected ${quantizedOffset + quantizedLength}`);
    }
    if (qjlOffset + qjlLength !== binary.length) {
        throw new Error(`Invalid payload length: qjlOffset + qjlLength = ${qjlOffset + qjlLength}, binary length = ${binary.length}`);
    }
    if (dimensions < 1 || dimensions > 8192) {
        throw new Error(`Invalid dimensions: ${dimensions}`);
    }
    if (paddedDimensions !== (isPowerOfTwo(dimensions) ? dimensions : nextPowerOfTwo(dimensions))) {
        throw new Error(`Invalid paddedDimensions: ${paddedDimensions} for dimensions ${dimensions}`);
    }
    if (![2, 3, 4, 8].includes(bitsPerValue)) {
        throw new Error(`Invalid bitsPerValue: ${bitsPerValue}`);
    }
    const expectedNormsLength = vectorCount * 4;
    if (normsLength !== expectedNormsLength) {
        throw new Error(`Invalid normsLength: ${normsLength}, expected ${expectedNormsLength}`);
    }
    const vectorLength = Math.ceil(paddedDimensions * bitsPerValue / 8);
    const expectedQuantizedLength = vectorCount * vectorLength;
    if (quantizedLength !== expectedQuantizedLength) {
        throw new Error(`Invalid quantizedLength: ${quantizedLength}, expected ${expectedQuantizedLength}`);
    }
    const payload = binary.slice(normsOffset, qjlOffset + qjlLength);
    if (!verifyCrc32(payload, storedPayloadCrc)) {
        throw new Error('Payload CRC32 verification failed');
    }
    const norms = new Float32Array(vectorCount);
    for (let i = 0; i < vectorCount; i++) {
        norms[i] = binaryView.getFloat32(normsOffset + i * 4, true);
    }
    const vectors = [];
    for (let i = 0; i < vectorCount; i++) {
        const start = quantizedOffset + i * vectorLength;
        vectors.push(binary.slice(start, start + vectorLength));
    }
    let qjlSketch;
    if (qjlLength > 0) {
        qjlSketch = binary.slice(qjlOffset, qjlOffset + qjlLength);
    }
    return {
        magic: MAGIC,
        version: VERSION_V2,
        dimensions,
        paddedDimensions,
        vectorCount,
        bitsPerValue,
        rotationSeed,
        flags,
        headerLength,
        normsOffset,
        normsLength,
        quantizedOffset,
        quantizedLength,
        qjlOffset,
        qjlLength,
        payloadCrc32: storedPayloadCrc,
        vectors,
        norms,
        qjlSketch
    };
}
function decodeV3(binary, binaryView) {
    const dimensions = binaryView.getUint32(8, true);
    const paddedDimensions = binaryView.getUint32(12, true);
    const vectorCount = binaryView.getUint32(16, true);
    const bitsPerValue = binaryView.getUint8(20);
    const rotationSeed = binaryView.getUint32(24, true);
    const flags = binaryView.getUint8(28);
    const headerLength = binaryView.getUint32(32, true);
    const normsOffset = binaryView.getUint32(40, true);
    const normsLength = binaryView.getUint32(44, true);
    const quantizedOffset = binaryView.getUint32(48, true);
    const quantizedLength = binaryView.getUint32(52, true);
    const qjlOffset = binaryView.getUint32(56, true);
    const qjlLength = binaryView.getUint32(60, true);
    const storedPayloadCrc = binaryView.getUint32(64, true);
    if (headerLength !== HEADER_SIZE_V3) {
        throw new Error(`Invalid headerLength: ${headerLength}, expected ${HEADER_SIZE_V3}`);
    }
    if (vectorCount < 1 || vectorCount > 4096) {
        throw new Error(`Invalid vectorCount: ${vectorCount}`);
    }
    if (normsOffset !== HEADER_SIZE_V3) {
        throw new Error(`Invalid normsOffset: ${normsOffset}, expected ${HEADER_SIZE_V3}`);
    }
    if (quantizedOffset !== normsOffset + normsLength) {
        throw new Error(`Invalid quantizedOffset: ${quantizedOffset}, expected ${normsOffset + normsLength}`);
    }
    if (qjlOffset !== quantizedOffset + quantizedLength) {
        throw new Error(`Invalid qjlOffset: ${qjlOffset}, expected ${quantizedOffset + quantizedLength}`);
    }
    if (qjlOffset + qjlLength !== binary.length) {
        throw new Error(`Invalid payload length: qjlOffset + qjlLength = ${qjlOffset + qjlLength}, binary length = ${binary.length}`);
    }
    if (dimensions < 1 || dimensions > 8192) {
        throw new Error(`Invalid dimensions: ${dimensions}`);
    }
    if (paddedDimensions !== (isPowerOfTwo(dimensions) ? dimensions : nextPowerOfTwo(dimensions))) {
        throw new Error(`Invalid paddedDimensions: ${paddedDimensions} for dimensions ${dimensions}`);
    }
    if (![2, 3, 4, 8].includes(bitsPerValue)) {
        throw new Error(`Invalid bitsPerValue: ${bitsPerValue}`);
    }
    const expectedNormsLength = vectorCount * 4;
    if (normsLength !== expectedNormsLength) {
        throw new Error(`Invalid normsLength: ${normsLength}, expected ${expectedNormsLength}`);
    }
    const vectorLength = Math.ceil(paddedDimensions * bitsPerValue / 8);
    const expectedQuantizedLength = vectorCount * vectorLength;
    if (quantizedLength !== expectedQuantizedLength) {
        throw new Error(`Invalid quantizedLength: ${quantizedLength}, expected ${expectedQuantizedLength}`);
    }
    const payload = binary.slice(normsOffset, qjlOffset + qjlLength);
    if (!verifyCrc32(payload, storedPayloadCrc)) {
        throw new Error('Payload CRC32 verification failed');
    }
    const norms = new Float32Array(vectorCount);
    for (let i = 0; i < vectorCount; i++) {
        norms[i] = binaryView.getFloat32(normsOffset + i * 4, true);
    }
    const vectors = [];
    for (let i = 0; i < vectorCount; i++) {
        const start = quantizedOffset + i * vectorLength;
        vectors.push(binary.slice(start, start + vectorLength));
    }
    let qjlSketch;
    if (qjlLength > 0) {
        qjlSketch = binary.slice(qjlOffset, qjlOffset + qjlLength);
    }
    return {
        magic: MAGIC,
        version: VERSION_V3,
        dimensions,
        paddedDimensions,
        vectorCount,
        bitsPerValue,
        rotationSeed,
        flags,
        headerLength,
        normsOffset,
        normsLength,
        quantizedOffset,
        quantizedLength,
        qjlOffset,
        qjlLength,
        payloadCrc32: storedPayloadCrc,
        vectors,
        norms,
        qjlSketch
    };
}
export function encodeBase64(data) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    let result = '';
    for (let i = 0; i < data.length; i += 3) {
        const b1 = data[i];
        const b2 = i + 1 < data.length ? data[i + 1] : 0;
        const b3 = i + 2 < data.length ? data[i + 2] : 0;
        const triplet = (b1 << 16) | (b2 << 8) | b3;
        result += chars[(triplet >> 18) & 0x3F];
        result += chars[(triplet >> 12) & 0x3F];
        result += i + 1 < data.length ? chars[(triplet >> 6) & 0x3F] : '=';
        result += i + 2 < data.length ? chars[triplet & 0x3F] : '=';
    }
    return result;
}
export function decodeBase64(base64) {
    if (typeof base64 !== 'string') {
        throw new Error('Invalid base64 input: expected string');
    }
    if (base64.length === 0) {
        return new Uint8Array(0);
    }
    if (base64.length % 4 !== 0) {
        throw new Error('Invalid base64 length');
    }
    const base64Regex = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
    if (!base64Regex.test(base64)) {
        throw new Error('Invalid base64 format');
    }
    return new Uint8Array(Buffer.from(base64, 'base64'));
}
