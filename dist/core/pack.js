/**
 * Bit-level packing for quantized values.
 * Handles cross-byte boundaries correctly.
 */
export function packValues(values, bitsPerValue) {
    const bitsPerByte = 8;
    const totalBits = values.length * bitsPerValue;
    const totalBytes = Math.ceil(totalBits / bitsPerByte);
    const bytes = new Uint8Array(totalBytes);
    let bitOffset = 0;
    for (const value of values) {
        const byteIndex = Math.floor(bitOffset / bitsPerByte);
        const bitPosition = bitOffset % bitsPerByte;
        const bitsRemainingInByte = bitsPerByte - bitPosition;
        const lowerBits = Math.min(bitsRemainingInByte, bitsPerValue);
        const upperBits = bitsPerValue - lowerBits;
        bytes[byteIndex] |= (value & ((1 << lowerBits) - 1)) << bitPosition;
        if (upperBits > 0) {
            bytes[byteIndex + 1] |= (value >> lowerBits) & ((1 << upperBits) - 1);
        }
        bitOffset += bitsPerValue;
    }
    return bytes;
}
export function unpackValues(bytes, count, bitsPerValue) {
    const values = new Float32Array(count);
    const bitsPerByte = 8;
    const mask = (1 << bitsPerValue) - 1;
    let bitOffset = 0;
    for (let i = 0; i < count; i++) {
        const byteIndex = Math.floor(bitOffset / bitsPerByte);
        const bitPosition = bitOffset % bitsPerByte;
        const bitsRemainingInByte = bitsPerByte - bitPosition;
        const lowerBits = Math.min(bitsRemainingInByte, bitsPerValue);
        const upperBits = bitsPerValue - lowerBits;
        let value = (bytes[byteIndex] >> bitPosition) & ((1 << lowerBits) - 1);
        if (upperBits > 0) {
            value |= (bytes[byteIndex + 1] & ((1 << upperBits) - 1)) << lowerBits;
        }
        values[i] = value;
        bitOffset += bitsPerValue;
    }
    return values;
}
export function packVectors(vectors, bitsPerValue) {
    const totalLength = vectors.reduce((sum, v) => sum + v.length, 0);
    const packed = new Uint8Array(totalLength);
    let offset = 0;
    for (const vector of vectors) {
        packed.set(vector, offset);
        offset += vector.length;
    }
    return packed;
}
export function packedByteLength(vectorCount, dimensions, bitsPerValue) {
    const totalValues = vectorCount * dimensions;
    const bitsPerByte = 8;
    return Math.ceil((totalValues * bitsPerValue) / bitsPerByte);
}
