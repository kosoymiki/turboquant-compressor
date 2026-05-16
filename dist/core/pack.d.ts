/**
 * Bit-level packing for quantized values.
 * Handles cross-byte boundaries correctly.
 */
export declare function packValues(values: number[], bitsPerValue: number): Uint8Array;
export declare function unpackValues(bytes: Uint8Array, count: number, bitsPerValue: number): Float32Array;
export declare function packVectors(vectors: Uint8Array[], bitsPerValue: number): Uint8Array;
export declare function packedByteLength(vectorCount: number, dimensions: number, bitsPerValue: number): number;
