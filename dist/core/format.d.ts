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
export interface CompressedDatabase {
    magic: Uint8Array;
    version: number;
    dimensions: number;
    paddedDimensions: number;
    vectorCount: number;
    bitsPerValue: number;
    rotationSeed: number;
    flags: number;
    headerLength: number;
    normsOffset: number;
    normsLength: number;
    quantizedOffset: number;
    quantizedLength: number;
    qjlOffset: number;
    qjlLength: number;
    payloadCrc32: number;
    vectors: Uint8Array[];
    norms: Float32Array;
    qjlSketch?: Uint8Array;
}
export interface V3Header {
    magic: Uint8Array;
    version: number;
    dimensions: number;
    paddedDimensions: number;
    vectorCount: number;
    bitsPerValue: number;
    rotationSeed: number;
    flags: number;
    headerLength: number;
    normsOffset: number;
    normsLength: number;
    quantizedOffset: number;
    quantizedLength: number;
    qjlOffset: number;
    qjlLength: number;
    payloadCrc32: number;
}
export declare function encodeCompressedDatabase(vectors: Uint8Array[], dimensions: number, bitsPerValue: number, rotationSeed: number, norms: Float32Array, qjlSketch?: Uint8Array, version?: number): Uint8Array;
export declare function decodeCompressedDatabase(base64: string): CompressedDatabase;
export declare function encodeBase64(data: Uint8Array): string;
export declare function decodeBase64(base64: string): Uint8Array;
