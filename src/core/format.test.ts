import { encodeCompressedDatabase, decodeCompressedDatabase, encodeBase64, decodeBase64 } from './format.js';

describe('Binary Format v2', () => {
  test('should encode and decode database', () => {
    const vectors = [
      new Uint8Array([128, 128]),
      new Uint8Array([200, 100])
    ];
    const norms = new Float32Array([1.0, 1.5]);
    const binary = encodeCompressedDatabase(vectors, 4, 4, 42, norms, undefined, 'uniform', 2);
    expect(binary.length).toBeGreaterThan(0);

    const encoded = encodeBase64(binary);
    const decoded = decodeCompressedDatabase(encoded);
    expect(decoded.version).toBe(2);
    expect(decoded.vectorCount).toBe(2);
    expect(decoded.dimensions).toBe(4);
  });
});

describe('Binary Format v3', () => {
  test('should encode and decode v3 database', () => {
    const vectors = [
      new Uint8Array([128, 128]),
      new Uint8Array([200, 100])
    ];
    const norms = new Float32Array([1.0, 1.5]);
    const binary = encodeCompressedDatabase(vectors, 4, 4, 42, norms, undefined, 'uniform', 3);
    expect(binary.length).toBeGreaterThan(0);
    expect(binary.length).toBe(80 + 8 + 4);

    const encoded = encodeBase64(binary);
    const decoded = decodeCompressedDatabase(encoded);
    expect(decoded.version).toBe(3);
    expect(decoded.vectorCount).toBe(2);
    expect(decoded.dimensions).toBe(4);
    expect(decoded.rotationSeed).toBe(42);
  });

  test('v3 header is 80 bytes', () => {
    const vectors = [new Uint8Array([128, 128])];
    const norms = new Float32Array([1.25]);
    const binary = encodeCompressedDatabase(vectors, 4, 4, 42, norms, undefined, 'uniform', 3);
    expect(binary.length).toBe(80 + 2 + 4);
  });

  test('v3 with QJL sketch', () => {
    const vectors = [new Uint8Array([128, 128])];
    const norms = new Float32Array([1.25]);
    const qjlSketch = new Uint8Array([1, 2, 3, 4]);
    const binary = encodeCompressedDatabase(vectors, 4, 4, 42, norms, qjlSketch, 'uniform', 3);
    expect(binary.length).toBe(80 + 2 + 4 + 4);

    const encoded = encodeBase64(binary);
    const decoded = decodeCompressedDatabase(encoded);
    expect(decoded.version).toBe(3);
    expect(decoded.qjlSketch).toEqual(qjlSketch);
  });
});

describe('Binary Format Utilities', () => {
  test('should throw on invalid magic', () => {
    expect(() => decodeCompressedDatabase('!!!')).toThrow();
  });

  test('base64 roundtrip', () => {
    const original = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const encoded = encodeBase64(original);
    const decoded = decodeBase64(encoded);
    expect(decoded).toEqual(original);
  });

  test('should reject corrupted norm payload via CRC', () => {
    const vectors = [new Uint8Array([128, 128])];
    const norms = new Float32Array([1.25]);
    const binary = encodeCompressedDatabase(vectors, 4, 4, 42, norms, undefined, 'uniform', 2);
    binary[72] = binary[72]! ^ 0xff;
    const encoded = encodeBase64(binary);
    expect(() => decodeCompressedDatabase(encoded)).toThrow(/CRC/);
  });

  test('should reject truncated payload', () => {
    const vectors = [new Uint8Array([128, 128])];
    const norms = new Float32Array([1.25]);
    const binary = encodeCompressedDatabase(vectors, 4, 4, 42, norms, undefined, 'uniform', 2);
    const truncated = binary.slice(0, binary.length - 1);
    const encoded = encodeBase64(truncated);
    expect(() => decodeCompressedDatabase(encoded)).toThrow();
  });

  test('should reject invalid base64 padding and characters', () => {
    expect(() => decodeBase64('!!!!')).toThrow();
    expect(() => decodeBase64('abc')).toThrow();
    expect(() => decodeBase64('abcd=')).toThrow();
  });

  test('rejects bad headerLength', () => {
    const vectors = [new Uint8Array([0xaa, 0xbb])];
    const norms = new Float32Array([1.25]);
    const binary = encodeCompressedDatabase(vectors, 4, 4, 42, norms, undefined, 'uniform', 2);
    const view = new DataView(binary.buffer, binary.byteOffset, binary.byteLength);
    view.setUint32(28, 0, true);
    expect(() => decodeCompressedDatabase(encodeBase64(binary))).toThrow(/headerLength/);
  });

  test('rejects payload length mismatch', () => {
    const vectors = [new Uint8Array([0xaa, 0xbb])];
    const norms = new Float32Array([1.25]);
    const binary = encodeCompressedDatabase(vectors, 4, 4, 42, norms, undefined, 'uniform', 2);
    const view = new DataView(binary.buffer, binary.byteOffset, binary.byteLength);
    view.setUint32(52, 99, true);
    expect(() => decodeCompressedDatabase(encodeBase64(binary))).toThrow(/payload length|qjl/i);
  });
});
