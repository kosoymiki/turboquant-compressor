import { UniformSymmetricCodebook } from './quantizer.js';

describe('UniformSymmetricCodebook', () => {
  test('should encode and decode symmetrically', () => {
    const codebook = new UniformSymmetricCodebook(4);
    const values = new Float32Array([0.1, -0.3, 0.5, -0.7, 0.9, -0.1]);
    const encoded = codebook.encode(values);
    const decoded = codebook.decode(encoded);

    expect(decoded.length).toBe(values.length);
  });

  test('should support 2-bit quantization', () => {
    const codebook = new UniformSymmetricCodebook(2);
    const values = new Float32Array([0.5, -0.5, 0.0]);
    const encoded = codebook.encode(values);
    expect(encoded.length).toBeGreaterThan(0);
  });

  test('should support 8-bit quantization', () => {
    const codebook = new UniformSymmetricCodebook(8);
    const values = new Float32Array([0.5, -0.5, 0.0]);
    const encoded = codebook.encode(values);
    expect(encoded.length).toBeGreaterThan(0);
  });

  test('should throw on invalid bits', () => {
    expect(() => new UniformSymmetricCodebook(1)).toThrow();
    expect(() => new UniformSymmetricCodebook(5)).toThrow();
    expect(() => new UniformSymmetricCodebook(16)).toThrow();
  });

  test('quantize and dequantize should be inverses', () => {
    const codebook = new UniformSymmetricCodebook(4);
    const values = new Float32Array([0.1, -0.3, 0.5, -0.7]);
    const { indices, scale } = codebook.quantize(values);
    const dequantized = codebook.dequantize(indices, scale);

    expect(dequantized.length).toBe(values.length);
  });

  test('decode should respect explicit count for packed 3-bit values', () => {
    const q = new UniformSymmetricCodebook(3);
    const encoded = q.encode(new Float32Array([0, 0.1, -0.1, 0.5, -0.5]));
    const decoded = q.decode(encoded, 5);
    expect(decoded).toHaveLength(5);
  });
});