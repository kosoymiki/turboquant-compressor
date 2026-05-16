import { packValues, unpackValues, packedByteLength } from './pack.js';
import { deterministicIntArray } from './test-utils.js';

describe('packValues', () => {
  test('2-bit roundtrip', () => {
    const values = [0, 1, 2, 3, 0, 1, 2, 3];
    const packed = packValues(values, 2);
    const unpacked = unpackValues(packed, values.length, 2);
    expect(Array.from(unpacked)).toEqual(values);
  });

  test('3-bit roundtrip with byte boundary crossing', () => {
    const values = [0, 1, 2, 3, 4, 5, 6, 7, 0, 7];
    const packed = packValues(values, 3);
    const unpacked = unpackValues(packed, values.length, 3);
    expect(Array.from(unpacked)).toEqual(values);
  });

  test('4-bit roundtrip', () => {
    const values = [0, 5, 10, 15, 0, 5, 10, 15];
    const packed = packValues(values, 4);
    const unpacked = unpackValues(packed, values.length, 4);
    expect(Array.from(unpacked)).toEqual(values);
  });

  test('8-bit roundtrip', () => {
    const values = Array.from({ length: 16 }, (_, i) => i * 17);
    const packed = packValues(values, 8);
    const unpacked = unpackValues(packed, values.length, 8);
    expect(Array.from(unpacked)).toEqual(values);
  });

  test('3-bit with odd count', () => {
    const values = [0, 1, 2, 3, 4, 5, 6];
    const packed = packValues(values, 3);
    const unpacked = unpackValues(packed, values.length, 3);
    expect(Array.from(unpacked)).toEqual(values);
  });

  test('random 3-bit values', () => {
    const values = deterministicIntArray(100, 8, 789);
    const packed = packValues(values, 3);
    const unpacked = unpackValues(packed, values.length, 3);
    expect(Array.from(unpacked)).toEqual(values);
  });

  test('packedByteLength calculation', () => {
    expect(packedByteLength(10, 4, 4)).toBe(20);
    expect(packedByteLength(10, 4, 3)).toBe(15);
    expect(packedByteLength(10, 4, 2)).toBe(10);
    expect(packedByteLength(1, 1024, 4)).toBe(512);
  });
});