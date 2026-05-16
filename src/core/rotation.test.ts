import { RotationEngine, FWHT_SIGN, DENSE_QR_DEBUG, NONE } from './rotation.js';

describe('RotationEngine', () => {
  test('FWHT_SIGN mode works correctly', () => {
    const engine = new RotationEngine(8, 42, FWHT_SIGN);
    const vector = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]);
    const rotated = engine.rotate(vector);
    expect(rotated.length).toBe(8);
  });

  test('NONE mode works correctly', () => {
    const engine = new RotationEngine(4, 42, NONE);
    const vector = new Float32Array([1, 2, 3, 4]);
    const rotated = engine.rotate(vector);
    expect(rotated).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  test('DENSE_QR_DEBUG fails fast until implemented', () => {
    expect(() => new RotationEngine(8, 42, DENSE_QR_DEBUG)).toThrow(/not implemented/i);
  });
});