import { Mulberry32, createMulberry32 } from './prng.js';

describe('Mulberry32', () => {
  test('should produce deterministic sequence', () => {
    const prng1 = createMulberry32(42);
    const prng2 = createMulberry32(42);

    for (let i = 0; i < 100; i++) {
      expect(prng1.nextUint32()).toBe(prng2.nextUint32());
    }
  });

  test('should produce values in [0, 1)', () => {
    const prng = createMulberry32(12345);
    for (let i = 0; i < 1000; i++) {
      const value = prng.nextUint32();
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  test('randomSign should return -1 or 1', () => {
    const prng = createMulberry32(54321);
    for (let i = 0; i < 100; i++) {
      const sign = prng.randomSign();
      expect([ -1, 1 ]).toContain(sign);
    }
  });

  test('gaussian should produce reasonable values', () => {
    const prng = createMulberry32(99999);
    const samples: number[] = [];
    for (let i = 0; i < 1000; i++) {
      samples.push(prng.gaussian());
    }
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(Math.abs(mean)).toBeLessThan(0.5);
  });
});