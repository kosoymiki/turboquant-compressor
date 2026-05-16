/**
 * RaBitQ baseline: symmetric comparison infrastructure.
 */
import { rabitqEncode, rabitqScore, rabitqBatchScore, exactInnerProduct, recallAtK } from '../../src/core/rabitq_baseline.js';

function randomVec(d: number, seed: number): Float32Array {
  const v = new Float32Array(d);
  let s = seed;
  for (let i = 0; i < d; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    v[i] = (s / 0x7fffffff) * 2 - 1;
  }
  return v;
}

describe('RaBitQ baseline', () => {
  const DIM = 64;

  it('encode produces correct sign byte length', () => {
    const vec = randomVec(DIM, 1);
    const encoded = rabitqEncode(vec, 42);
    expect(encoded.signs.length).toBe(Math.ceil(DIM / 8));
    expect(encoded.dimension).toBe(DIM);
    expect(Number.isFinite(encoded.norm)).toBe(true);
    expect(encoded.norm).toBeGreaterThan(0);
  });

  it('score is finite', () => {
    const q = randomVec(DIM, 10);
    const v = randomVec(DIM, 20);
    const encoded = rabitqEncode(v, 42);
    const score = rabitqScore(q, encoded, 42);
    expect(Number.isFinite(score)).toBe(true);
  });

  it('self-score is positive (same direction)', () => {
    const v = randomVec(DIM, 5);
    const encoded = rabitqEncode(v, 42);
    const score = rabitqScore(v, encoded, 42);
    expect(score).toBeGreaterThan(0);
  });

  it('batchScore returns correct length', () => {
    const q = randomVec(DIM, 1);
    const vecs = Array.from({ length: 10 }, (_, i) => randomVec(DIM, i + 100));
    const encoded = vecs.map(v => rabitqEncode(v, 42));
    const scores = rabitqBatchScore(q, encoded, 42);
    expect(scores.length).toBe(10);
    scores.forEach(s => expect(Number.isFinite(s)).toBe(true));
  });

  it('exactInnerProduct matches manual computation', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([4, 5, 6]);
    expect(exactInnerProduct(a, b)).toBe(32); // 4+10+18
  });

  it('recallAtK returns value in [0, 1]', () => {
    const db = Array.from({ length: 20 }, (_, i) => randomVec(DIM, i + 200));
    const encoded = db.map(v => rabitqEncode(v, 42));
    const query = randomVec(DIM, 999);
    const recall = recallAtK(query, db, encoded, 5, 42);
    expect(recall).toBeGreaterThanOrEqual(0);
    expect(recall).toBeLessThanOrEqual(1);
  });

  it('deterministic: same seed same result', () => {
    const v = randomVec(DIM, 77);
    const e1 = rabitqEncode(v, 42);
    const e2 = rabitqEncode(v, 42);
    expect(Array.from(e1.signs)).toEqual(Array.from(e2.signs));
    expect(e1.norm).toBe(e2.norm);
  });
});
