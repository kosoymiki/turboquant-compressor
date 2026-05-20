import {
  buildHashedTfidfVectorizer,
  createTokenHashVectorizer,
  createVectorizerFromSpec,
} from './vectorizer.js';

describe('vectorizer', () => {
  test('token hash is deterministic', () => {
    const vectorizer = createTokenHashVectorizer(32);
    const a = vectorizer.embed('TurboQuant custom stack retrieval');
    const b = vectorizer.embed('TurboQuant custom stack retrieval');
    expect(a).toEqual(b);
  });

  test('hashed tfidf is deterministic and reconstructible from spec', () => {
    const vectorizer = buildHashedTfidfVectorizer(32, [
      'turboquant custom stack retrieval',
      'mesa rusticl kgsl turnip',
      'lloyd max beats uniform on this corpus',
    ]);

    const rebuilt = createVectorizerFromSpec(vectorizer.spec);
    const sample = 'custom stack retrieval';
    expect(rebuilt.embed(sample)).toEqual(vectorizer.embed(sample));
  });

  test('hashed tfidf differentiates topical queries', () => {
    const vectorizer = buildHashedTfidfVectorizer(64, [
      'turboquant compression retrieval baseline',
      'mesa rusticl kgsl turnip vulkan opencl',
      'lloyd max beta qjl quantization',
    ]);

    const retrieval = vectorizer.embed('retrieval baseline');
    const driver = vectorizer.embed('rusticl kgsl');
    const quant = vectorizer.embed('lloyd max quantization');

    const dot = (a: number[], b: number[]) => a.reduce((sum, value, index) => sum + value * (b[index] ?? 0), 0);

    expect(dot(retrieval, driver)).toBeLessThan(dot(retrieval, retrieval));
    expect(dot(quant, driver)).toBeLessThan(dot(quant, quant));
  });
});
