/**
 * Beta PDF and Lloyd-Max codebook computation for TurboQuant.
 *
 * After random rotation, each coordinate of a unit-norm vector follows:
 * f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
 * which is a scaled Beta distribution on [-1, 1].
 */

/**
 * Log Gamma function approximation (Lanczos approximation).
 */
export function logGamma(x: number): number {
  if (x <= 0) throw new Error(`logGamma: x must be positive, got ${x}`);

  const g = 7;
  const p = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  ];

  if (x < 0.5) return Math.log(Math.PI / (Math.sin(Math.PI * x) * Math.exp(logGamma(1 - x))));

  x -= 1;
  let a = p[0]!;
  for (let i = 1; i < g + 2; i++) {
    a += p[i]! / (x + i);
  }
  const t = x + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * PDF of a single coordinate of a uniform random point on S^{d-1}.
 * f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
 */
export function sphereCoordinateBetaPdf(x: number, dimension: number): number {
  if (dimension <= 2) {
    throw new Error(`Dimension d=${dimension} too small, need d >= 3`);
  }
  if (x < -1 || x > 1) return 0;

  const logConst = logGamma(dimension / 2) - 0.5 * Math.log(Math.PI) - logGamma((dimension - 1) / 2);
  const exponent = (dimension - 3) / 2;
  const logVal = logConst + exponent * Math.log(Math.max(1e-15, 1 - x * x));
  return Math.exp(logVal);
}

/**
 * Adaptive Simpson integration.
 */
export function integrateAdaptiveSimpson(
  f: (x: number) => number,
  lo: number,
  hi: number,
  options?: { eps?: number; maxDepth?: number }
): number {
  const eps = options?.eps ?? 1e-10;
  const maxDepth = options?.maxDepth ?? 50;

  function simpson(a: number, b: number): number {
    const c = (a + b) / 2;
    return (b - a) / 6 * (f(a) + 4 * f(c) + f(b));
  }

  function adaptive(a: number, b: number, whole: number, depth: number): number {
    const c = (a + b) / 2;
    const left = simpson(a, c);
    const right = simpson(c, b);
    const delta = left + right - whole;

    if (depth >= maxDepth || Math.abs(delta) < 15 * eps) {
      return left + right + delta / 15;
    }

    return adaptive(a, c, left, depth + 1) + adaptive(c, b, right, depth + 1);
  }

  const whole = simpson(lo, hi);
  return adaptive(lo, hi, whole, 0);
}

/**
 * E[X | lo < X < hi] under the Beta PDF on [-1, 1].
 */
export function conditionalMeanBetaSphere(lo: number, hi: number, dimension: number): number {
  const num = integrateAdaptiveSimpson(
    (x) => x * sphereCoordinateBetaPdf(x, dimension),
    lo, hi
  );
  const den = integrateAdaptiveSimpson(
    (x) => sphereCoordinateBetaPdf(x, dimension),
    lo, hi
  );
  if (den < 1e-30) return (lo + hi) / 2;
  return num / den;
}

/**
 * Lloyd-Max codebook computation.
 */
export interface TurboQuantCodebook {
  centroids: number[];
  boundaries: number[];
  msePerCoord: number;
  mseTotal: number;
  dimension: number;
  bits: number;
  algorithm: 'lloyd-max-beta';
  source: 'computed' | 'preloaded';
}

export interface LloydMaxOptions {
  maxIter?: number;
  tol?: number;
}

export function computeLloydMaxBetaCodebook(
  dimension: number,
  bits: 1 | 2 | 3 | 4 | 5 | 6,
  options?: LloydMaxOptions
): TurboQuantCodebook {
  const nClusters = 2 ** bits;
  const maxIter = options?.maxIter ?? 200;
  const tol = options?.tol ?? 1e-12;

  // Initialize centroids using quantile midpoints
  const xGrid = Array.from({ length: 10000 }, (_, i) => -1 + 2 * i / 9999);
  const pdfVals = xGrid.map(x => sphereCoordinateBetaPdf(x, dimension));

  // Compute CDF
  const cdfVals: number[] = [];
  let cumsum = 0;
  for (let i = 0; i < pdfVals.length; i++) {
    cumsum += pdfVals[i]!;
    cdfVals.push(cumsum);
  }
  const total = cdfVals[cdfVals.length - 1]!;
  cdfVals.forEach((v, i) => cdfVals[i] = v / total);

  // Initial centroids at quantile midpoints
  const centroids = new Array(nClusters);
  for (let i = 0; i < nClusters; i++) {
    const qLo = i / nClusters;
    const qHi = (i + 1) / nClusters;
    const qMid = (qLo + qHi) / 2;
    const idx = cdfVals.findIndex(c => c >= qMid);
    centroids[i] = xGrid[Math.min(idx, xGrid.length - 1)];
  }

  // Lloyd-Max iterations.
  // IMPORTANT: cost must be recomputed from zero each iteration.
  let prevCost = Infinity;
  let finalCost = Infinity;

  for (let iteration = 0; iteration < maxIter; iteration++) {
    const boundaries = new Array<number>(nClusters + 1);
    boundaries[0] = -1;
    boundaries[nClusters] = 1;
    for (let i = 0; i < nClusters - 1; i++) {
      const left = centroids[i] as number;
      const right = centroids[i + 1] as number;
      boundaries[i + 1] = (left + right) / 2;
    }

    const newCentroids = new Array<number>(nClusters);
    for (let i = 0; i < nClusters; i++) {
      const lo = boundaries[i] as number;
      const hi = boundaries[i + 1] as number;
      newCentroids[i] = conditionalMeanBetaSphere(lo, hi, dimension);
    }

    let nextCost = 0;
    for (let i = 0; i < nClusters; i++) {
      const lo = boundaries[i] as number;
      const hi = boundaries[i + 1] as number;
      const c = newCentroids[i] as number;
      const integrand = (x: number) => (x - c) ** 2 * sphereCoordinateBetaPdf(x, dimension);
      nextCost += integrateAdaptiveSimpson(integrand, lo, hi);
    }

    for (let i = 0; i < nClusters; i++) {
      centroids[i] = newCentroids[i];
    }

    finalCost = nextCost;
    if (Math.abs(prevCost - nextCost) < tol) break;
    prevCost = nextCost;
  }

  // Final boundaries from converged centroids
  const finalBoundaries = new Array<number>(nClusters + 1);
  finalBoundaries[0] = -1;
  finalBoundaries[nClusters] = 1;
  for (let i = 0; i < nClusters - 1; i++) {
    const left = centroids[i] as number;
    const right = centroids[i + 1] as number;
    finalBoundaries[i + 1] = (left + right) / 2;
  }

  return {
    centroids: centroids as number[],
    boundaries: finalBoundaries,
    msePerCoord: finalCost,
    mseTotal: finalCost * dimension,
    dimension,
    bits,
    algorithm: 'lloyd-max-beta' as const,
    source: 'computed' as const
  };
}
