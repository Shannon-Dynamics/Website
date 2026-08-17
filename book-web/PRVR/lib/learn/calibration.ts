/**
 * Calibration diagnostics for density forecasts — Chapter 25.
 *
 * A learned observation model is admissible inside a Bayes filter only if the
 * confidence it states is the confidence it earns. That is a *measurable*
 * property, and this file is the measuring instrument: probability integral
 * transforms, reliability (quantile-coverage) diagrams, expected calibration
 * error, the Murphy decomposition of the Brier score, and the one-dimensional
 * search that fits a post-hoc temperature.
 *
 * Everything here is model-agnostic — it consumes PIT values and log scores,
 * never a model — so the same code grades a hand-tuned beam mixture, a learned
 * network, or the Kalman filter's own innovation sequence.
 */

/** The book uses 15 bins everywhere, following Guo et al. (2017). */
export const DEFAULT_BINS = 15;

export interface ReliabilityBin {
  /** Nominal (claimed) probability for this bin — the diagonal. */
  nominal: number;
  /** Empirical frequency actually observed at that nominal level. */
  empirical: number;
  /** How many evaluation points fell at or below the nominal level. */
  count: number;
}

export interface CalibrationReport {
  bins: ReliabilityBin[];
  /** D25.3, in the quantile-binned form: (1/B) Σ_b |F̂(p_b) − p_b|. */
  ece: number;
  /** Largest deviation from the diagonal — the Kolmogorov statistic. */
  maxCalibrationError: number;
  /** Mean negative log score, in nats per observation. Lower is sharper *and* better. */
  meanNll: number;
  /** Pearson χ² of the PIT histogram against uniform, with B − 1 d.o.f. */
  pitChi2: number;
  /** PIT histogram as *densities* (1.0 everywhere = perfectly uniform). */
  pitHistogram: number[];
  /** Number of evaluation points. */
  n: number;
}

/** Empirical CDF of the PIT values, evaluated at `level`. */
export function pitCoverage(pit: readonly number[], level: number): number {
  if (pit.length === 0) return 0;
  let hits = 0;
  for (const v of pit) if (v <= level) hits += 1;
  return hits / pit.length;
}

/**
 * Reliability diagram + ECE for a density forecast.
 *
 * For a *classifier*, D25.3 bins by predicted confidence. For a density
 * forecast the natural binning is by nominal quantile level: bin `b` claims
 * that a fraction `p_b = b/B` of observations fall at or below the model's
 * `p_b`-quantile, and the diagram plots what fraction actually did. Every bin
 * then holds the same number of claims, so `n_b / N = 1 / B` and D25.3 collapses
 * to a mean absolute deviation from the diagonal.
 */
export function reliabilityDiagram(
  pit: readonly number[],
  bins = DEFAULT_BINS,
): { bins: ReliabilityBin[]; ece: number; maxCalibrationError: number } {
  const out: ReliabilityBin[] = [];
  let sum = 0;
  let worst = 0;
  for (let b = 1; b <= bins; b++) {
    const nominal = b / bins;
    const empirical = pitCoverage(pit, nominal);
    const gap = Math.abs(empirical - nominal);
    sum += gap;
    if (gap > worst) worst = gap;
    out.push({ nominal, empirical, count: Math.round(empirical * pit.length) });
  }
  return { bins: out, ece: sum / bins, maxCalibrationError: worst };
}

/**
 * PIT histogram as densities, plus its Pearson χ² against uniformity.
 *
 * A uniform histogram is 1.0 in every bar. A **U shape** means the forecast is
 * too narrow (observations keep landing in the tails) — the signature of
 * overconfidence. A **hump** means it is too wide. A **tilt** means the forecast
 * is biased.
 */
export function pitHistogram(
  pit: readonly number[],
  bins = DEFAULT_BINS,
): { density: number[]; chi2: number } {
  const counts = new Array<number>(bins).fill(0);
  for (const v of pit) {
    const i = Math.min(bins - 1, Math.max(0, Math.floor(v * bins)));
    counts[i] += 1;
  }
  const n = pit.length;
  const expected = n / bins;
  let chi2 = 0;
  const density = counts.map((c) => {
    if (expected > 0) chi2 += ((c - expected) * (c - expected)) / expected;
    return n > 0 ? c / expected : 0;
  });
  return { density, chi2 };
}

/** The full report: one pass over PIT values and per-observation log scores. */
export function calibrationReport(
  pit: readonly number[],
  logScores: readonly number[],
  bins = DEFAULT_BINS,
): CalibrationReport {
  const rel = reliabilityDiagram(pit, bins);
  const hist = pitHistogram(pit, bins);
  const meanNll =
    logScores.length > 0
      ? -logScores.reduce((a, b) => a + b, 0) / logScores.length
      : Number.NaN;
  return {
    bins: rel.bins,
    ece: rel.ece,
    maxCalibrationError: rel.maxCalibrationError,
    meanNll,
    pitChi2: hist.chi2,
    pitHistogram: hist.density,
    n: pit.length,
  };
}

/* -------------------------------------------------------------------------- */
/* Proper scoring rules                                                        */
/* -------------------------------------------------------------------------- */

/** The log score, D25.1's strictly proper reference. Negatively oriented. */
export const logScore = (density: number): number => -Math.log(Math.max(density, 1e-300));

/** Brier score for a binary outcome: (p − y)², also strictly proper. */
export const brierScore = (p: number, y: 0 | 1): number => (p - y) * (p - y);

export interface MurphyDecomposition {
  brier: number;
  /** Σ n_b/N (conf_b − acc_b)² — how far the diagram sits off the diagonal. Lower is better. */
  reliability: number;
  /** Σ n_b/N (acc_b − ȳ)² — how much the forecast separates outcomes. Higher is better. */
  resolution: number;
  /** ȳ(1 − ȳ) — the problem's own difficulty. Nothing you do changes it. */
  uncertainty: number;
}

/**
 * Murphy's decomposition, `BS = reliability − resolution + uncertainty`.
 *
 * The identity is exact only when each bin's forecast is replaced by its bin
 * mean, which is why the estimator carries a binning bias — visible as the sum
 * of the three terms drifting from the raw Brier score when B is large and the
 * bins are thin.
 */
export function murphyDecomposition(
  forecasts: readonly number[],
  outcomes: readonly (0 | 1)[],
  bins = DEFAULT_BINS,
): MurphyDecomposition {
  const n = forecasts.length;
  const sumP = new Array<number>(bins).fill(0);
  const sumY = new Array<number>(bins).fill(0);
  const count = new Array<number>(bins).fill(0);
  let brier = 0;
  let yBar = 0;

  for (let i = 0; i < n; i++) {
    const b = Math.min(bins - 1, Math.max(0, Math.floor(forecasts[i] * bins)));
    sumP[b] += forecasts[i];
    sumY[b] += outcomes[i];
    count[b] += 1;
    brier += brierScore(forecasts[i], outcomes[i]);
    yBar += outcomes[i];
  }
  brier /= n;
  yBar /= n;

  let reliability = 0;
  let resolution = 0;
  for (let b = 0; b < bins; b++) {
    if (count[b] === 0) continue;
    const conf = sumP[b] / count[b];
    const acc = sumY[b] / count[b];
    reliability += (count[b] / n) * (conf - acc) * (conf - acc);
    resolution += (count[b] / n) * (acc - yBar) * (acc - yBar);
  }

  return { brier, reliability, resolution, uncertainty: yBar * (1 - yBar) };
}

/* -------------------------------------------------------------------------- */
/* Post-hoc calibration                                                        */
/* -------------------------------------------------------------------------- */

/**
 * `fit_temperature` — golden-section search for the scale minimizing a
 * validation objective on a log scale.
 *
 * Deliberately derivative-free and one-dimensional: temperature scaling has a
 * single parameter, it is fitted on *held-out* data, and it never touches the
 * model's ranking of hypotheses — which is exactly why it can fix calibration
 * without costing accuracy.
 */
export function fitTemperature(
  objective: (scale: number) => number,
  lo = 0.2,
  hi = 5,
  iters = 48,
): { scale: number; value: number } {
  const phi = (Math.sqrt(5) - 1) / 2;
  let a = Math.log(lo);
  let b = Math.log(hi);
  let c = b - phi * (b - a);
  let d = a + phi * (b - a);
  let fc = objective(Math.exp(c));
  let fd = objective(Math.exp(d));

  for (let k = 0; k < iters; k++) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - phi * (b - a);
      fc = objective(Math.exp(c));
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + phi * (b - a);
      fd = objective(Math.exp(d));
    }
  }
  const scale = Math.exp((a + b) / 2);
  return { scale, value: objective(scale) };
}

/* -------------------------------------------------------------------------- */
/* What overconfidence costs a particle filter                                 */
/* -------------------------------------------------------------------------- */

/**
 * Effective sample size of weights obtained by tempering log-likelihoods:
 * `w_i ∝ exp(κ ℓ_i)`, then `ESS = (Σw)² / Σw²`.
 *
 * The derivation in Chapter 25 (F3) predicts `ESS/M ≈ exp(−κ² s²)` when the
 * per-particle log-likelihoods have spread `s` — the reason a model that is
 * merely *twice* as sharp as it should be can cost a filter most of its
 * population in one update.
 */
export function temperedEss(logLikelihoods: readonly number[], kappa: number): number {
  const m = logLikelihoods.length;
  if (m === 0) return 0;
  let max = -Infinity;
  for (const l of logLikelihoods) if (kappa * l > max) max = kappa * l;
  if (!Number.isFinite(max)) return m;
  let s1 = 0;
  let s2 = 0;
  for (const l of logLikelihoods) {
    const w = Math.exp(kappa * l - max);
    s1 += w;
    s2 += w * w;
  }
  return s2 > 0 ? (s1 * s1) / s2 : 0;
}

/** The closed-form prediction of F3: ESS/M ≈ exp(−κ² s²). */
export function predictedEssFraction(logLikSpread: number, kappa: number): number {
  return Math.exp(-kappa * kappa * logLikSpread * logLikSpread);
}

/** Sample standard deviation of a set of log-likelihoods — the `s` above. */
export function spread(values: readonly number[]): number {
  const n = values.length;
  if (n < 2) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  let v = 0;
  for (const x of values) v += (x - mean) * (x - mean);
  return Math.sqrt(v / (n - 1));
}
