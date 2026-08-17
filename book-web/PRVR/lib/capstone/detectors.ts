/**
 * Derivation F4: the three statistics that let the supervisor notice a layer's
 * assumption has been violated.
 *
 * None of them is new. Each is a test the book already derived, transplanted
 * onto a different quantity — which is the point the capstone is making. A
 * stack does not need a fourth kind of mathematics to be safe; it needs the
 * mathematics it already has, pointed at the right signal, with a threshold
 * someone can defend.
 *
 * Rust counterpart: `crates/capstone/src/tasks/supervisor.rs`.
 */

import { SurpriseDetector, type SurpriseParams } from '../localize/augmented-mcl';

/**
 * F4(a) — mislocalisation, from the dual-EMA fitness ratio ρ = w_fast / w_slow.
 *
 * Chapter 12 computes this ratio over the *average particle weight* to decide
 * when Augmented MCL should inject random poses. Here the same detector eats
 * scan-match fitness instead. The justification transfers verbatim: the
 * statistic is a ratio, so it is invariant to the absolute scale of the score,
 * and the long EMA supplies the baseline that a fixed threshold on fitness
 * could never supply — a fitness of 0.4 is excellent in a cluttered room and
 * catastrophic in a corridor.
 *
 * ρ ≈ 1 in steady state. ρ below `rhoMin` for `patience` consecutive scans is
 * the kidnap alarm; requiring several in a row is what stops one bad match
 * behind a door from triggering a full relocalisation.
 */
export class FitnessMonitor {
  private ema: SurpriseDetector;
  private streak = 0;
  rho = 1;

  constructor(
    /**
     * Calibrated, not guessed: over four nominal missions (4232 scans) the
     * smallest ρ ever observed after the first five seconds was 0.919, so 0.80
     * leaves twelve percentage points of headroom before a healthy filter is
     * accused of being lost. The chapter's worked example redoes the arithmetic
     * of the alarm itself.
     */
    readonly rhoMin = 0.8,
    readonly patience = 2,
    params?: SurpriseParams,
  ) {
    this.ema = new SurpriseDetector(params);
  }

  /** Fold in this scan's fitness; returns true when the alarm fires. */
  update(fitness: number): boolean {
    this.ema.update(Math.max(fitness, 1e-6));
    this.rho = this.ema.wSlow > 0 ? this.ema.wFast / this.ema.wSlow : 1;
    if (this.rho < this.rhoMin) this.streak++;
    else this.streak = 0;
    return this.streak >= this.patience;
  }

  reset(): void {
    this.ema.reset();
    this.streak = 0;
    this.rho = 1;
  }
}

/** χ²₃ upper tail: the gate the SLAM front end's innovation has to clear. */
export const CHI2_3DOF_95 = 7.815;
export const CHI2_3DOF_99 = 11.345;
/** The gross gate. An innovation past this is not noise; it is a different room. */
export const CHI2_3DOF_999 = 16.266;

/**
 * F4(b) — filter divergence, from the normalised innovation squared
 * ε = νᵀ S⁻¹ ν.
 *
 * Under a correctly specified filter ε ~ χ²ₘ, so a single excursion past the
 * 95% point is expected once every twenty scans and means nothing. `patience`
 * consecutive excursions is a different event: the probability of that under
 * the null is 0.05^k, and by k = 4 it is one in 160 000. That is the entire
 * design of the test, and it is why the gate reports a *streak*, not a flag.
 */
export class ChiSquareGate {
  private streak = 0;
  last = 0;
  worst = 0;

  constructor(
    readonly threshold = CHI2_3DOF_95,
    readonly patience = 4,
  ) {}

  update(nis: number): boolean {
    this.last = nis;
    if (nis > this.worst) this.worst = nis;
    if (nis > this.threshold) this.streak++;
    else this.streak = 0;
    return this.streak >= this.patience;
  }

  get consecutive(): number {
    return this.streak;
  }

  reset(): void {
    this.streak = 0;
    this.last = 0;
  }
}

/**
 * F4(c) — the least glamorous and most load-bearing detector in the stack.
 *
 * A message that never arrives produces no statistic to test, so nothing
 * upstream can notice it. The watchdog notices the *absence*: if the last scan
 * is older than a few nominal periods, the estimator is running open-loop and
 * everything downstream is consuming a belief that is drifting with no
 * correction. Three periods is the usual choice — long enough to ride out one
 * dropped frame, short enough that the covariance has not yet grown past the
 * safety margin.
 */
export class Watchdog {
  private last = 0;

  constructor(
    readonly period: number,
    readonly tolerance = 3,
  ) {}

  touch(now: number): void {
    this.last = now;
  }

  age(now: number): number {
    return now - this.last;
  }

  expired(now: number): boolean {
    return this.age(now) > this.tolerance * this.period;
  }

  reset(now = 0): void {
    this.last = now;
  }
}
