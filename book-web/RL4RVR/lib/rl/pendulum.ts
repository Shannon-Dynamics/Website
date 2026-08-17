/**
 * Pendle — the book's pendulum / cart-pole, introduced in Chapter 2 as the
 * bridge between continuous robot dynamics and discrete-time decision-making.
 *
 * Dynamics (a torque-actuated pendulum about a pivot, θ measured from the
 * upright position, positive counter-clockwise):
 *
 *     m ℓ² θ̈ = m g ℓ sin θ − b θ̇ + τ
 *
 * Chapter 2 uses this to show why the integrator matters: explicit Euler
 * injects energy and the pendulum spirals out, while RK4 conserves it to
 * fourth order. Chapter 15 turns the same fact into the sim-to-real lesson.
 */

export interface PendleParams {
  mass: number;
  length: number;
  gravity: number;
  damping: number;
  maxTorque: number;
}

export const DEFAULT_PENDLE: PendleParams = {
  mass: 1,
  length: 1,
  gravity: 9.81,
  damping: 0.1,
  maxTorque: 2,
};

/** State: [θ, θ̇] with θ = 0 at the upright (inverted) position. */
export type PendleState = [number, number];

export function pendleDerivative(
  state: PendleState,
  torque: number,
  p: PendleParams = DEFAULT_PENDLE,
): PendleState {
  const [theta, omega] = state;
  const clamped = Math.max(-p.maxTorque, Math.min(p.maxTorque, torque));
  const inertia = p.mass * p.length * p.length;
  const alpha = (p.mass * p.gravity * p.length * Math.sin(theta) - p.damping * omega + clamped) / inertia;
  return [omega, alpha];
}

export function eulerStep(
  state: PendleState,
  torque: number,
  dt: number,
  p: PendleParams = DEFAULT_PENDLE,
): PendleState {
  const [dTheta, dOmega] = pendleDerivative(state, torque, p);
  return [state[0] + dt * dTheta, state[1] + dt * dOmega];
}

/** Semi-implicit (symplectic) Euler — updates velocity first, then position. */
export function semiImplicitEulerStep(
  state: PendleState,
  torque: number,
  dt: number,
  p: PendleParams = DEFAULT_PENDLE,
): PendleState {
  const [, dOmega] = pendleDerivative(state, torque, p);
  const omega = state[1] + dt * dOmega;
  return [state[0] + dt * omega, omega];
}

export function rk4Step(
  state: PendleState,
  torque: number,
  dt: number,
  p: PendleParams = DEFAULT_PENDLE,
): PendleState {
  const add = (s: PendleState, d: PendleState, h: number): PendleState => [
    s[0] + h * d[0],
    s[1] + h * d[1],
  ];
  const k1 = pendleDerivative(state, torque, p);
  const k2 = pendleDerivative(add(state, k1, dt / 2), torque, p);
  const k3 = pendleDerivative(add(state, k2, dt / 2), torque, p);
  const k4 = pendleDerivative(add(state, k3, dt), torque, p);
  return [
    state[0] + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
    state[1] + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
  ];
}

export type Integrator = 'euler' | 'semi-implicit' | 'rk4';

export const INTEGRATORS: Record<
  Integrator,
  (s: PendleState, tau: number, dt: number, p?: PendleParams) => PendleState
> = {
  euler: eulerStep,
  'semi-implicit': semiImplicitEulerStep,
  rk4: rk4Step,
};

/** Total mechanical energy — the quantity a good integrator should conserve. */
export function energy(state: PendleState, p: PendleParams = DEFAULT_PENDLE): number {
  const [theta, omega] = state;
  const kinetic = 0.5 * p.mass * p.length * p.length * omega * omega;
  const potential = p.mass * p.gravity * p.length * Math.cos(theta);
  return kinetic + potential;
}

/** Wrap an angle to (−π, π]. */
export function wrapAngle(theta: number): number {
  let t = theta;
  while (t > Math.PI) t -= 2 * Math.PI;
  while (t <= -Math.PI) t += 2 * Math.PI;
  return t;
}

/**
 * An energy-shaping swing-up controller with an LQR-style catch near upright —
 * the classical baseline Chapter 13 says RL must beat.
 */
export function swingUpController(
  state: PendleState,
  p: PendleParams = DEFAULT_PENDLE,
  kEnergy = 0.6,
): number {
  const [theta, omega] = state;
  const upright = Math.abs(wrapAngle(theta)) < 0.35;
  if (upright) {
    // Linear feedback: τ = −k_θ θ − k_ω θ̇ (poles placed for a fast catch).
    return Math.max(-p.maxTorque, Math.min(p.maxTorque, -18 * wrapAngle(theta) - 4.5 * omega));
  }
  const desired = p.mass * p.gravity * p.length; // energy at the upright rest
  const err = energy(state, p) - desired;
  return Math.max(-p.maxTorque, Math.min(p.maxTorque, -kEnergy * err * Math.sign(omega * Math.cos(theta))));
}

/** Simulate a trajectory, returning states sampled every step. */
export function simulate(
  initial: PendleState,
  steps: number,
  dt: number,
  integrator: Integrator,
  control: (s: PendleState, t: number) => number = () => 0,
  p: PendleParams = DEFAULT_PENDLE,
): { states: PendleState[]; energies: number[] } {
  const stepFn = INTEGRATORS[integrator];
  let s = initial;
  const states: PendleState[] = [s];
  const energies: number[] = [energy(s, p)];
  for (let i = 0; i < steps; i++) {
    const tau = control(s, i * dt);
    s = stepFn(s, tau, dt, p);
    states.push(s);
    energies.push(energy(s, p));
    if (!Number.isFinite(s[0]) || Math.abs(s[1]) > 1e4) break; // diverged
  }
  return { states, energies };
}
