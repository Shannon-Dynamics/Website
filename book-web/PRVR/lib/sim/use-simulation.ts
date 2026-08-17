'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useWidgetVisible } from '@/components/sim/deferred';

export interface SimulationOptions<S> {
  /** Build the initial state. Called on mount and on every reset/re-seed. */
  init: (seed: number) => S;
  /** Advance one step. Return the next state (may mutate and return the same object). */
  step: (state: S, tick: number) => S;
  /** Start playing immediately. Widgets autoplay by default: interaction is an
   *  invitation, not a requirement. */
  autoplay?: boolean;
  /** Simulation steps per second. */
  fps?: number;
  /** Stop after this many steps (undefined = run forever). */
  maxTicks?: number;
  /** Restart from tick 0 when maxTicks is reached. */
  loop?: boolean;
  initialSeed?: number;
}

export interface Simulation<S> {
  state: S;
  tick: number;
  playing: boolean;
  seed: number;
  speed: number;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  stepOnce: () => void;
  reset: () => void;
  reseed: (seed?: number) => void;
  setSpeed: (fps: number) => void;
  /** Imperatively replace state — for widgets with draggable entities. */
  setState: (updater: (s: S) => S) => void;
  finished: boolean;
}

/**
 * The animation spine shared by every simulation in the book.
 *
 * One requestAnimationFrame loop drives a fixed-rate simulation clock, so the
 * visible motion is frame-rate independent and a reader on a 120 Hz display
 * sees the same trajectory as one on 60 Hz. Everything is seeded, so "re-roll"
 * produces a genuinely different run and typing the seed back reproduces it.
 */
export function useSimulation<S>(opts: SimulationOptions<S>): Simulation<S> {
  const { init, step, autoplay = true, fps = 30, maxTicks, loop = true, initialSeed = 42 } = opts;

  const [seed, setSeed] = useState(initialSeed);
  const [state, setStateRaw] = useState<S>(() => init(initialSeed));
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(autoplay);
  const [speed, setSpeed] = useState(fps);

  // A widget scrolled out of view keeps its state but stops spending frames.
  // `null` means it is not inside a <Deferred> boundary, so it always runs.
  const onScreen = useWidgetVisible();
  const active = playing && onScreen !== false;

  // Latest values live in refs so the RAF loop never restarts mid-run.
  const stateRef = useRef(state);
  const tickRef = useRef(0);
  const stepRef = useRef(step);
  const accRef = useRef(0);
  const lastRef = useRef<number | null>(null);

  stepRef.current = step;

  const advance = useCallback(() => {
    const next = stepRef.current(stateRef.current, tickRef.current);
    stateRef.current = next;
    tickRef.current += 1;
    setStateRaw(next);
    setTick(tickRef.current);
  }, []);

  const finished = maxTicks !== undefined && tick >= maxTicks && !loop;

  useEffect(() => {
    if (!active) {
      lastRef.current = null;
      return;
    }

    let raf = 0;
    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      if (lastRef.current === null) {
        lastRef.current = now;
        return;
      }
      const dt = Math.min(now - lastRef.current, 250); // clamp after a tab switch
      lastRef.current = now;
      accRef.current += dt;

      const interval = 1000 / Math.max(speed, 1);
      let guard = 0;
      while (accRef.current >= interval && guard < 8) {
        accRef.current -= interval;
        guard += 1;

        if (maxTicks !== undefined && tickRef.current >= maxTicks) {
          if (!loop) {
            setPlaying(false);
            return;
          }
          tickRef.current = 0;
          stateRef.current = init(seed);
          setStateRaw(stateRef.current);
          setTick(0);
          continue;
        }
        advance();
      }
    };

    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [active, speed, maxTicks, loop, init, seed, advance]);

  const reset = useCallback(() => {
    tickRef.current = 0;
    accRef.current = 0;
    stateRef.current = init(seed);
    setStateRaw(stateRef.current);
    setTick(0);
  }, [init, seed]);

  const reseed = useCallback(
    (next?: number) => {
      const s = next ?? Math.floor(Math.random() * 100000);
      setSeed(s);
      tickRef.current = 0;
      accRef.current = 0;
      stateRef.current = init(s);
      setStateRaw(stateRef.current);
      setTick(0);
    },
    [init],
  );

  const setState = useCallback((updater: (s: S) => S) => {
    const next = updater(stateRef.current);
    stateRef.current = next;
    setStateRaw(next);
  }, []);

  return {
    state,
    tick,
    playing,
    seed,
    speed,
    finished,
    play: () => setPlaying(true),
    pause: () => setPlaying(false),
    toggle: () => setPlaying((p) => !p),
    stepOnce: () => {
      setPlaying(false);
      advance();
    },
    reset,
    reseed,
    setSpeed,
    setState,
  };
}
