'use client';

import { useCallback, useSyncExternalStore } from 'react';

/**
 * What the reader has done, remembered between visits.
 *
 * Kept deliberately small and local: exercise outcomes only, in localStorage,
 * with no account and nothing leaving the browser. A book should not need a
 * backend to remember that you answered question four.
 */

export type Outcome = 'unanswered' | 'correct' | 'incorrect' | 'revealed';

const KEY = 'prr.progress.v1';
type State = Record<string, Outcome>;

let state: State | null = null;
const listeners = new Set<() => void>();

function load(): State {
  if (state) return state;
  if (typeof window === 'undefined') return {};
  try {
    state = JSON.parse(window.localStorage.getItem(KEY) ?? '{}') as State;
  } catch {
    state = {};
  }
  return state;
}

function save() {
  if (typeof window === 'undefined' || !state) return;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    // A full or disabled store is not worth interrupting the reader over.
  }
  for (const fn of listeners) fn();
}

export function record(id: string, outcome: Outcome) {
  const s = load();
  // Never downgrade a solved exercise because the reader came back to reread it.
  if (s[id] === 'correct' && outcome !== 'correct') return;
  s[id] = outcome;
  save();
}

export function clearChapter(prefix: string) {
  const s = load();
  for (const k of Object.keys(s)) if (k.startsWith(prefix)) delete s[k];
  save();
}

function subscribe(fn: () => void) {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

export function useOutcome(id: string): Outcome {
  return useSyncExternalStore(
    subscribe,
    () => load()[id] ?? 'unanswered',
    () => 'unanswered' as Outcome,
  );
}

/** How many exercises in a chapter are answered, and how many correctly. */
export function useChapterProgress(prefix: string) {
  const snapshot = useSyncExternalStore(
    subscribe,
    () => JSON.stringify(load()),
    () => '{}',
  );
  return useCallback(() => {
    const s = JSON.parse(snapshot) as State;
    const keys = Object.keys(s).filter((k) => k.startsWith(prefix));
    return {
      attempted: keys.length,
      correct: keys.filter((k) => s[k] === 'correct').length,
    };
  }, [snapshot, prefix])();
}
