'use client';

import { useCallback, useSyncExternalStore } from 'react';

/**
 * The channel between prose and figures.
 *
 * A sentence can carry a draggable number — "turn the sensor noise up to 0.35"
 * — and the figure beneath it responds as the reader drags. That only works if
 * the number and the figure share state without either one owning the other,
 * so both talk to this tiny store instead.
 *
 * Deliberately not React context: a widget is often several DOM subtrees away
 * from the sentence that drives it, and wrapping every chapter in providers
 * would make authoring worse for no benefit. Keys are namespaced by chapter
 * (`ch05.sigma`) so two chapters cannot collide.
 */

type Listener = () => void;

const values = new Map<string, number>();
const listeners = new Map<string, Set<Listener>>();

/** The role a reader is currently pointing at, for prose↔figure highlighting. */
export type Role = 'prior' | 'prediction' | 'measurement' | 'posterior' | 'truth';
let hovered: Role | null = null;
const hoverListeners = new Set<Listener>();

function emit(key: string) {
  const set = listeners.get(key);
  if (set) for (const fn of set) fn();
}

export function setValue(key: string, value: number) {
  if (values.get(key) === value) return;
  values.set(key, value);
  emit(key);
}

export function getValue(key: string, fallback: number): number {
  const v = values.get(key);
  return v === undefined ? fallback : v;
}

function subscribe(key: string, fn: Listener) {
  let set = listeners.get(key);
  if (!set) {
    set = new Set();
    listeners.set(key, set);
  }
  set.add(fn);
  return () => {
    set!.delete(fn);
  };
}

/**
 * Read an explorable value. `fallback` is what the figure uses until (and
 * unless) a `<Scrub>` with this key is dragged, so a widget is always usable
 * on its own.
 */
export function useExplorable(key: string, fallback: number): number {
  const sub = useCallback((fn: Listener) => subscribe(key, fn), [key]);
  const get = useCallback(() => getValue(key, fallback), [key, fallback]);
  // The server has no reader, so it always renders the default.
  return useSyncExternalStore(sub, get, () => fallback);
}

/** Read and write, for a control that both displays and sets a value. */
export function useExplorableState(
  key: string,
  fallback: number,
): [number, (v: number) => void] {
  const value = useExplorable(key, fallback);
  const set = useCallback((v: number) => setValue(key, v), [key]);
  return [value, set];
}

/* -------------------------------------------------------------------------- */
/* Role highlighting                                                           */
/* -------------------------------------------------------------------------- */

export function setHoveredRole(role: Role | null) {
  if (hovered === role) return;
  hovered = role;
  for (const fn of hoverListeners) fn();
}

function subscribeHover(fn: Listener) {
  hoverListeners.add(fn);
  return () => {
    hoverListeners.delete(fn);
  };
}

/**
 * Which estimation role the reader is pointing at, or null.
 *
 * Figures use this to emphasise one layer and mute the rest, so that hovering
 * the green term in an equation makes the measurement curve — and only that
 * curve — stand out.
 */
export function useHoveredRole(): Role | null {
  return useSyncExternalStore(
    subscribeHover,
    () => hovered,
    () => null,
  );
}

/**
 * Opacity for a layer drawn in `role`, given what the reader is pointing at.
 * 1 when nothing is hovered, so the default reading experience is unchanged.
 */
export function roleAlpha(role: Role, hoveredRole: Role | null): number {
  if (!hoveredRole) return 1;
  return hoveredRole === role ? 1 : 0.12;
}
