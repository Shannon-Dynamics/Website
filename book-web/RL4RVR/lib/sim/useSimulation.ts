'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { JobName } from './worker';

interface State<T> {
  data: T | null;
  running: boolean;
  error: string | null;
  progress: unknown;
}

/**
 * Runs a simulation job in a Web Worker and returns its result.
 *
 * Widgets in this book train real algorithms, and some of them take a second.
 * Doing that on the main thread would freeze the page mid-drag, so the work
 * goes to a worker and the UI stays responsive while it runs. Results arrive
 * whole; progress messages arrive as the job advances.
 *
 * The worker is created lazily on first use and torn down with the component,
 * so a reader who never scrolls to a widget never pays for it.
 */
export function useSimulation<T>(job: JobName, params: Record<string, unknown>, enabled = true) {
  const [state, setState] = useState<State<T>>({
    data: null,
    running: false,
    error: null,
    progress: null,
  });

  const workerRef = useRef<Worker | null>(null);
  const requestId = useRef(0);
  // Serialized params, so the effect re-runs on value changes rather than
  // on every new object identity.
  const key = JSON.stringify(params);

  useEffect(() => {
    if (!enabled) return;
    if (typeof window === 'undefined' || typeof Worker === 'undefined') return;

    if (!workerRef.current) {
      workerRef.current = new Worker(new URL('./worker.ts', import.meta.url), {
        type: 'module',
      });
    }
    const worker = workerRef.current;
    const id = ++requestId.current;

    setState((s) => ({ ...s, running: true, error: null }));

    const onMessage = (e: MessageEvent) => {
      const msg = e.data as { id: number; type: string; payload: unknown };
      if (msg.id !== requestId.current) return; // a stale run; ignore it
      if (msg.type === 'progress') {
        setState((s) => ({ ...s, progress: msg.payload }));
      } else if (msg.type === 'done') {
        setState({ data: msg.payload as T, running: false, error: null, progress: null });
      } else if (msg.type === 'error') {
        setState((s) => ({ ...s, running: false, error: String(msg.payload) }));
      }
    };

    worker.addEventListener('message', onMessage);
    worker.postMessage({ id, job, params: JSON.parse(key) });

    return () => worker.removeEventListener('message', onMessage);
  }, [job, key, enabled]);

  useEffect(
    () => () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    },
    [],
  );

  return state;
}

/**
 * Debounces rapidly-changing values, so dragging a slider queues one simulation
 * when the reader settles rather than one per animation frame.
 */
export function useDebounced<T>(value: T, delay = 260): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return debounced;
}

/** True when the reader has asked the OS to reduce motion. */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(mq.matches);
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

/**
 * Widget state that survives in the URL, so a configuration can be linked to.
 *
 * Chapter prose can point at "the dashboard at γ = 0.99", and a reader who
 * finds something interesting can send someone else the exact view. Values are
 * namespaced by widget id and only written once the reader changes something,
 * so a clean URL stays clean.
 */
export function useWidgetState<T extends Record<string, number | string | boolean>>(
  widgetId: string,
  defaults: T,
): [T, (patch: Partial<T> | ((prev: T) => Partial<T>)) => void, () => void] {
  const [value, setValue] = useState<T>(defaults);
  const hydrated = useRef(false);

  // Read once on mount.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const params = new URLSearchParams(window.location.search);
    const raw = params.get(widgetId);
    if (raw) {
      try {
        const parsed = JSON.parse(decodeURIComponent(raw));
        setValue((v) => ({ ...v, ...parsed }));
      } catch {
        /* a malformed link should not break the page */
      }
    }
    hydrated.current = true;
  }, [widgetId]);

  const update = useCallback(
    (patch: Partial<T> | ((prev: T) => Partial<T>)) => {
      setValue((prev) => {
        // The functional form matters for rapid successive edits — painting
        // cells on a grid, say — where a patch computed from a captured value
        // would be stale by the time React applies it.
        const resolved = typeof patch === 'function' ? patch(prev) : patch;
        const next = { ...prev, ...resolved };
        if (hydrated.current && typeof window !== 'undefined') {
          const params = new URLSearchParams(window.location.search);
          // Only record what differs from the book's default.
          const diff: Record<string, unknown> = {};
          for (const k of Object.keys(next)) {
            if (next[k] !== defaults[k]) diff[k] = next[k];
          }
          if (Object.keys(diff).length === 0) params.delete(widgetId);
          else params.set(widgetId, encodeURIComponent(JSON.stringify(diff)));
          const qs = params.toString();
          window.history.replaceState(
            null,
            '',
            `${window.location.pathname}${qs ? `?${qs}` : ''}${window.location.hash}`,
          );
        }
        return next;
      });
    },
    [defaults, widgetId],
  );

  const reset = useCallback(() => update(defaults), [defaults, update]);

  return [value, update, reset];
}
