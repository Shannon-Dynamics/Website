'use client';

import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react';

/**
 * Visibility for the simulation inside this boundary.
 *
 * `null` means "no boundary" — a widget used outside <Deferred> behaves exactly
 * as it always did, which keeps the component usable on its own.
 */
const VisibilityContext = createContext<boolean | null>(null);

export const useWidgetVisible = () => useContext(VisibilityContext);

export interface DeferredProps {
  children: ReactNode;
  /** Reserve this much vertical space before mounting, to avoid layout jump. */
  minHeight?: number;
  /** How far ahead of the viewport to mount. */
  rootMargin?: string;
  label?: string;
}

/**
 * Mount a widget only when the reader is near it, and let it idle when they
 * scroll away.
 *
 * A chapter carries three or four simulations. Mounting all of them on page
 * load means three or four `requestAnimationFrame` loops competing for a core —
 * particle filters, occupancy grids, optimisers — while the reader is still on
 * the first paragraph. On a modest machine that is enough to make the page feel
 * broken before it has said anything.
 *
 * So: nothing runs until it is nearly on screen, and once seen, a widget stays
 * mounted (its state and seed survive scrolling) but stops stepping while it is
 * out of view.
 */
export function Deferred({
  children,
  minHeight = 420,
  rootMargin = '600px 0px',
  label = 'interactive figure',
}: DeferredProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [mounted, setMounted] = useState(false);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    // Without IntersectionObserver, show everything rather than nothing.
    if (typeof IntersectionObserver === 'undefined') {
      setMounted(true);
      setVisible(true);
      return;
    }

    // If the widget is already on screen, mount it now rather than waiting for
    // the observer's first callback. A figure the reader is looking at must
    // never sit on "Loading" because a callback was late — and this also makes
    // the behaviour deterministic for screenshot and print tooling, where the
    // observer may never fire at all.
    const rect = el.getBoundingClientRect();
    const vh = window.innerHeight || 0;
    if (rect.top < vh + 600 && rect.bottom > -600) {
      setMounted(true);
      setVisible(true);
    }

    // Two observers: a generous one decides when to mount, a tight one decides
    // when it is worth spending frames on.
    const mountObserver = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setMounted(true);
          mountObserver.disconnect();
        }
      },
      { rootMargin },
    );
    const runObserver = new IntersectionObserver(
      (entries) => setVisible(entries.some((e) => e.isIntersecting)),
      { rootMargin: '120px 0px' },
    );

    mountObserver.observe(el);
    runObserver.observe(el);
    return () => {
      mountObserver.disconnect();
      runObserver.disconnect();
    };
  }, [rootMargin]);

  return (
    <div ref={ref} className="pr-full" style={mounted ? undefined : { minHeight }}>
      {mounted ? (
        <VisibilityContext.Provider value={visible}>{children}</VisibilityContext.Provider>
      ) : (
        <div
          className="my-8 flex items-center justify-center rounded-md border border-dashed border-fd-border bg-fd-card/40"
          style={{ minHeight }}
          aria-hidden="true"
        >
          <span className="font-ui text-xs text-fd-muted-foreground">Loading {label}…</span>
        </div>
      )}
    </div>
  );
}
