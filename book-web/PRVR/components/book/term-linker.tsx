'use client';

import { useEffect } from 'react';
import { setHoveredRole, type Role } from '@/lib/explorable/store';

const ROLES: Role[] = ['prior', 'prediction', 'measurement', 'posterior', 'truth'];

/**
 * Wires every colour-coded equation term on the page, once.
 *
 * The obvious implementation wraps each equation in a client component — but
 * children of a client component are serialised into the page a second time for
 * hydration, and this book's chapters carry enough rendered KaTeX that the
 * duplicate dominated the page weight. Doing it with one listener that finds
 * the spans after mount keeps every equation on the server side of the
 * boundary, where it costs nothing.
 */
export function TermLinker() {
  useEffect(() => {
    const spans: HTMLElement[] = [];
    for (const role of ROLES) {
      for (const el of document.querySelectorAll<HTMLElement>(`.term-${role}`)) {
        el.dataset.prTerm = role;
        el.tabIndex = 0;
        el.setAttribute('role', 'button');
        el.setAttribute(
          'aria-label',
          `Highlight the ${role === 'truth' ? 'ground truth' : role} in the figure`,
        );
        spans.push(el);
      }
    }
    if (!spans.length) return;

    // One delegated listener rather than five per span.
    const enter = (e: Event) => {
      const t = (e.target as HTMLElement)?.closest<HTMLElement>('[data-pr-term]');
      if (t?.dataset.prTerm) setHoveredRole(t.dataset.prTerm as Role);
    };
    const leave = () => setHoveredRole(null);

    document.addEventListener('pointerover', enter);
    document.addEventListener('pointerout', leave);
    document.addEventListener('focusin', enter);
    document.addEventListener('focusout', leave);
    return () => {
      document.removeEventListener('pointerover', enter);
      document.removeEventListener('pointerout', leave);
      document.removeEventListener('focusin', enter);
      document.removeEventListener('focusout', leave);
      setHoveredRole(null);
    };
  }, []);

  return null;
}
