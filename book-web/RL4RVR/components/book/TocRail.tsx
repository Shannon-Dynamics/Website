'use client';

import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';

/** In-page section rail with scroll-spy. */
export function TocRail({
  headings,
}: {
  headings: Array<{ id: string; text: string; level: 2 | 3 }>;
}) {
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    if (headings.length === 0) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { rootMargin: '-80px 0px -70% 0px', threshold: 0 },
    );

    for (const h of headings) {
      const el = document.getElementById(h.id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, [headings]);

  if (headings.length === 0) return null;

  return (
    <nav aria-label="On this page" className="text-[12.5px]">
      <p className="mb-2 text-[10.5px] font-semibold uppercase tracking-[0.09em] text-ink-muted">
        On this page
      </p>
      <ul className="space-y-px border-l border-hairline">
        {headings.map((h) => (
          <li key={h.id}>
            <a
              href={`#${h.id}`}
              className={cn(
                '-ml-px block border-l-2 py-1 no-underline transition-colors',
                h.level === 3 ? 'pl-5' : 'pl-3',
                active === h.id
                  ? 'border-series-1 font-medium text-ink'
                  : 'border-transparent text-ink-muted hover:border-baseline hover:text-ink-secondary',
              )}
            >
              {h.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
