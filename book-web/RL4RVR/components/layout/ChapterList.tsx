'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { CHAPTERS, PARTS } from '@/lib/chapters';
import { cn } from '@/lib/utils';

/** Sidebar / mobile-drawer navigation over the full table of contents. */
export function ChapterList() {
  const pathname = usePathname();

  return (
    <nav aria-label="Chapters" className="space-y-5 text-[13px]">
      {PARTS.map((part) => (
        <div key={part.id}>
          <p className="mb-1.5 px-2 text-[10.5px] font-semibold uppercase tracking-[0.09em] text-ink-muted">
            Part {part.id} · {part.title}
          </p>
          <ul className="space-y-px">
            {part.chapters.map((n) => {
              const ch = CHAPTERS.find((c) => c.n === n);
              if (!ch) return null;
              const href = `/chapters/${ch.slug}`;
              const active = pathname === href;
              return (
                <li key={ch.n}>
                  <Link
                    href={href}
                    className={cn(
                      'flex gap-2.5 rounded-md px-2 py-1.5 no-underline transition-colors',
                      active
                        ? 'bg-surface-sunken font-medium text-ink'
                        : 'text-ink-secondary hover:bg-surface-sunken hover:text-ink',
                    )}
                    aria-current={active ? 'page' : undefined}
                  >
                    <span
                      className={cn(
                        'tabular w-4 shrink-0 text-right text-[11.5px]',
                        active ? 'text-accent' : 'text-ink-muted',
                      )}
                    >
                      {ch.n}
                    </span>
                    <span className="leading-snug">{ch.title}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </nav>
  );
}
