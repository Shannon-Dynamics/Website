import type { ReactNode } from 'react';
import { cx } from './chart-frame';

export interface DashboardProps {
  children: ReactNode;
  /** Columns at the widest breakpoint; always one column on a phone. */
  columns?: 2 | 3 | 4;
  className?: string;
}

/*
 * Static class strings, not template literals: Tailwind reads source text, so a
 * computed `sm:grid-cols-${n}` would never be generated.
 */
const COLUMN_CLASS: Record<2 | 3 | 4, string> = {
  2: 'sm:grid-cols-2',
  3: 'sm:grid-cols-2 lg:grid-cols-3',
  4: 'sm:grid-cols-2 lg:grid-cols-4',
};

/** The grid a chapter's readouts and charts sit in. */
export function Dashboard({ children, columns = 3, className }: DashboardProps) {
  return (
    <div className={cx('not-prose grid grid-cols-1 items-start gap-3', COLUMN_CLASS[columns], className)}>
      {children}
    </div>
  );
}

export interface DashboardPanelProps {
  title: string;
  children: ReactNode;
  /** Controls for this panel — a play button, a reset, a mode switch. */
  actions?: ReactNode;
  /** How many dashboard columns the panel occupies. */
  span?: 1 | 2 | 'full';
  className?: string;
}

const SPAN_CLASS: Record<1 | 2 | 'full', string> = {
  1: '',
  2: 'sm:col-span-2',
  full: 'col-span-full',
};

/**
 * A titled instrument panel: hairline border, eyebrow title, optional controls.
 *
 * The title is a labelled `section` rather than a heading — inside a chapter,
 * `#nd-page h3` would drag it into the display face and into the table of
 * contents, neither of which a widget wants.
 */
export function DashboardPanel({ title, children, actions, span = 1, className }: DashboardPanelProps) {
  return (
    <section
      aria-label={title}
      className={cx(
        'not-prose flex min-w-0 flex-col gap-3 rounded-lg border border-fd-border bg-fd-card p-3.5',
        SPAN_CLASS[span],
        className,
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <p className="eyebrow m-0">{title}</p>
        {actions ? <div className="flex shrink-0 items-center gap-1.5">{actions}</div> : null}
      </div>
      <div className="min-w-0">{children}</div>
    </section>
  );
}
