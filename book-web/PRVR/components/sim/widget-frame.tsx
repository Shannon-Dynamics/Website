import type { ReactNode } from 'react';
import { ColorKey } from '@/components/book/color-key';

export interface WidgetFrameProps {
  /** Widget id from the chapter design docs, e.g. "w5.1". */
  id: string;
  title: string;
  /** One line: what the reader should come away understanding. */
  teaches?: string;
  /** Which role colors appear in this widget. */
  colorKey?: ('prior' | 'prediction' | 'measurement' | 'posterior' | 'truth')[];
  children: ReactNode;
  /** Caption beneath the widget: how to read it, what to try. */
  caption?: ReactNode;
  /** Use the full width of the reading column rather than the prose measure. */
  wide?: boolean;
}

/**
 * The chrome around every interactive figure in the book.
 *
 * Deliberately quiet: a hairline border, an id in mono, the title, and one line
 * of intent. The simulation is the thing the reader looks at, so the frame does
 * not compete with it.
 */
export function WidgetFrame({
  id,
  title,
  teaches,
  colorKey,
  children,
  caption,
  wide = false,
}: WidgetFrameProps) {
  return (
    <figure
      id={id}
      className={`not-prose my-8 ${wide ? 'pr-figure-wide' : ''}`}
      aria-labelledby={`${id}-title`}
    >
      <div className="overflow-hidden rounded-md border border-fd-border bg-fd-card">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-fd-border px-3 py-2">
          <span className="font-mono text-[0.7rem] font-medium text-fd-primary">{id}</span>
          <span id={`${id}-title`} className="font-display text-sm font-semibold">
            {title}
          </span>
          {teaches ? (
            <span className="w-full font-ui text-xs text-fd-muted-foreground sm:w-auto sm:flex-1">
              {teaches}
            </span>
          ) : null}
        </div>

        {children}

        {colorKey && colorKey.length > 0 ? (
          <div className="border-t border-fd-border px-3 py-2">
            <ColorKey items={colorKey} />
          </div>
        ) : null}
      </div>

      {caption ? (
        <figcaption className="pr-caption mt-2 font-ui text-[0.8rem] leading-relaxed text-fd-muted-foreground">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}

/** A static fallback shown where a simulation would be, when JS is unavailable. */
export function StaticFallback({ children }: { children: ReactNode }) {
  return (
    <noscript>
      <div className="border-t border-fd-border p-3 font-ui text-xs text-fd-muted-foreground">
        {children}
      </div>
    </noscript>
  );
}
