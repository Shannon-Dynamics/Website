import type { ReactNode } from 'react';

export interface FigureProps {
  children: ReactNode;
  caption?: string;
  /** Anchor id, so prose can link to "#fig-belief-drift". */
  id?: string;
  /** Let the figure bleed past the reading column on wide viewports. */
  wide?: boolean;
}

export function Figure({ children, caption, id, wide }: FigureProps) {
  return (
    <figure
      id={id}
      className={`my-8 ${id ? 'scroll-mt-24' : ''} ${
        wide ? 'pr-figure-wide' : ''
      }`}
    >
      {/* Wide content scrolls here, never on the page. */}
      <div className="max-w-full overflow-x-auto">{children}</div>

      {caption ? (
        // The number comes from a CSS counter incremented per captioned figure,
        // so figures renumber themselves as chapters are edited. No reset is
        // declared: the counter is implicitly created on the root, and each
        // chapter is its own document.
        <figcaption className="mt-2.5 flex gap-2.5 border-t border-fd-border pt-2 font-ui text-[0.8125rem] leading-snug text-fd-muted-foreground [counter-increment:pr-figure]">
          <span className="shrink-0 font-mono text-[0.6875rem] font-semibold uppercase tracking-[0.12em] text-fd-primary">
            {'Figure '}
            <span aria-hidden="true" className="[&::after]:[content:counter(pr-figure)]" />
          </span>
          <span className="text-pretty">{caption}</span>
        </figcaption>
      ) : null}
    </figure>
  );
}
