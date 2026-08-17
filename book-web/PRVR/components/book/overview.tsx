import type { ReactNode } from 'react';

export interface OverviewProps {
  /** Prose: what this chapter does and why. */
  children: ReactNode;
  /** Capability statements — "estimate a 1-D belief with a histogram filter". */
  goals?: string[];
  /** Assumed knowledge, rendered as terse chips. */
  prerequisites?: string[];
  /** One line naming what this chapter is built on top of. */
  builds?: string;
}

/**
 * Chapter orientation block. Prose on the left, a metadata rail on the right at
 * md+ and stacked below it on narrow screens.
 *
 * Labels are styled <p> rather than headings on purpose: headings inside
 * chapter components would inherit the display-serif h2/h3 chrome and pollute
 * the document outline, which belongs to the MDX author.
 */
export function Overview({ children, goals, prerequisites, builds }: OverviewProps) {
  const hasGoals = Boolean(goals?.length);
  const hasPrereqs = Boolean(prerequisites?.length);
  const hasRail = hasGoals || hasPrereqs || Boolean(builds);

  return (
    <section
      aria-label="Chapter overview"
      className={`my-8 border-y border-fd-border py-6 ${
        hasRail ? 'md:grid md:grid-cols-[minmax(0,1fr)_15rem] md:gap-x-10' : ''
      }`}
    >
      <div>
        <p className="eyebrow mt-0 mb-2">In this chapter</p>
        <div className="[&>:first-child]:mt-0 [&>:last-child]:mb-0">{children}</div>
      </div>

      {hasRail ? (
        <aside className="not-prose mt-6 flex flex-col gap-5 border-t border-fd-border pt-5 md:mt-0 md:border-t-0 md:border-s md:border-fd-border md:ps-8 md:pt-0">
          {hasGoals ? (
            <div>
              <p className="eyebrow">You will be able to</p>
              <ol className="mt-2 list-decimal space-y-1.5 ps-5 font-ui text-[0.8125rem] leading-snug text-fd-foreground/80 marker:font-mono marker:text-[0.6875rem] marker:text-fd-primary">
                {goals?.map((goal) => (
                  <li key={goal} className="ps-1">
                    {goal}
                  </li>
                ))}
              </ol>
            </div>
          ) : null}

          {hasPrereqs ? (
            <div>
              <p className="eyebrow">Assumes</p>
              <ul role="list" className="mt-2 flex flex-wrap gap-1.5">
                {prerequisites?.map((item) => (
                  <li
                    key={item}
                    className="rounded-[2px] border border-fd-border bg-fd-muted/60 px-1.5 py-0.5 font-mono text-[0.6875rem] leading-normal text-fd-muted-foreground"
                  >
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          {builds ? (
            <div>
              <p className="eyebrow">Builds on</p>
              <p className="mt-1.5 font-ui text-[0.8125rem] leading-snug text-fd-muted-foreground">
                {builds}
              </p>
            </div>
          ) : null}
        </aside>
      ) : null}
    </section>
  );
}
