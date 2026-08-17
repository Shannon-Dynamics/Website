import type { ReactNode } from 'react';

export type ExerciseLevel = 'F' | 'C' | 'P';
export type ExerciseDifficulty = 1 | 2 | 3;

export interface ExercisesProps {
  children: ReactNode;
}

export interface ExerciseProps {
  children: ReactNode;
  title?: string;
  /** F = Foundation, C = Conceptual, P = Practical. */
  level?: ExerciseLevel;
  /** 1–3, rendered as filled dots. */
  difficulty?: ExerciseDifficulty;
  /** Plain text — a nudge, not the answer. */
  hint?: ReactNode;
  solution?: ReactNode;
}

/**
 * Badge palette: three steps of the teal chrome accent (neutral → tint →
 * solid). The data palette stays reserved for data.
 */
const LEVELS: Record<ExerciseLevel, { label: string; className: string }> = {
  F: {
    label: 'Foundation',
    className: 'border-fd-border bg-fd-muted text-fd-muted-foreground',
  },
  C: {
    label: 'Conceptual',
    className: 'border-fd-primary/35 bg-fd-primary/10 text-fd-primary',
  },
  P: {
    label: 'Practical',
    className: 'border-fd-primary bg-fd-primary text-fd-primary-foreground',
  },
};

export function Exercises({ children }: ExercisesProps) {
  return (
    <ol className="my-8 list-decimal ps-9 marker:font-mono marker:text-[0.75rem] marker:text-fd-muted-foreground">
      {children}
    </ol>
  );
}

export function Exercise({
  children,
  title,
  level,
  difficulty,
  hint,
  solution,
}: ExerciseProps) {
  const badge = level ? LEVELS[level] : undefined;

  return (
    <li className="my-0 border-t border-fd-border py-4 ps-2 first:border-t-0 first:pt-1">
      {badge || difficulty || title ? (
        <div className="not-prose mb-2 flex flex-wrap items-center gap-x-2.5 gap-y-1.5">
          {badge && level ? (
            <span
              className={`inline-flex size-[1.15rem] items-center justify-center rounded-[2px] border font-mono text-[0.65rem] font-semibold ${badge.className}`}
            >
              <span className="sr-only">{badge.label} exercise</span>
              <span aria-hidden="true">{level}</span>
            </span>
          ) : null}

          {difficulty ? <DifficultyDots difficulty={difficulty} /> : null}

          {title ? (
            <span className="font-display text-[0.975rem] font-medium text-pretty text-fd-foreground">
              {title}
            </span>
          ) : null}
        </div>
      ) : null}

      <div className="[&>:first-child]:mt-0 [&>:last-child]:mb-0">{children}</div>

      {hint ? <Reveal label="Hint">{hint}</Reveal> : null}
      {solution ? <Reveal label="Solution">{solution}</Reveal> : null}
    </li>
  );
}

function DifficultyDots({ difficulty }: { difficulty: ExerciseDifficulty }) {
  return (
    <span className="inline-flex items-center gap-[3px]">
      <span className="sr-only">Difficulty {difficulty} of 3</span>
      {[1, 2, 3].map((step) => (
        <span
          key={step}
          aria-hidden="true"
          className={`size-[5px] rounded-full ${
            step <= difficulty ? 'bg-fd-muted-foreground' : 'bg-fd-border'
          }`}
        />
      ))}
    </span>
  );
}

/** Collapsible hint/solution. Native <details>, no client JavaScript. */
function Reveal({ label, children }: { label: string; children: ReactNode }) {
  return (
    <details className="group mt-3 border-s-2 border-fd-border ps-3 open:border-s-fd-primary/40">
      <summary className="inline-flex cursor-pointer list-none items-center gap-1.5 font-ui text-[0.6875rem] font-semibold uppercase tracking-[0.14em] text-fd-muted-foreground transition-colors select-none hover:text-fd-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fd-ring [&::-webkit-details-marker]:hidden">
        <svg
          aria-hidden="true"
          viewBox="0 0 12 12"
          className="size-2.5 shrink-0 transition-transform group-open:rotate-90"
        >
          <path
            d="M4 2.5 8 6l-4 3.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        {label}
      </summary>
      <p className="mt-1.5 mb-0 font-prose text-[0.95rem] leading-relaxed text-fd-muted-foreground">
        {children}
      </p>
    </details>
  );
}
