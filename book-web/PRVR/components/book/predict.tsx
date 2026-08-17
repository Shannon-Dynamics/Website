'use client';

import { useState, type ReactNode } from 'react';
import { Check, X, Lightbulb, Eye } from 'lucide-react';
import { record, useOutcome } from '@/lib/explorable/progress';

/* -------------------------------------------------------------------------- */
/* Predict — commit before you look                                            */
/* -------------------------------------------------------------------------- */

export interface PredictOption {
  label: string;
  correct?: boolean;
  /** Shown after committing — why this answer is right or wrong. */
  because?: string;
}

export interface PredictProps {
  /** Stable id, namespaced by chapter: "ch05.e4". */
  id: string;
  question: string;
  options: PredictOption[];
  /** Revealed once the reader has committed. */
  children?: ReactNode;
}

/**
 * Make the reader commit to an answer before the explanation appears.
 *
 * The book's conceptual exercises say "predict what will happen, then verify",
 * which only teaches anything if the prediction is actually pinned down first —
 * a reader who reads the answer and thinks *I knew that* has learned nothing.
 * Committing costs one click and makes being wrong informative.
 */
export function Predict({ id, question, options, children }: PredictProps) {
  const outcome = useOutcome(id);
  const [choice, setChoice] = useState<number | null>(null);
  const committed = choice !== null;

  const commit = (i: number) => {
    setChoice(i);
    record(id, options[i]?.correct ? 'correct' : 'incorrect');
  };

  return (
    <div className="not-prose my-4 rounded-md border border-fd-border bg-fd-card">
      <div className="flex items-baseline gap-2 border-b border-fd-border px-3 py-2">
        <span className="eyebrow">Predict first</span>
        {outcome === 'correct' && !committed ? (
          <span className="font-mono text-[0.68rem] text-fd-primary">answered previously</span>
        ) : null}
      </div>

      <div className="px-3 py-3">
        <p className="font-prose text-[0.95rem] leading-relaxed">{question}</p>

        <div className="mt-3 flex flex-col gap-1.5">
          {options.map((opt, i) => {
            const chosen = choice === i;
            const show = committed;
            const good = Boolean(opt.correct);
            return (
              <div key={opt.label}>
                <button
                  type="button"
                  onClick={() => !committed && commit(i)}
                  disabled={committed}
                  aria-pressed={chosen}
                  className={[
                    'flex w-full items-start gap-2 rounded-sm border px-2.5 py-1.5 text-left font-ui text-[0.85rem] transition-colors',
                    committed ? 'cursor-default' : 'hover:bg-fd-accent/50',
                    show && good
                      ? 'border-fd-primary/60 bg-fd-primary/8'
                      : show && chosen
                        ? 'border-fd-border bg-fd-muted'
                        : 'border-fd-border',
                  ].join(' ')}
                >
                  <span className="mt-[3px] flex size-3.5 shrink-0 items-center justify-center">
                    {show ? (
                      good ? (
                        <Check className="size-3.5 text-fd-primary" />
                      ) : chosen ? (
                        <X className="size-3.5 text-fd-muted-foreground" />
                      ) : (
                        <span className="size-2 rounded-full border border-fd-border" />
                      )
                    ) : (
                      <span className="size-2.5 rounded-full border border-fd-muted-foreground/50" />
                    )}
                  </span>
                  <span>
                    {opt.label}
                    {show && opt.because ? (
                      <span className="mt-0.5 block text-[0.8rem] leading-snug text-fd-muted-foreground">
                        {opt.because}
                      </span>
                    ) : null}
                  </span>
                </button>
              </div>
            );
          })}
        </div>

        {committed && children ? (
          <div className="mt-3 border-t border-fd-border pt-3 font-prose text-[0.9rem] leading-relaxed">
            {children}
          </div>
        ) : null}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* CheckAnswer — a number, with tolerance                                      */
/* -------------------------------------------------------------------------- */

export interface CheckAnswerProps {
  id: string;
  /** What the reader is being asked to compute. */
  prompt: string;
  answer: number;
  /** Absolute tolerance; defaults to 1% of the answer. */
  tolerance?: number;
  unit?: string;
  /** Shown once correct, or once revealed. */
  children?: ReactNode;
}

/**
 * A numeric answer box for the worked-example exercises.
 *
 * Tolerance is explicit because most of these answers come from a derivation
 * the reader did by hand, and marking 0.4637 wrong because they wrote 0.464
 * would be teaching arithmetic pedantry rather than estimation.
 */
export function CheckAnswer({
  id,
  prompt,
  answer,
  tolerance,
  unit,
  children,
}: CheckAnswerProps) {
  const tol = tolerance ?? Math.max(Math.abs(answer) * 0.01, 1e-9);
  const [entry, setEntry] = useState('');
  const [state, setState] = useState<'idle' | 'correct' | 'incorrect' | 'revealed'>('idle');

  const submit = () => {
    const v = Number(entry.replace(/,/g, '.').trim());
    if (!Number.isFinite(v)) return;
    const ok = Math.abs(v - answer) <= tol;
    setState(ok ? 'correct' : 'incorrect');
    record(id, ok ? 'correct' : 'incorrect');
  };

  const reveal = () => {
    setState('revealed');
    record(id, 'revealed');
  };

  const solved = state === 'correct' || state === 'revealed';

  return (
    <div className="not-prose my-4 rounded-md border border-fd-border bg-fd-card px-3 py-3">
      <p className="font-prose text-[0.95rem] leading-relaxed">{prompt}</p>

      <div className="mt-2.5 flex flex-wrap items-center gap-2">
        <input
          type="text"
          inputMode="decimal"
          value={solved ? String(answer) : entry}
          disabled={solved}
          onChange={(e) => setEntry(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && submit()}
          placeholder="your answer"
          aria-label={prompt}
          className="w-32 rounded-sm border border-fd-border bg-fd-background px-2 py-1 font-mono text-sm tabular-nums focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-fd-primary"
        />
        {unit ? <span className="font-mono text-xs text-fd-muted-foreground">{unit}</span> : null}

        {!solved ? (
          <>
            <button
              type="button"
              onClick={submit}
              className="rounded-sm bg-fd-primary px-2.5 py-1 font-ui text-xs font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
            >
              Check
            </button>
            <button
              type="button"
              onClick={reveal}
              className="inline-flex items-center gap-1 rounded-sm border border-fd-border px-2.5 py-1 font-ui text-xs transition-colors hover:bg-fd-accent"
            >
              <Eye className="size-3" /> Reveal
            </button>
          </>
        ) : null}

        {state === 'correct' ? (
          <span className="inline-flex items-center gap-1 font-ui text-xs font-medium text-fd-primary">
            <Check className="size-3.5" /> correct
          </span>
        ) : null}
        {state === 'incorrect' ? (
          <span className="inline-flex items-center gap-1 font-ui text-xs text-fd-muted-foreground">
            <X className="size-3.5" /> not yet — within ±{tol.toPrecision(2)} counts
          </span>
        ) : null}
        {state === 'revealed' ? (
          <span className="font-ui text-xs text-fd-muted-foreground">revealed</span>
        ) : null}
      </div>

      {solved && children ? (
        <div className="mt-3 border-t border-fd-border pt-2.5 font-prose text-[0.9rem] leading-relaxed">
          {children}
        </div>
      ) : null}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Hints — one step at a time                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Progressive hints. A reader who is stuck usually needs a nudge, not the
 * answer, so they arrive one at a time and the solution stays behind the last.
 */
export function Hints({ children }: { children: ReactNode[] | ReactNode }) {
  const items = Array.isArray(children) ? children : [children];
  const [shown, setShown] = useState(0);

  return (
    <div className="not-prose my-3">
      {items.slice(0, shown).map((item, i) => (
        <div
          key={i}
          className="mb-1.5 rounded-sm border-s-2 border-fd-primary/40 bg-fd-muted/40 px-3 py-2 font-prose text-[0.87rem] leading-relaxed"
        >
          <span className="eyebrow mb-1 block">Hint {i + 1}</span>
          {item}
        </div>
      ))}
      {shown < items.length ? (
        <button
          type="button"
          onClick={() => setShown((n) => n + 1)}
          className="inline-flex items-center gap-1.5 rounded-sm border border-fd-border px-2.5 py-1 font-ui text-xs text-fd-muted-foreground transition-colors hover:bg-fd-accent hover:text-fd-foreground"
        >
          <Lightbulb className="size-3" />
          {shown === 0 ? 'Stuck? Get a hint' : `Another hint (${items.length - shown} left)`}
        </button>
      ) : null}
    </div>
  );
}
