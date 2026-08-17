import { Cpu, Eye, Sigma } from 'lucide-react';
import { cn } from '@/lib/utils';

type Layer = 'F' | 'C' | 'P';

const LAYER: Record<Layer, { label: string; icon: React.ElementType; color: string }> = {
  F: { label: 'Foundation', icon: Sigma, color: 'var(--series-1)' },
  C: { label: 'Conceptual', icon: Eye, color: 'var(--series-3)' },
  P: { label: 'Practical', icon: Cpu, color: 'var(--series-2)' },
};

export interface Exercise {
  layer: Layer;
  title: string;
  body: string;
  /** Optional difficulty, 1–3. */
  difficulty?: 1 | 2 | 3;
}

/**
 * End-of-chapter exercises. Each is tagged by FCP layer with an icon + label,
 * so the tag never depends on color alone.
 */
export function Exercises({ items }: { items: Exercise[] }) {
  return (
    <section className="my-8">
      <ol className="space-y-3">
        {items.map((ex, i) => {
          const spec = LAYER[ex.layer];
          const Icon = spec.icon;
          return (
            <li
              key={i}
              className="rounded-xl border border-hairline bg-surface px-4 py-3.5"
              style={{ borderLeft: `3px solid ${spec.color}` }}
            >
              <div className="mb-1 flex flex-wrap items-center gap-2">
                <span className="tabular text-[12px] font-semibold text-ink-muted">
                  {String(i + 1).padStart(2, '0')}
                </span>
                <span
                  className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10.5px] font-semibold uppercase tracking-[0.06em]"
                  style={{
                    color: spec.color,
                    background: `color-mix(in srgb, ${spec.color} 12%, transparent)`,
                  }}
                >
                  <Icon size={11} aria-hidden />
                  {spec.label}
                </span>
                {ex.difficulty ? (
                  <span
                    className="text-[11px] text-ink-muted"
                    title={`Difficulty ${ex.difficulty} of 3`}
                  >
                    {'●'.repeat(ex.difficulty)}
                    <span className="opacity-30">{'●'.repeat(3 - ex.difficulty)}</span>
                  </span>
                ) : null}
                <span className="text-[14px] font-semibold text-ink">{ex.title}</span>
              </div>
              <p className="text-[14.5px] leading-relaxed text-ink-secondary">{ex.body}</p>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

/** A coding task with explicit deliverables — the chapter's "build this" block. */
export function CodingTask({
  title,
  crate,
  children,
  deliverables,
}: {
  title: string;
  crate?: string;
  children: React.ReactNode;
  deliverables?: string[];
}) {
  return (
    <section
      className="my-7 rounded-xl border border-hairline bg-surface-raised px-4 py-4"
      style={{ borderLeft: '3px solid var(--series-2)' }}
    >
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="flex items-center gap-1.5 rounded bg-series-2 px-2 py-0.5 text-[10.5px] font-semibold uppercase tracking-[0.07em] text-white">
          <Cpu size={11} aria-hidden />
          Coding task
        </span>
        {crate ? (
          <code className="rounded bg-surface-sunken px-1.5 py-0.5 font-mono text-[11.5px] text-ink-secondary">
            {crate}
          </code>
        ) : null}
      </div>
      <h4 className="mb-1.5 text-[15.5px] font-semibold tracking-tight text-ink">{title}</h4>
      <div className="text-[14.5px] leading-relaxed text-ink-secondary [&>p:last-child]:mb-0">
        {children}
      </div>
      {deliverables?.length ? (
        <div className="mt-3 border-t border-hairline pt-2.5">
          <p className="mb-1 text-[11px] font-semibold uppercase tracking-[0.06em] text-ink-muted">
            Deliverables
          </p>
          <ul className="space-y-1 text-[13.5px] text-ink-secondary">
            {deliverables.map((d, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-series-2" aria-hidden>
                  ▸
                </span>
                <span>{d}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

export { cn };
