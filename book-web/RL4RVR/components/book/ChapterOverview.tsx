import { Cpu, Eye, Sigma, Target } from 'lucide-react';

/**
 * The chapter opener: what this chapter is about, what the reader will be able
 * to do afterwards, and how the three FCP layers divide the work.
 */
export function ChapterOverview({
  summary,
  outcomes,
  foundation,
  conceptual,
  practical,
}: {
  summary: string;
  outcomes: string[];
  foundation: string;
  conceptual: string;
  practical: string;
}) {
  const layers = [
    { key: 'F', label: 'Foundation', icon: Sigma, color: 'var(--series-1)', text: foundation },
    { key: 'C', label: 'Conceptual', icon: Eye, color: 'var(--series-3)', text: conceptual },
    { key: 'P', label: 'Practical', icon: Cpu, color: 'var(--series-2)', text: practical },
  ];

  return (
    <section className="my-7">
      <p className="text-[16.5px] leading-relaxed text-ink-secondary">{summary}</p>

      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        {layers.map((l) => {
          const Icon = l.icon;
          return (
            <div
              key={l.key}
              className="rounded-xl border border-hairline bg-surface px-3.5 py-3"
              style={{ borderTop: `2px solid ${l.color}` }}
            >
              <p
                className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.07em]"
                style={{ color: l.color }}
              >
                <Icon size={12} aria-hidden />
                {l.label}
              </p>
              <p className="text-[13.5px] leading-relaxed text-ink-secondary">{l.text}</p>
            </div>
          );
        })}
      </div>

      <div className="mt-5 rounded-xl border border-hairline bg-surface-raised px-4 py-3.5">
        <p className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.07em] text-ink-muted">
          <Target size={12} aria-hidden />
          After this chapter you can
        </p>
        <ul className="space-y-1.5">
          {outcomes.map((o, i) => (
            <li key={i} className="flex gap-2 text-[14.5px] leading-relaxed text-ink-secondary">
              <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-series-1" aria-hidden />
              <span>{o}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
