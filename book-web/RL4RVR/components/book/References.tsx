import { ExternalLink } from 'lucide-react';

export interface Reference {
  /** Citation key used inline, e.g. "Kober 2013". */
  key: string;
  authors: string;
  year: number | string;
  title: string;
  venue?: string;
  url?: string;
  /** Marks one of the book's four baseline works. */
  baseline?: boolean;
  /** What this chapter draws from it. */
  note?: string;
}

/**
 * Chapter bibliography. Baseline references are listed first and flagged, so a
 * reader can always see which claims trace to the four foundational works and
 * which are modernizations layered on top.
 */
export function References({ items }: { items: Reference[] }) {
  const baselines = items.filter((r) => r.baseline);
  const modern = items.filter((r) => !r.baseline);

  return (
    <section className="my-8 space-y-5">
      {baselines.length > 0 && (
        <div>
          <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-muted">
            Baseline references
          </h3>
          <ul className="space-y-2.5">
            {baselines.map((r) => (
              <ReferenceRow key={r.key} r={r} />
            ))}
          </ul>
        </div>
      )}
      {modern.length > 0 && (
        <div>
          <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-muted">
            Further reading &amp; modern sources
          </h3>
          <ul className="space-y-2.5">
            {modern.map((r) => (
              <ReferenceRow key={r.key} r={r} />
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function ReferenceRow({ r }: { r: Reference }) {
  return (
    <li className="border-l-2 border-hairline pl-3 text-[13.5px] leading-relaxed">
      <span className="text-ink-secondary">{r.authors}</span>{' '}
      <span className="text-ink-muted">({r.year}).</span>{' '}
      <span className="font-medium text-ink">{r.title}</span>
      {r.venue ? <span className="italic text-ink-secondary">. {r.venue}</span> : null}
      {r.url ? (
        <>
          {' '}
          <a
            href={r.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-0.5 text-accent no-underline hover:underline"
          >
            link
            <ExternalLink size={11} aria-hidden />
          </a>
        </>
      ) : null}
      {r.note ? <div className="mt-0.5 text-[12.5px] text-ink-muted">{r.note}</div> : null}
    </li>
  );
}
