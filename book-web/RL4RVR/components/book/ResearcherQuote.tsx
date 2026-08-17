import { Quote } from 'lucide-react';

/**
 * The chapter's epigraph — a line from a leading robotics/RL researcher that
 * frames the chapter's central tension. Attribution is always complete:
 * person, affiliation, and the work it is drawn from.
 */
export function ResearcherQuote({
  text,
  author,
  affiliation,
  source,
}: {
  text: string;
  author: string;
  affiliation?: string;
  source?: string;
}) {
  return (
    <figure className="my-8 rounded-xl border border-hairline bg-surface-raised px-5 py-5 sm:px-6">
      <Quote
        size={18}
        aria-hidden
        className="mb-2 text-series-1"
        strokeWidth={2.2}
      />
      <blockquote className="text-[17px] font-medium leading-relaxed tracking-[-0.005em] text-ink">
        “{text}”
      </blockquote>
      <figcaption className="mt-3 border-t border-hairline pt-2.5 text-[12.5px] leading-relaxed text-ink-secondary">
        <span className="font-semibold text-ink">{author}</span>
        {affiliation ? <span className="text-ink-muted"> · {affiliation}</span> : null}
        {source ? <div className="mt-0.5 text-[11.5px] italic text-ink-muted">{source}</div> : null}
      </figcaption>
    </figure>
  );
}
