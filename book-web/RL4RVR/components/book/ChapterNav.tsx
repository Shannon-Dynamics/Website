import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { neighbours } from '@/lib/chapters';

/** Previous / next chapter links at the foot of every chapter. */
export function ChapterNav({ n }: { n: number }) {
  const { prev, next } = neighbours(n);

  return (
    <nav className="mt-12 grid gap-3 border-t border-hairline pt-6 sm:grid-cols-2">
      {prev ? (
        <Link
          href={`/chapters/${prev.slug}`}
          className="group rounded-xl border border-hairline px-4 py-3 no-underline transition-colors hover:bg-surface-sunken"
        >
          <span className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-[0.06em] text-ink-muted">
            <ArrowLeft size={12} />
            Chapter {prev.n}
          </span>
          <span className="mt-0.5 block text-[14px] font-semibold leading-snug text-ink">
            {prev.title}
          </span>
        </Link>
      ) : (
        <span />
      )}

      {next ? (
        <Link
          href={`/chapters/${next.slug}`}
          className="group rounded-xl border border-hairline px-4 py-3 text-right no-underline transition-colors hover:bg-surface-sunken sm:col-start-2"
        >
          <span className="flex items-center justify-end gap-1.5 text-[11px] font-medium uppercase tracking-[0.06em] text-ink-muted">
            Chapter {next.n}
            <ArrowRight size={12} />
          </span>
          <span className="mt-0.5 block text-[14px] font-semibold leading-snug text-ink">
            {next.title}
          </span>
        </Link>
      ) : null}
    </nav>
  );
}
