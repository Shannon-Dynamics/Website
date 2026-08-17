import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { ALL_CHAPTERS } from '@/lib/book-structure';

/**
 * Previous / next chapter links.
 *
 * A book is read in order even when it is browsed out of order, so the end of
 * every chapter names where the argument goes next rather than just linking to
 * it — the blurb is the whole point of the component.
 */
export function ChapterNav({ chapter }: { chapter?: number }) {
  if (chapter === undefined) return null;

  const index = ALL_CHAPTERS.findIndex((c) => c.n === chapter);
  if (index === -1) return null;

  const prev = index > 0 ? ALL_CHAPTERS[index - 1] : undefined;
  const next = index < ALL_CHAPTERS.length - 1 ? ALL_CHAPTERS[index + 1] : undefined;
  if (!prev && !next) return null;

  return (
    <nav
      aria-label="Chapter navigation"
      className="not-prose mt-14 grid gap-3 border-t border-fd-border pt-6 sm:grid-cols-2"
    >
      {prev ? (
        <Link
          href={`/chapters/${prev.slug}`}
          className="group flex flex-col gap-1 rounded-md border border-fd-border p-3 transition-colors hover:bg-fd-accent/40"
        >
          <span className="eyebrow flex items-center gap-1">
            <ArrowLeft className="size-3" /> Chapter {prev.n}
          </span>
          <span className="font-ui text-sm font-medium group-hover:text-fd-primary">
            {prev.title}
          </span>
          <span className="font-prose text-[0.8rem] leading-snug text-fd-muted-foreground">
            {prev.blurb}
          </span>
        </Link>
      ) : (
        <span aria-hidden="true" />
      )}

      {next ? (
        <Link
          href={`/chapters/${next.slug}`}
          className="group flex flex-col gap-1 rounded-md border border-fd-border p-3 text-right transition-colors hover:bg-fd-accent/40"
        >
          <span className="eyebrow flex items-center justify-end gap-1">
            Chapter {next.n} <ArrowRight className="size-3" />
          </span>
          <span className="font-ui text-sm font-medium group-hover:text-fd-primary">
            {next.title}
          </span>
          <span className="font-prose text-[0.8rem] leading-snug text-fd-muted-foreground">
            {next.blurb}
          </span>
        </Link>
      ) : null}
    </nav>
  );
}
