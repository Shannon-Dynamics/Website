import type { ReactNode } from 'react';

export interface ReferencesProps {
  children: ReactNode;
}

export interface ReferenceProps {
  /** Optional trailing commentary; rendered under the entry. */
  children?: ReactNode;
  /** Citation key, e.g. "thrun2005". Replaces the number in the gutter and
   *  anchors the entry at `#ref-<id>`. */
  id?: string;
  authors: string;
  year: string | number;
  title: string;
  venue?: string;
  url?: string;
  doi?: string;
  /** Why this reference matters — one italic line. */
  note?: string;
}

export function References({ children }: ReferencesProps) {
  return (
    <ol
      // Keyed entries drop their marker, which costs list semantics in Safari;
      // the explicit role keeps them.
      role="list"
      className="my-8 list-decimal ps-10 marker:font-mono marker:text-[0.75rem] marker:text-fd-muted-foreground"
    >
      {children}
    </ol>
  );
}

/** DOIs may arrive bare ("10.1000/xyz") or as a resolver URL. */
function doiHref(doi: string): string {
  return /^https?:\/\//.test(doi) ? doi : `https://doi.org/${doi}`;
}

function doiLabel(doi: string): string {
  return doi.replace(/^https?:\/\/(dx\.)?doi\.org\//, '');
}

export function Reference({
  children,
  id,
  authors,
  year,
  title,
  venue,
  url,
  doi,
  note,
}: ReferenceProps) {
  return (
    <li
      id={id ? `ref-${id}` : undefined}
      // A keyed entry prints its key in the gutter instead of an ordinal, so the
      // list marker is suppressed for that item only.
      className={`relative my-0 scroll-mt-24 py-1.5 font-prose text-[0.95rem] leading-[1.6] ${
        id ? 'list-none' : ''
      }`}
    >
      {id ? (
        <span
          aria-hidden="true"
          className="absolute top-[0.35em] -start-10 w-8 text-end font-mono text-[0.7rem] text-fd-muted-foreground"
        >
          {id}
        </span>
      ) : null}

      <span className="text-fd-foreground">{authors}</span>{' '}
      <span className="font-mono text-[0.85em] text-fd-muted-foreground">({year})</span>{' '}
      <span className="italic">{title}.</span>
      {venue ? <span className="text-fd-muted-foreground"> {venue}.</span> : null}

      {url || doi ? (
        <span className="ms-1 inline-flex flex-wrap gap-x-3">
          {url ? (
            <a
              href={url}
              target="_blank"
              rel="noreferrer"
              className="font-mono text-[0.75rem] break-words text-fd-primary underline decoration-fd-primary/30 underline-offset-2 transition-colors hover:decoration-fd-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fd-ring"
            >
              link<span className="sr-only"> to {title} (opens in a new tab)</span>
            </a>
          ) : null}
          {doi ? (
            <a
              href={doiHref(doi)}
              target="_blank"
              rel="noreferrer"
              className="font-mono text-[0.75rem] break-words text-fd-primary underline decoration-fd-primary/30 underline-offset-2 transition-colors hover:decoration-fd-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fd-ring"
            >
              doi:{doiLabel(doi)}
              <span className="sr-only"> (opens in a new tab)</span>
            </a>
          ) : null}
        </span>
      ) : null}

      {note ? (
        <p className="mt-1 mb-0 font-ui text-[0.8rem] leading-snug italic text-fd-muted-foreground">
          {note}
        </p>
      ) : null}

      {children ? (
        <div className="mt-1 text-[0.9rem] text-fd-muted-foreground [&>:first-child]:mt-0 [&>:last-child]:mb-0">
          {children}
        </div>
      ) : null}
    </li>
  );
}
