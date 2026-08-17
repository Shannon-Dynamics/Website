export interface ChapterHeaderProps {
  /** 1-based chapter number; rendered zero-padded ("CHAPTER 07"). */
  chapter?: number;
  /** Part label, e.g. "Part II". */
  part?: string;
  /** Part title, e.g. "Recursive State Estimation". */
  partTitle?: string;
  difficulty?: string;
  readingTime?: string;
}

/** Hairline tick separating rail items. */
function Tick() {
  return <span aria-hidden="true" className="h-3 w-px shrink-0 bg-fd-border" />;
}

/**
 * The metadata rail that sits above the chapter title: a single hairline of
 * provenance (which chapter, which part, how hard, how long) and nothing else.
 * Renders nothing when the frontmatter carries none of it.
 */
export function ChapterHeader({
  chapter,
  part,
  partTitle,
  difficulty,
  readingTime,
}: ChapterHeaderProps) {
  const hasIdentity = chapter !== undefined || Boolean(part) || Boolean(partTitle);
  const hasMeasures = Boolean(difficulty) || Boolean(readingTime);
  if (!hasIdentity && !hasMeasures) return null;

  return (
    <div className="not-prose mb-5 flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-fd-border pb-3">
      {chapter !== undefined ? (
        <span className="font-mono text-[0.6875rem] font-medium uppercase tracking-[0.18em] text-fd-primary">
          Chapter {String(chapter).padStart(2, '0')}
        </span>
      ) : null}

      {chapter !== undefined && (part || partTitle) ? <Tick /> : null}

      {part ? <span className="eyebrow">{part}</span> : null}

      {partTitle ? (
        <span className="font-ui text-[0.8125rem] text-fd-muted-foreground">{partTitle}</span>
      ) : null}

      {hasMeasures ? (
        <span className="ms-auto flex items-center gap-x-3">
          {difficulty ? (
            <span className="rounded-[2px] border border-fd-border bg-fd-muted/60 px-1.5 py-px font-mono text-[0.65rem] uppercase tracking-[0.1em] text-fd-muted-foreground">
              <span className="sr-only">Difficulty: </span>
              {difficulty}
            </span>
          ) : null}

          {difficulty && readingTime ? <Tick /> : null}

          {readingTime ? (
            <span className="font-mono text-[0.6875rem] tabular-nums text-fd-muted-foreground">
              <span className="sr-only">Estimated reading time: </span>
              {readingTime}
            </span>
          ) : null}
        </span>
      ) : null}
    </div>
  );
}
