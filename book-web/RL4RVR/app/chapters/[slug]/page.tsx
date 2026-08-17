import { notFound } from 'next/navigation';
import type { Metadata } from 'next';
import { CHAPTERS, getChapter, getPart } from '@/lib/chapters';
import { extractHeadings, loadChapter } from '@/lib/mdx';
import { MdxContent } from '@/components/book/MdxContent';
import { ResearcherQuote } from '@/components/book/ResearcherQuote';
import { ChapterNav } from '@/components/book/ChapterNav';
import { TocRail } from '@/components/book/TocRail';
import { ChapterList } from '@/components/layout/ChapterList';

export function generateStaticParams() {
  return CHAPTERS.map((c) => ({ slug: c.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const chapter = getChapter(slug);
  if (!chapter) return { title: 'Not found' };
  return {
    title: `${chapter.n}. ${chapter.title}`,
    description: chapter.blurb,
  };
}

export default async function ChapterPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const chapter = getChapter(slug);
  if (!chapter) notFound();

  const loaded = loadChapter(chapter.n, chapter.slug);
  const part = getPart(chapter.part);
  const headings = loaded ? extractHeadings(loaded.content) : [];

  return (
    <div className="mx-auto flex max-w-[1600px] gap-8 px-4 py-8">
      {/* Left rail: full contents */}
      <aside className="hidden w-60 shrink-0 lg:block">
        <div className="thin-scroll sticky top-20 max-h-[calc(100vh-6rem)] overflow-y-auto pr-2">
          <ChapterList />
        </div>
      </aside>

      {/* Chapter body */}
      <article className="min-w-0 flex-1 pb-10">
        <header className="mb-8">
          <p className="text-[11.5px] font-medium uppercase tracking-[0.09em] text-ink-muted">
            Part {part.id} · {part.title}
          </p>
          <h1 className="mt-1.5 text-[clamp(1.75rem,4vw,2.5rem)] font-bold leading-[1.15] tracking-[-0.02em] text-ink">
            <span className="mr-2.5 text-ink-muted">{chapter.n}.</span>
            {chapter.title}
          </h1>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {chapter.sources.map((s) => (
              <span
                key={s}
                className="rounded-md border border-hairline px-2 py-0.5 text-[11px] text-ink-secondary"
              >
                {s}
              </span>
            ))}
            {chapter.robots.map((r) => (
              <span
                key={r}
                className="rounded-md px-2 py-0.5 text-[11px] font-medium"
                style={{
                  color: 'var(--series-7)',
                  background: 'color-mix(in srgb, var(--series-7) 12%, transparent)',
                }}
              >
                {r}
              </span>
            ))}
          </div>
        </header>

        {loaded?.frontmatter.quote ? (
          <ResearcherQuote {...loaded.frontmatter.quote} />
        ) : null}

        <div className="prose-book max-w-none">
          {loaded ? (
            <MdxContent source={loaded.content} />
          ) : (
            <div className="rounded-xl border border-hairline bg-surface px-5 py-8 text-center">
              <p className="text-[15px] font-medium text-ink">This chapter is being written.</p>
              <p className="mx-auto mt-1.5 max-w-md text-[13.5px] leading-relaxed text-ink-secondary">
                {chapter.blurb}
              </p>
              <p className="mt-3 text-[12px] text-ink-muted">
                The full design — sections, widgets, Rust plan and exercises — lives in{' '}
                <code className="rounded bg-surface-sunken px-1.5 py-0.5 font-mono text-[11.5px]">
                  Chapter-{chapter.n}.md
                </code>
                .
              </p>
            </div>
          )}
        </div>

        <ChapterNav n={chapter.n} />
      </article>

      {/* Right rail: on this page */}
      <aside className="hidden w-52 shrink-0 xl:block">
        <div className="thin-scroll sticky top-20 max-h-[calc(100vh-6rem)] overflow-y-auto">
          <TocRail headings={headings} />
        </div>
      </aside>
    </div>
  );
}
