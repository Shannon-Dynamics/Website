import { source } from '@/lib/source';
import { notFound } from 'next/navigation';
import { getMDXComponents } from '@/mdx-components';
import { ChapterHeader } from '@/components/book/chapter-header';
import { Epigraph } from '@/components/book/epigraph';
import { ChapterNav } from '@/components/book/chapter-nav';
import { TermLinker } from '@/components/book/term-linker';

/**
 * The chapter reader.
 *
 * Rendered entirely on the server, deliberately: fumadocs' DocsPage and
 * DocsBody are client components, and anything passed through a client
 * boundary is serialised into the HTML a second time so React can hydrate it.
 * For a chapter of this book — thirty thousand words of prose and several
 * hundred pre-rendered KaTeX expressions — that duplicate was about two thirds
 * of the page weight, and the browser had to parse all of it before showing
 * anything. Hand-rolling the shell keeps the content on the server side, where
 * it is sent once.
 *
 * The interactive parts stay client components, but they are now leaves rather
 * than wrappers: <TermLinker> takes no children, and each simulation sits
 * behind a <Deferred> boundary that mounts it when the reader reaches it.
 */
export default async function Page(props: { params: Promise<{ slug?: string[] }> }) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  const MDX = page.data.body;
  const toc = page.data.toc ?? [];
  const fm = page.data as unknown as {
    chapter?: number;
    part?: string;
    partTitle?: string;
    difficulty?: string;
    readingTime?: string;
    quote?: string;
    quoteAuthor?: string;
    quoteSource?: string;
  };

  return (
    <div className="pr-chapter-shell [grid-area:main] mx-auto flex w-full min-w-0 max-w-6xl gap-10 px-4 py-8 md:px-6 xl:px-8">
      <article id="nd-page" className="prose min-w-0 flex-1">
        <ChapterHeader
          chapter={fm.chapter}
          part={fm.part}
          partTitle={fm.partTitle}
          difficulty={fm.difficulty}
          readingTime={fm.readingTime}
        />

        <h1 className="mb-2">{page.data.title}</h1>
        {page.data.description ? (
          <p className="not-prose mb-6 font-prose text-lg leading-relaxed text-fd-muted-foreground">
            {page.data.description}
          </p>
        ) : null}

        {fm.quote ? (
          <Epigraph author={fm.quoteAuthor} source={fm.quoteSource}>
            {fm.quote}
          </Epigraph>
        ) : null}

        <MDX components={getMDXComponents()} />
        <ChapterNav chapter={fm.chapter} />
      </article>

      {toc.length > 0 ? (
        <nav
          aria-label="On this page"
          className="sticky top-24 hidden h-fit w-56 shrink-0 self-start xl:block"
        >
          <p className="eyebrow mb-2">On this page</p>
          <ul className="space-y-1 border-s border-fd-border">
            {toc.map((item) => (
              <li key={item.url}>
                <a
                  href={item.url}
                  className="-ms-px block border-s border-transparent py-0.5 ps-3 font-ui text-[0.8rem] leading-snug text-fd-muted-foreground transition-colors hover:border-fd-primary hover:text-fd-foreground"
                  style={{ paddingInlineStart: `${0.75 + (item.depth - 2) * 0.6}rem` }}
                >
                  {item.title}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      ) : null}

      <TermLinker />
    </div>
  );
}

export function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata(props: { params: Promise<{ slug?: string[] }> }) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();
  return {
    title: page.data.title,
    description: page.data.description,
  };
}
