import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import type { CSSProperties, ReactNode } from 'react';
import { source } from '@/lib/source';
import { BOOK_LINKS, bookNav } from '@/lib/nav';
import { SiteHeader } from '@/components/shannon/site-header';

/**
 * The reader.
 *
 * The Shannon bar is fixed and sits above the docs grid rather than inside it —
 * Fumadocs' own header slot is the mobile sub-bar, hidden from `md` up, so it
 * is not a place a site-wide bar can live. `--fd-banner-height` is the knob
 * Fumadocs exposes for exactly this: it offsets the sticky sidebar and table of
 * contents, and the matching padding drops the grid itself clear of the bar.
 */
export default function Layout({ children }: { children: ReactNode }) {
  return (
    <>
      <SiteHeader bookLinks={BOOK_LINKS} spacer={false} />
      <DocsLayout
        tree={source.pageTree}
        {...bookNav}
        containerProps={{
          style: {
            '--fd-banner-height': 'var(--sd-nav-h)',
            paddingTop: 'var(--sd-nav-h)',
          } as CSSProperties,
        }}
      >
        {children}
      </DocsLayout>
    </>
  );
}
