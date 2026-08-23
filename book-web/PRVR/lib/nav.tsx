import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import type { BookLink } from '@/components/shannon/site-header';

/**
 * The book's own routes — the whole of the top bar's navigation now that the
 * book is a site of its own rather than a directory of the marketing site.
 * `Overview` is added by the bar itself, since every book has one.
 */
export const BOOK_LINKS: BookLink[] = [
  { label: 'Chapters', href: '/chapters' },
  { label: 'Notation', href: '/notation', exact: true },
];

/**
 * What is left of Fumadocs' own chrome.
 *
 * The site bar above it carries the branding and the navigation, so Fumadocs
 * keeps only its mobile sub-bar — the one that holds the sidebar trigger — and
 * that bar just needs to say which book you are in.
 */
export const bookNav: BaseLayoutProps = {
  nav: {
    title: <span className="text-[0.9rem] font-semibold tracking-tight">Probabilistic Robotics</span>,
    url: '/',
  },
  // Search is live again: the book has a server, so `app/api/search/route.ts`
  // answers from an index built off the same source as the page tree. It was
  // disabled for as long as the book was a static export with no endpoint.
};
