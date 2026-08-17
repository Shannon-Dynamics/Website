import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import type { BookLink } from '@/components/shannon/site-header';

/**
 * The book's own routes, shown in the Shannon bar ahead of the links back out
 * to the site.
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
  // Search needs an index served at request time; this book is a static export
  // with no such endpoint, so the trigger would open a dialog that can never
  // return a result.
  searchToggle: { enabled: false },
};
