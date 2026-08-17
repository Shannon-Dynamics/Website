import type { ReactNode } from 'react';
import { SiteHeader } from '@/components/shannon/site-header';
import { BOOK_LINKS } from '@/lib/nav';

/**
 * The book's front door. It opens on the halftone banner, so the bar starts
 * transparent over the plate and gains its surface on scroll — as every
 * sub-page on the marketing site does.
 */
export default function Layout({ children }: { children: ReactNode }) {
  return (
    <>
      <SiteHeader bookLinks={BOOK_LINKS} overlay />
      {children}
    </>
  );
}
