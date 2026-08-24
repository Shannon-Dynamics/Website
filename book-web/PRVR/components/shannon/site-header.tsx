'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState, type ReactNode } from 'react';
import { BOOK_HOME, SHANNON_HOME, asset } from '@/lib/shannon';

export interface BookLink {
  label: string;
  href: string;
  /** Match the path exactly rather than as a prefix. */
  exact?: boolean;
}

interface SiteHeaderProps {
  /** The book's own routes — the whole of the bar's navigation. */
  bookLinks: BookLink[];
  /**
   * True on pages that open with a banner. The bar then starts transparent over
   * the dark plate and gains its surface on scroll, as on the marketing site.
   * On a page without a banner it stays solid — the logo and links are
   * white-only assets and would be invisible on the white reading surface.
   */
  overlay?: boolean;
  /**
   * Whether to reserve the bar's height in the flow. Defaults to true on a
   * solid bar, which is what a normal page wants; pass false where the layout
   * underneath already offsets itself (the docs grid does, through
   * `--fd-banner-height`).
   */
  spacer?: boolean;
  /** Slot at the far left, before the logo — the sidebar trigger. */
  leading?: ReactNode;
  /** Slot at the far right — the theme switch. */
  trailing?: ReactNode;
  /** Full-width slot pinned along the bar's bottom edge — the progress bar. */
  under?: ReactNode;
}

/**
 * The book's top bar.
 *
 * The book reads as its own site on its own subdomain, so the bar carries only
 * the book's routes. The wordmark is the one way back to Shannon Dynamics —
 * everything else that used to sit here (Capabilities, Products, the Library
 * submenu, Ecosystem, the Contact CTA) belongs to the marketing site's own
 * navigation and only made sense while the book was a directory inside it.
 */
export function SiteHeader({
  bookLinks,
  overlay = false,
  spacer = !overlay,
  leading,
  trailing,
  under,
}: SiteHeaderProps) {
  const pathname = usePathname();
  const [stuck, setStuck] = useState(false);

  useEffect(() => {
    if (!overlay) return;
    const onScroll = () => setStuck(window.scrollY > 40);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, [overlay]);

  // `trailingSlash` is on, so routes arrive as `/chapters/`; normalise both
  // sides or every comparison here misses.
  const route = pathname.replace(/\/+$/, '') || '/';
  const isOn = (link: BookLink) => {
    const href = link.href.replace(/\/+$/, '') || '/';
    return link.exact ? route === href : route === href || route.startsWith(`${href}/`);
  };

  return (
    <>
      <header
        className={['sd-nav', overlay ? (stuck ? 'is-stuck' : '') : 'is-solid']
          .filter(Boolean)
          .join(' ')}
      >
        <div className="sd-nav-in">
          <div className="sd-nav-start">
            {leading}
            <a className="sd-nav-logo" href={SHANNON_HOME}>
              <img src={asset('shannon/logo-horizontal-white.png')} alt="Shannon Dynamics" />
            </a>
          </div>

          <nav className="sd-nav-links sd-glass-dark">
            <Link
              href={BOOK_HOME}
              className={`sd-nav-link${route === '/' ? ' is-on' : ''}`}
              aria-current={route === '/' ? 'page' : undefined}
            >
              Overview
            </Link>
            {bookLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`sd-nav-link${isOn(link) ? ' is-on' : ''}`}
                aria-current={isOn(link) ? 'page' : undefined}
              >
                {link.label}
              </Link>
            ))}
          </nav>

          <div className="sd-nav-end">{trailing}</div>
        </div>
        {under && <div className="sd-nav-under">{under}</div>}
      </header>

      {/* the bar is fixed, so a page it does not overlay has to start below it */}
      {spacer && <div className="sd-nav-spacer" aria-hidden />}
    </>
  );
}
