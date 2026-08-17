'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import {
  LIBRARY_LINKS,
  SHANNON_CAPABILITIES,
  SHANNON_CONTACT,
  SHANNON_ECOSYSTEM,
  SHANNON_HOME,
  SHANNON_LIBRARY,
  SHANNON_PRODUCTS,
  asset,
} from '@/lib/shannon';

export interface BookLink {
  label: string;
  href: string;
  /** Match the path exactly rather than as a prefix. */
  exact?: boolean;
}

interface SiteHeaderProps {
  /** The book's own routes, shown ahead of the links back out to the site. */
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
  /** Slot at the far right, before the CTA — the theme switch. */
  trailing?: ReactNode;
  /** Full-width slot pinned along the bar's bottom edge — the progress bar. */
  under?: ReactNode;
}

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
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!overlay) return;
    const onScroll = () => setStuck(window.scrollY > 40);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, [overlay]);

  // CSS handles hover and keyboard focus-within on its own; this covers only
  // what CSS cannot — a tap on touch, where there is no hover state to open it —
  // plus closing on an outside click or Escape.
  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: PointerEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMenuOpen(false);
    };
    document.addEventListener('pointerdown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [menuOpen]);

  useEffect(() => setMenuOpen(false), [pathname]);

  const isOn = (link: BookLink) =>
    link.exact ? pathname === link.href : pathname === link.href || pathname.startsWith(`${link.href}/`);

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

            <span className="sd-nav-div" aria-hidden />

            <a className="sd-nav-link" href={SHANNON_CAPABILITIES}>Capabilities</a>
            <a className="sd-nav-link" href={SHANNON_PRODUCTS}>Products</a>

            <div
              ref={menuRef}
              className={`sd-nav-item${menuOpen ? ' is-open' : ''}`}
              onPointerLeave={() => setMenuOpen(false)}
            >
              <a
                className="sd-nav-link sd-nav-link-sub"
                href={SHANNON_LIBRARY}
                aria-haspopup="true"
                aria-expanded={menuOpen}
                onClick={(e) => {
                  // Touch has no hover, so the first tap opens the menu rather
                  // than leaving the page.
                  if (e.nativeEvent.detail === 0 || window.matchMedia('(hover: hover)').matches)
                    return;
                  e.preventDefault();
                  setMenuOpen((v) => !v);
                }}
              >
                Library
                <svg
                  className="sd-nav-chev"
                  width="8"
                  height="8"
                  viewBox="0 0 8 8"
                  fill="none"
                  aria-hidden="true"
                >
                  <path
                    d="M1 2.5L4 5.5L7 2.5"
                    stroke="currentColor"
                    strokeWidth="1.3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </a>
              <div className="sd-nav-sub" role="menu" aria-label="Library">
                {LIBRARY_LINKS.map((item) => (
                  <a key={item.href} className="sd-nav-sub-link" href={item.href} role="menuitem">
                    <b>{item.title}</b>
                    <small>{item.blurb}</small>
                  </a>
                ))}
              </div>
            </div>

            <a className="sd-nav-link" href={SHANNON_ECOSYSTEM}>
              Ecosystem
            </a>
          </nav>

          <div className="sd-nav-end">
            {trailing}
            <a className="sd-nav-cta" href={SHANNON_CONTACT}>
              Contact Us <span aria-hidden="true">→</span>
            </a>
          </div>
        </div>
        {under && <div className="sd-nav-under">{under}</div>}
      </header>

      {/* the bar is fixed, so a page it does not overlay has to start below it */}
      {spacer && <div className="sd-nav-spacer" aria-hidden />}
    </>
  );
}
