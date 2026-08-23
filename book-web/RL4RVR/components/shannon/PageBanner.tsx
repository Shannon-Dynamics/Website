import Link from 'next/link';
import { Fragment, type ReactNode } from 'react';
import { Halftone } from './Halftone';

export interface Crumb {
  label: string;
  /** Root-relative for a route in this book, absolute for anything outside it. */
  href?: string;
  /** Use a plain <a> instead of Next.js <Link> — for links outside the book. */
  external?: boolean;
}

interface PageBannerProps {
  crumb: Crumb[];
  title: ReactNode;
  sub?: ReactNode;
  /** Secondary pages take a shorter plate than the book's front door. */
  small?: boolean;
}

/**
 * The dark halftone plate every Shannon sub-page opens with. It carries the
 * page's <h1>, so a page using it must not declare another one.
 */
export function PageBanner({ crumb, title, sub, small = false }: PageBannerProps) {
  return (
    <div className={`sd-banner sd-tex sd-tex-bottom${small ? ' sd-banner-sm' : ''}`}>
      <Halftone speed={1.6} />
      <div className="sd-banner-in">
        <p className="sd-crumb">
          {crumb.map((c, i) => (
            <Fragment key={`${c.label}-${i}`}>
              {c.href === undefined ? (
                <span className="on">{c.label}</span>
              ) : (c.href.startsWith('/') && !c.external) ? (
                <Link href={c.href}>{c.label}</Link>
              ) : (
                <a href={c.href}>{c.label}</a>
              )}
              {i < crumb.length - 1 && <span aria-hidden>/</span>}
            </Fragment>
          ))}
        </p>
        <h1 className="sd-banner-title">{title}</h1>
        {sub && <p className="sd-banner-sub">{sub}</p>}
      </div>
    </div>
  );
}

/**
 * The trail a page in this book shows.
 *
 * The book stands on its own domain, so the trail starts at the book itself.
 * It used to be prefixed with SHANNON / LIBRARY / BOOKS — the walk back up
 * through the marketing site the book was a directory of — and that prefix is
 * what this function existed to add. It is kept as the one place a shared
 * prefix would go again, so no page has to know whether there is one.
 */
export function bookCrumb(...tail: Crumb[]): Crumb[] {
  return [...tail];
}
