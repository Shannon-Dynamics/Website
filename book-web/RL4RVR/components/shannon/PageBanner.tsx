import { Fragment, type ReactNode } from 'react';
import { Halftone } from './Halftone';
import { SHANNON_BOOKS, SHANNON_HOME, SHANNON_LIBRARY, SHANNON_SITE } from '@/lib/shannon';

export interface Crumb {
  label: string;
  /** Absolute for the marketing site, root-relative for a route in this book. */
  href?: string;
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
 * The trail every page in this book shares: the site, its Library, the Books
 * shelf, then wherever the reader actually is.
 */
export function bookCrumb(...tail: Crumb[]): Crumb[] {
  return [
    { label: 'SHANNON', href: SHANNON_HOME },
    { label: 'LIBRARY', href: SHANNON_LIBRARY },
    { label: 'BOOKS', href: SHANNON_BOOKS },
    ...tail,
  ];
}
