'use client';

import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import { Menu, Moon, Sun, X } from 'lucide-react';
import { SiteHeader, type BookLink } from './SiteHeader';
import { useTheme } from '@/components/layout/ThemeProvider';
import { ChapterList } from '@/components/layout/ChapterList';

const BOOK_LINKS: BookLink[] = [
  { label: 'Contents', href: '/chapters', exact: true },
  { label: 'Method', href: '/about', exact: true },
];

/**
 * Routes that open on a halftone banner. On those the bar starts transparent
 * over the dark plate; everywhere else it is solid from the first frame,
 * because the reading surface behind it is white.
 */
const BANNER_ROUTES = new Set(['/', '/chapters', '/about']);

/**
 * The Shannon bar, wired to the things this book needs it to carry: the
 * chapter drawer on narrow screens, the theme switch, and the reading-progress
 * hairline along its bottom edge.
 */
export function BookHeader() {
  const { mode, toggle, ready } = useTheme();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [progress, setProgress] = useState(0);
  const pathname = usePathname();

  // `trailingSlash` is on, so routes arrive as `/chapters/` — normalise before
  // matching or every comparison here misses.
  const route = pathname.replace(/\/+$/, '') || '/';
  const overlay = BANNER_ROUTES.has(route);
  const onChapter = route.startsWith('/chapters/');

  useEffect(() => setDrawerOpen(false), [pathname]);

  useEffect(() => {
    if (!onChapter) {
      setProgress(0);
      return;
    }
    const onScroll = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      setProgress(max > 0 ? Math.min(100, (window.scrollY / max) * 100) : 0);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, [onChapter, pathname]);

  return (
    <>
      <SiteHeader
        bookLinks={BOOK_LINKS}
        overlay={overlay}
        leading={
          <button
            type="button"
            onClick={() => setDrawerOpen((v) => !v)}
            className="sd-nav-icon lg:hidden"
            aria-label={drawerOpen ? 'Close chapter list' : 'Open chapter list'}
            aria-expanded={drawerOpen}
          >
            {drawerOpen ? <X size={16} /> : <Menu size={16} />}
          </button>
        }
        trailing={
          <button
            type="button"
            onClick={toggle}
            className="sd-nav-icon"
            aria-label={`Switch to ${mode === 'light' ? 'dark' : 'light'} theme`}
          >
            {ready && mode === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
          </button>
        }
        under={
          onChapter ? (
            <div
              className="h-0.5 bg-accent transition-[width] duration-150"
              style={{ width: `${progress}%` }}
              role="progressbar"
              aria-label="Reading progress"
              aria-valuenow={Math.round(progress)}
              aria-valuemin={0}
              aria-valuemax={100}
            />
          ) : null
        }
      />

      {drawerOpen && (
        <div
          className="fixed inset-0 z-30 overflow-y-auto bg-surface px-4 pb-10 lg:hidden"
          style={{ paddingTop: 'calc(var(--sd-nav-h) + 24px)' }}
        >
          <ChapterList />
        </div>
      )}
    </>
  );
}
