'use client';

import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import type { Mode } from '@/lib/theme';

interface ThemeContextValue {
  mode: Mode;
  toggle: () => void;
  /** False until the client has resolved the stored/OS preference. */
  ready: boolean;
}

const ThemeContext = createContext<ThemeContextValue>({
  mode: 'light',
  toggle: () => {},
  ready: false,
});

export const useTheme = () => useContext(ThemeContext);

/**
 * Reads the persisted preference (falling back to the OS setting), stamps
 * `data-theme` on <html>, and exposes the resolved mode to charts — Nivo needs
 * concrete color values, so every visualisation subscribes to this.
 */
export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<Mode>('light');
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const stored = window.localStorage.getItem('rl4r-theme') as Mode | null;
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initial: Mode = stored ?? (prefersDark ? 'dark' : 'light');
    setMode(initial);
    document.documentElement.setAttribute('data-theme', initial);
    setReady(true);
  }, []);

  const toggle = useCallback(() => {
    setMode((prev) => {
      const next: Mode = prev === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      window.localStorage.setItem('rl4r-theme', next);
      return next;
    });
  }, []);

  return (
    <ThemeContext.Provider value={{ mode, toggle, ready }}>{children}</ThemeContext.Provider>
  );
}

/**
 * Inline script that applies the theme before first paint, so a dark-mode
 * reader never sees a white flash.
 */
export const themeScript = `
(function() {
  try {
    var stored = localStorage.getItem('rl4r-theme');
    var dark = stored ? stored === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  } catch (e) {}
})();
`;
