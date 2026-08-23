import type { Metadata } from 'next';
// The Shannon Dynamics pair, self-hosted through Fontsource.
//
// NOT `next/font/google`: that fetches the woff2 files from fonts.gstatic.com
// during the build, so a runner that cannot reach Google fails the deploy
// outright rather than degrading — which is exactly what happened. Fontsource
// ships the same files through npm, so the only network the build needs is the
// registry it already depends on.
//
// `index.css` is the weight-axis variable face across all subsets; the Greek
// one earns its place in a book written in γ, λ and σ.
import '@fontsource-variable/inter';
import '@fontsource-variable/jetbrains-mono';
import './globals.css';
import { ThemeProvider, themeScript } from '@/components/layout/ThemeProvider';
import { BookHeader } from '@/components/shannon/BookHeader';
import { SiteFooter } from '@/components/shannon/SiteFooter';

export const metadata: Metadata = {
  // The book has its own origin, so relative OG and canonical URLs resolve
  // against it rather than against the marketing site it used to live under.
  metadataBase: new URL('https://rl4rvr.shannon.id'),
  title: {
    default: 'Reinforcement Learning for Robotics — The FCP Way',
    template: '%s · RL for Robotics',
  },
  description:
    'An interactive web book teaching reinforcement learning for robotics through Foundation (full mathematical formalism), Conceptual (interactive simulations) and Practical (Rust implementations) layers.',
  keywords: [
    'reinforcement learning',
    'robotics',
    'deep RL',
    'Rust',
    'sim-to-real',
    'PPO',
    'SAC',
    'interactive book',
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="flex min-h-screen flex-col antialiased">
        <ThemeProvider>
          <BookHeader />
          <main id="main" className="flex-1">
            {children}
          </main>
          <SiteFooter />
        </ThemeProvider>
      </body>
    </html>
  );
}
