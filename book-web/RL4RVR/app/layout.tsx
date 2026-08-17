import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { ThemeProvider, themeScript } from '@/components/layout/ThemeProvider';
import { BookHeader } from '@/components/shannon/BookHeader';
import { SiteFooter } from '@/components/shannon/SiteFooter';

/**
 * The Shannon Dynamics pair. Inter carries everything the site sets in type;
 * JetBrains Mono carries code, and the mono eyebrows the chrome is built from.
 */
const inter = Inter({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  display: 'swap',
  variable: '--font-inter',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  weight: ['400', '500'],
  display: 'swap',
  variable: '--font-jetbrains-mono',
});

export const metadata: Metadata = {
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
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
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
