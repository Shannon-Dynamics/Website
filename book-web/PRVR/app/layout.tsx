import './global.css';
import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import { RootProvider } from 'fumadocs-ui/provider/next';
import { SiteFooter } from '@/components/shannon/site-footer';

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
    default: 'Probabilistic Robotics via Rust',
    template: '%s · Probabilistic Robotics via Rust',
  },
  description:
    'An interactive web book on probabilistic robotics: rigorous mathematical foundations, ' +
    'interactive simulations for every hard idea, and implementations in Rust.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
      <body className="flex min-h-screen flex-col">
        <RootProvider>
          {children}
          <SiteFooter />
        </RootProvider>
      </body>
    </html>
  );
}
