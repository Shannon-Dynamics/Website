import type { MetadataRoute } from 'next';
import { CHAPTERS } from '@/lib/chapters';

export const dynamic = 'force-static';

const ORIGIN = 'https://rl4rvr.shannon.id';

// Built from the same CHAPTERS list the chapter route prerenders from.
export default function sitemap(): MetadataRoute.Sitemap {
  return ['/', '/chapters/', '/about/', ...CHAPTERS.map((c) => `/chapters/${c.slug}/`)].map(
    (path) => ({ url: `${ORIGIN}${path}` }),
  );
}
