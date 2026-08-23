import { createFromSource } from 'fumadocs-core/search/server';
import { source } from '@/lib/source';

/**
 * The search index, served at request time.
 *
 * This is what the static export could not have: with no server there was no
 * endpoint to query, so the search trigger was hidden (see `lib/nav.tsx`). The
 * book runs as a Next.js app now, so the index is built from the same source
 * the page tree comes from and queried over HTTP.
 */
export const { GET } = createFromSource(source);
