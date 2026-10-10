import { docs } from '@/.source/server';
import { loader } from 'fumadocs-core/source';

export const source = loader({
  baseUrl: '/chapters',
  source: docs.toFumadocsSource(),
});

/**
 * The loader strips trailing slashes from page URLs (it normalises any `url`
 * option too), but next.config sets trailingSlash, so a slashless URL costs a
 * 308. next/link and the metadata API add the slash on their own; anything else
 * that hands a page URL to the outside world — the sitemap, search results —
 * goes through this.
 */
export const withTrailingSlash = (url: string) => (url.endsWith('/') ? url : `${url}/`);
