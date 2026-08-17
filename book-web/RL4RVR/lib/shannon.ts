/**
 * Where the book sits inside Shannon Dynamics.
 *
 * The book is mounted inside the marketing site — at `/books/<slug>/` — so the
 * links back out are same-origin, and CI passes the site root in through
 * `NEXT_PUBLIC_SHANNON_SITE` (`/web-shannon-26` for a project site, empty for a
 * custom domain). The absolute fallback below keeps those links working when
 * the book is served on its own, as `npm run dev` and `npm run preview` do.
 */

const trimEnd = (s: string) => s.replace(/\/+$/, '');
const trimStart = (s: string) => s.replace(/^\/+/, '');

/**
 * The marketing site this book is a Library entry of.
 *
 * A root-relative value ('' or '/web-shannon-26') keeps every link same-origin;
 * an absolute URL is the fallback for serving the book standalone. `??` rather
 * than `||` on purpose: an empty string is the meaningful "site is at the
 * domain root" value, not a missing one.
 */
export const SHANNON_SITE = trimEnd(
  process.env.NEXT_PUBLIC_SHANNON_SITE ?? 'https://shannon-dynamics.github.io/web-shannon-26',
);

/** A URL on the marketing site. */
export const site = (path: string) => `${SHANNON_SITE}/${trimStart(path)}`;

/**
 * A URL for a file in `public/`.
 *
 * `next/link` and `next/image` apply `basePath` on their own; a plain `<img>`
 * does not, so anything served straight out of `public/` has to be prefixed by
 * hand or it 404s on a project site.
 */
export const asset = (path: string) =>
  `${trimEnd(process.env.NEXT_PUBLIC_BASE_PATH ?? '')}/${trimStart(path)}`;

export const SHANNON_HOME = site('index.html');
export const SHANNON_CAPABILITIES = site('index.html#capabilities');
export const SHANNON_PRODUCTS = site('index.html#showcase');
export const SHANNON_LIBRARY = site('index.html#library');
export const SHANNON_ECOSYSTEM = site('index.html#ecosystem');
export const SHANNON_CONTACT = site('index.html#contact');
export const SHANNON_BOOKS = site('library-books.html');
export const SHANNON_EMAIL = 'hello@shannon.id';

/**
 * The PDF edition.
 *
 * It used to hang off the static landing page this book replaced; that page is
 * now a redirect, so the download lives here — this is the only route to the
 * PDF the site still has.
 */
export const BOOK_PDF = site('book-pdf/reinforcement-learning-for-robotics.pdf');

/** The Library submenu, mirroring the marketing site's nav. */
export const LIBRARY_LINKS = [
  {
    href: site('library-publications.html'),
    title: 'Publications',
    blurb: 'Papers, preprints and field notes',
  },
  {
    href: SHANNON_BOOKS,
    title: 'Books',
    blurb: 'Technical books and guides',
  },
  {
    href: site('library-open-source.html'),
    title: 'Open Source',
    blurb: 'Repositories, guides and API reference',
  },
];
