/**
 * Book-level consistency gate.
 *
 * `npm run build` proves the site compiles; `npm run check` proves the
 * algorithms are right. This proves the *book* is coherent: every chapter
 * present and correctly labelled, every widget id matching its chapter, every
 * cross-reference resolving, every citation carrying a verifiable locator, the
 * color code never bypassed with a raw hex, and every internal link routed
 * through next/link so it survives being served from a sub-path.
 */

import { readFileSync, readdirSync, existsSync, statSync } from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const WEB = join(dirname(fileURLToPath(import.meta.url)), '..');
const CHAPTERS_DIR = join(WEB, 'content', 'chapters');

// Parsed out of book-structure.ts so there is exactly one source of truth.
const structure = readFileSync(join(WEB, 'lib', 'book-structure.ts'), 'utf8');
const SLUGS = [...structure.matchAll(/slug:\s*'([^']+)'/g)].map((m) => m[1]);
const NUMBERS = [...structure.matchAll(/\n\s*n:\s*(\d+),/g)].map((m) => Number(m[1]));
const BY_SLUG = new Map(SLUGS.map((s, i) => [s, NUMBERS[i]]));

const problems = [];
const warnings = [];
const fail = (chapter, msg) => problems.push(`${chapter}: ${msg}`);
const warn = (chapter, msg) => warnings.push(`${chapter}: ${msg}`);

const REQUIRED_FRONTMATTER = [
  'title',
  'description',
  'chapter',
  'part',
  'difficulty',
  'readingTime',
  'quote',
  'quoteAuthor',
  'quoteSource',
];

let widgetTotal = 0;
let referenceTotal = 0;
let present = 0;

for (const slug of SLUGS) {
  const file = join(CHAPTERS_DIR, `${slug}.mdx`);
  if (!existsSync(file)) {
    fail(slug, 'chapter file is missing');
    continue;
  }
  present++;

  const text = readFileSync(file, 'utf8');
  const n = BY_SLUG.get(slug);
  const fm = text.match(/^---\n([\s\S]*?)\n---/);

  if (!fm) {
    fail(slug, 'no frontmatter block');
    continue;
  }

  for (const key of REQUIRED_FRONTMATTER) {
    if (!new RegExp(`^${key}:`, 'm').test(fm[1])) fail(slug, `frontmatter missing "${key}"`);
  }

  const declared = fm[1].match(/^chapter:\s*(\d+)/m);
  if (declared && Number(declared[1]) !== n) {
    fail(slug, `frontmatter chapter ${declared[1]} does not match structure (${n})`);
  }

  const body = text.slice(fm[0].length);

  // Cross-chapter links must resolve.
  for (const [, target] of body.matchAll(/\/chapters\/([a-z0-9-]+)/g)) {
    if (!BY_SLUG.has(target)) fail(slug, `link to unknown chapter slug "${target}"`);
  }

  // Every reference needs a locator a reader can follow.
  const refs = [...body.matchAll(/<Reference\b[\s\S]*?\/>/g)].map((m) => m[0]);
  referenceTotal += refs.length;
  for (const ref of refs) {
    const who = (ref.match(/title="([^"]{0,60})/) ?? [, '(untitled)'])[1];
    if (!/\b(url|doi)=/.test(ref)) fail(slug, `reference "${who}" has neither url nor doi`);
    if (!/\byear=/.test(ref)) fail(slug, `reference "${who}" has no year`);
  }
  if (refs.length < 4) warn(slug, `only ${refs.length} references`);

  if (!/<Overview/.test(body)) fail(slug, 'no <Overview> block');
  if (!/<Exercises/.test(body)) fail(slug, 'no <Exercises> block');
  if (!/```rust/.test(body)) fail(slug, 'no Rust listing');
}

// Widget ids live in the components, not the MDX: check them there, and make
// sure a chNN directory only ever declares wNN.k ids.
const CH_DIR = join(WEB, 'components', 'ch');
if (existsSync(CH_DIR)) {
  for (const dir of readdirSync(CH_DIR)) {
    const dirPath = join(CH_DIR, dir);
    const chapterOfDir = Number(dir.replace(/^ch0?/, ''));
    const seen = new Set();

    for (const f of readdirSync(dirPath)) {
      if (!f.endsWith('.tsx') && !f.endsWith('.ts')) continue;
      const src = readFileSync(join(dirPath, f), 'utf8');

      for (const [, chapterPart, index] of src.matchAll(/\bid="w(\d+)\.(\d+)"/g)) {
        widgetTotal++;
        const id = `w${chapterPart}.${index}`;
        if (Number(chapterPart) !== chapterOfDir) {
          fail(dir, `widget id ${id} declared in ${dir}`);
        }
        if (seen.has(id)) fail(dir, `duplicate widget id ${id}`);
        seen.add(id);
      }

      // Data colours must come from the palette; greys inside canvas shading are fine.
      const hex = [...src.matchAll(/#[0-9a-fA-F]{6}\b/g)].map((m) => m[0]);
      const suspicious = hex.filter((h) => {
        const [r, g, b] = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
        const spread = Math.max(r, g, b) - Math.min(r, g, b);
        return spread > 40; // a saturated colour, i.e. not a neutral
      });
      if (suspicious.length) {
        warn(`${dir}/${f}`, `hardcoded colour(s) ${[...new Set(suspicious)].join(', ')} — use var(--pr-*)`);
      }
    }
  }
}

// Internal links must go through next/link.
//
// The book is deployed to GitHub Pages under `/<repo>/`, and `basePath` only
// rewrites what Next itself emits: <Link>, next/font, the asset URLs. A raw
// `<a href="/chapters/…">` keeps pointing at the domain root, so it 404s in
// production while working perfectly on localhost — exactly the failure that
// is invisible until it is live. Fragment links (`#w16.1`) are unaffected.
const walk = (dir) =>
  readdirSync(dir).flatMap((entry) => {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) return entry === 'node_modules' ? [] : walk(path);
    return path.endsWith('.tsx') ? [path] : [];
  });

for (const dir of ['app', 'components', 'lib'].map((d) => join(WEB, d))) {
  if (!existsSync(dir)) continue;
  for (const path of walk(dir)) {
    const src = readFileSync(path, 'utf8');
    // Every `<a …>` opening tag, then the ones whose href is site-absolute.
    for (const [tag] of src.matchAll(/<a(?=[\s/>])[^>]*>/g)) {
      if (/href="\//.test(tag) || /href=\{`\//.test(tag)) {
        fail(relative(WEB, path), 'raw <a> to an internal path — use next/link so basePath applies');
      }
    }
  }
}

const chapterCount = SLUGS.length;
console.log(`Chapters present: ${present}/${chapterCount}`);
console.log(`Widgets: ${widgetTotal}   References: ${referenceTotal}`);

if (warnings.length) {
  console.log(`\n${warnings.length} warning(s):`);
  for (const w of warnings) console.log(`  ~ ${w}`);
}

if (problems.length) {
  console.log(`\n${problems.length} problem(s):`);
  for (const p of problems) console.log(`  ✗ ${p}`);
  process.exit(1);
}

console.log('\nBook consistency: OK');
