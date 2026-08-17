import katex from 'katex';

import { katexMacros } from '@/lib/katex-macros';

/**
 * KaTeX with somewhere to scribble.
 *
 * The book's global macro table is `Object.freeze`d on purpose, so a stray
 * `\gdef` in one chapter cannot leak into another. KaTeX, however, writes
 * `\cr` into the macro namespace while parsing *any* environment — `pmatrix`,
 * `aligned`, `cases`, `substack` — which means a block matrix written in
 * ordinary `$$…$$` throws before it renders.
 *
 * This chapter is largely about the block structure of Σ, so it renders its
 * matrices through here instead: the same options as `source.config.ts`, but
 * with a fresh copy of the macro table that KaTeX is allowed to mutate. The
 * shared table stays frozen and this file owns the copy.
 *
 * Server Component by construction — it must never become a client component,
 * or KaTeX and the macro table ship to the reader.
 */
export function TeX({ children, display = false }: { children: string; display?: boolean }) {
  const html = katex.renderToString(children, {
    displayMode: display,
    throwOnError: false,
    strict: 'ignore',
    trust: true,
    macros: katexMacros(),
    minRuleThickness: 0.06,
    maxSize: 20,
  });

  return display ? (
    <div className="my-6" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span dangerouslySetInnerHTML={{ __html: html }} />
  );
}
