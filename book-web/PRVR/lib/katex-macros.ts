/**
 * The book's global KaTeX macro table.
 *
 * Notation follows TOC.md §2 (Thrun-compatible, extended with manifold operators).
 *
 * The canonical table is frozen so nothing can mutate it, but it must NOT be
 * handed to KaTeX directly: KaTeX writes `\cr` into the macro namespace when it
 * enters any array-family environment, so a frozen table makes every
 * `pmatrix`, `bmatrix`, `cases`, `aligned`, and `array` throw
 * "Cannot add property \cr, object is not extensible". Pass `katexOptions()`
 * (or spread the table yourself) at every render site instead.
 */
const MACROS = Object.freeze({
  // Sets
  '\\R': '\\mathbb{R}',
  '\\N': '\\mathbb{N}',
  '\\Z': '\\mathbb{Z}',

  // Probability
  '\\E': '\\mathbb{E}',
  '\\Var': '\\operatorname{Var}',
  '\\Cov': '\\operatorname{Cov}',
  '\\Prob': '\\operatorname{p}',
  '\\given': '\\mid',
  '\\Normal': '\\mathcal{N}',
  '\\KL': '\\operatorname{KL}',

  // The Bayes-filter vocabulary
  '\\bel': '\\operatorname{bel}',
  '\\belbar': '\\overline{\\operatorname{bel}}',

  // Lie groups and manifolds
  '\\SOtwo': '\\mathrm{SO}(2)',
  '\\SOthree': '\\mathrm{SO}(3)',
  '\\SEtwo': '\\mathrm{SE}(2)',
  '\\SEthree': '\\mathrm{SE}(3)',
  '\\sotwo': '\\mathfrak{so}(2)',
  '\\sethree': '\\mathfrak{se}(3)',
  '\\bplus': '\\boxplus',
  '\\bminus': '\\boxminus',
  '\\Ad': '\\operatorname{Ad}',

  // Matrix helpers
  '\\tr': '\\operatorname{tr}',
  '\\diag': '\\operatorname{diag}',
  '\\rank': '\\operatorname{rank}',
  '\\T': '^{\\mathsf{T}}',

  // Parameterized
  '\\norm': '\\left\\lVert #1 \\right\\rVert',
  '\\abs': '\\left\\lvert #1 \\right\\rvert',
  '\\vv': '\\mathbf{#1}',
  '\\mat': '\\mathbf{#1}',
});

/**
 * A fresh, mutable copy of the macro table for one render pass.
 *
 * Mutable because KaTeX needs to write into it (see above); a *copy* so that a
 * stray `\gdef` in one chapter cannot leak its definition into whichever
 * chapter happens to compile next and make the build order-dependent.
 */
export const katexMacros = () => ({ ...MACROS });

/** The standard KaTeX options for this book. Every render site should use these. */
export const katexOptions = () => ({
  macros: katexMacros(),
  // Turbopack cannot serialize functions into plugin options, so every value
  // here is a plain one.
  strict: 'ignore' as const,
  trust: true,
  minRuleThickness: 0.06,
  maxSize: 20,
});
