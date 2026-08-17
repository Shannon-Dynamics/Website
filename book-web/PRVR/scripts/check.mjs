import katex from 'katex';
import { runSelfChecks } from '../lib/__checks__.ts';
import { katexOptions } from '../lib/katex-macros.ts';

/**
 * Everything the book asserts about itself, checked in one place:
 *
 *  1. the numerical invariants of the algorithm library that powers the
 *     in-page simulations, and
 *  2. that the KaTeX configuration can actually render the environments the
 *     chapters use.
 *
 * (2) exists because a frozen macro table once made every matrix, cases, and
 * aligned environment in the book render as a red error — KaTeX writes `\cr`
 * into the macro namespace, so the table it is handed must be mutable. That
 * failure is invisible in a passing build, so it gets a test.
 */

const ENVIRONMENTS = [
  ['pmatrix', String.raw`\begin{pmatrix} a & b \\ c & d \end{pmatrix}`],
  ['bmatrix', String.raw`\mat{K} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`],
  ['cases', String.raw`f(x) = \begin{cases} 1 & x > 0 \\ 0 & \text{otherwise} \end{cases}`],
  ['aligned', String.raw`\begin{aligned} \bel(x_t) &= \eta\, p(z_t \mid x_t) \\ &\quad \times \belbar(x_t) \end{aligned}`],
  ['array', String.raw`\begin{array}{cc} 1 & 2 \\ 3 & 4 \end{array}`],
  ['substack', String.raw`\sum_{\substack{i=1 \\ i \neq j}}^{n} x_i`],
  ['book macros', String.raw`\belbar(x_t) = \int p(x_t \mid u_t, x_{t-1})\, \bel(x_{t-1})\, dx_{t-1}`],
  ['manifold macros', String.raw`T \in \SEtwo,\quad x \bplus \tau,\quad a \bminus b,\quad \Ad_T`],
  ['term colors', String.raw`\htmlClass{term-prior}{\bel(x_{t-1})} \cdot \htmlClass{term-measurement}{p(z_t \mid x_t)}`],
];

function katexChecks() {
  return ENVIRONMENTS.map(([name, tex]) => {
    try {
      const html = katex.renderToString(tex, { ...katexOptions(), throwOnError: false });
      const bad = html.includes('katex-error');
      return { name: `katex: ${name} renders`, pass: !bad, detail: bad ? 'rendered as katex-error' : undefined };
    } catch (error) {
      return { name: `katex: ${name} renders`, pass: false, detail: String(error.message).slice(0, 120) };
    }
  });
}

const results = [...runSelfChecks(), ...katexChecks()];

let failed = 0;
for (const r of results) {
  if (!r.pass) failed++;
  console.log(`${r.pass ? 'PASS' : 'FAIL'}  ${r.name}${r.detail ? '  — ' + r.detail : ''}`);
}
console.log(`\n${results.length - failed}/${results.length} passed`);
process.exit(failed ? 1 : 0);
