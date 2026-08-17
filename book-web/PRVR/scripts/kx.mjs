import katex from 'katex';
import { katexMacros } from '../lib/katex-macros.ts';
const cases = [
  ['pmatrix', String.raw`\begin{pmatrix} a & b \\ c & d \end{pmatrix}`],
  ['bmatrix', String.raw`\begin{bmatrix} 1 \\ 2 \end{bmatrix}`],
  ['cases', String.raw`f(x)=\begin{cases} 1 & x>0 \\ 0 & \text{else}\end{cases}`],
  ['aligned', String.raw`\begin{aligned} a &= b \\ c &= d \end{aligned}`],
];
for (const [name, macros] of [['frozen', katexMacros], ['copy', {...katexMacros}]]) {
  for (const [label, tex] of cases) {
    let verdict;
    try {
      const out = katex.renderToString(tex, {macros, strict:'ignore', trust:true, throwOnError:false});
      verdict = out.includes('katex-error') ? 'FAIL (katex-error)' : 'ok';
    } catch (e) { verdict = 'THREW: ' + e.message.slice(0,70); }
    console.log(`${name.padEnd(7)} ${label.padEnd(9)} ${verdict}`);
  }
}
