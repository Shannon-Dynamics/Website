import { Pomcp, layoutTree } from '../lib/pomdp/pomcp.ts';
import { Rng } from '../lib/prob/rng.ts';
const f = (x, n = 2) => Number(x).toFixed(n);
const DONE = -1;
function makeModel(acc = 0.85, gamma = 0.95) {
  return {
    nActions: 3,
    gamma,
    step(x, u, rng) {
      if (x === DONE) return { x, z: 0, r: 0 };
      if (u === 0) return { x, z: rng.next() < acc ? x : 1 - x, r: -1 };
      const opened = u === 1 ? 0 : 1;
      return { x: DONE, z: 0, r: opened === x ? -100 : 10 };
    },
    terminal: (x) => x === DONE,
    rollout: (x, d, rng) => (d < 2 ? 0 : rng.next() < 0.5 ? 1 : 2),
  };
}
console.log('c sweep, 4000 sims, maxDepth 8');
for (const c of [40, 60, 80, 110, 150]) {
  const out = [];
  for (const pLeft of [0.5, 0.85, 0.97]) {
    const rng = new Rng(11);
    const parts = Array.from({ length: 500 }, () => (rng.next() < pLeft ? 0 : 1));
    const tree = new Pomcp(makeModel(), parts, rng, { c, maxDepth: 8 });
    tree.search(4000);
    out.push(`b=${pLeft}: ${['listen', 'open-L', 'open-R'][tree.bestAction()]} Q=${tree.rootQ().map((q) => f(q)).join('/')} N=${tree.rootVisits().join('/')} nodes=${tree.beliefs.length}`);
  }
  console.log(`c=${c}\n   ${out.join('\n   ')}`);
}

console.log('\n--- anytime at b=0.5, c=80 ---');
{
  const rng = new Rng(3);
  const parts = Array.from({ length: 500 }, () => (rng.next() < 0.5 ? 0 : 1));
  const tree = new Pomcp(makeModel(), parts, rng, { c: 80, maxDepth: 8 });
  for (const n of [16, 64, 256, 1024, 4096]) {
    tree.search(n - tree.simulations);
    const lay = layoutTree(tree, 2, Math.max(2, Math.round(n / 60)));
    console.log(`  sims=${tree.simulations} Q=${tree.rootQ().map((q) => f(q)).join('/')} N=${tree.rootVisits().join('/')} nodes=${tree.beliefs.length} laid=${lay.nodes.length} best=${['listen', 'open-L', 'open-R'][tree.bestAction()]}`);
  }
  const lay = layoutTree(tree, 2, 40);
  console.log('  layout width', lay.width, 'nodes', lay.nodes.length);
  for (const nd of lay.nodes) {
    console.log(`   ${nd.kind} row=${nd.row} x=${f(nd.x)} n=${nd.n} q=${Number.isNaN(nd.q) ? '-' : f(nd.q)} act=${nd.action} obs=${nd.obs} parts=${nd.particles} mean=${Number.isNaN(nd.particleMean) ? '-' : f(nd.particleMean, 3)}`);
  }
}
