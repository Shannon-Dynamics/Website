'use client';

import Link from 'next/link';
import { useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';

/**
 * w26.4 — the Retrospective Scorecard.
 *
 * Two panels, deliberately the same size: what Rust's type system caught while
 * this book was being written, and what it cost. Every diagnostic below is the
 * shape rustc actually emits for that mistake — the error code, the primary
 * label, and the note — reproduced so the reader recognises it in their own
 * terminal rather than only in prose.
 */

interface Vignette {
  id: string;
  title: string;
  code: string;
  wrong: string;
  rustc: string[];
  moral: string;
  chapter: string;
  slug: string;
}

const VIGNETTES: Vignette[] = [
  {
    id: 'frames',
    title: 'A pose in the wrong frame',
    chapter: 'Ch. 3',
    slug: 'ch03-geometry-of-motion',
    code: 'E0308',
    wrong: 'let goal_base: Pose<Base> = plan.waypoint(); // waypoint is Pose<Map>',
    rustc: [
      'error[E0308]: mismatched types',
      '   --> crates/capstone/src/tasks/control.rs:88:37',
      '    |',
      ' 88 |     let goal_base: Pose<Base> = plan.waypoint();',
      '    |                    ----------   ^^^^^^^^^^^^^^^ expected `Pose<Base>`, found `Pose<Map>`',
      '    |                    |',
      '    |                    expected due to this',
      '    |',
      '    = note: expected struct `Pose<frames::Base>`',
      '               found struct `Pose<frames::Map>`',
    ],
    moral:
      'The single most expensive class of robotics bug — a transform applied in the wrong frame — becomes a type error. Chapter 3 pays a small ergonomic tax for this and it is the best trade in the book.',
  },
  {
    id: 'dims',
    title: 'A Jacobian of the wrong shape',
    chapter: 'Ch. 6',
    slug: 'ch06-kalman-filters',
    code: 'E0308',
    wrong: 'let s = h * sigma * h.transpose() + q; // h: 2×3, sigma: 3×3, q: 3×3',
    rustc: [
      'error[E0308]: mismatched types',
      '   --> crates/ch06_kf/src/ekf.rs:141:41',
      '    |',
      '141 |     let s = h * sigma * h.transpose() + q;',
      '    |                                         ^ expected `Const<2>`, found `Const<3>`',
      '    |',
      '    = note: expected struct `Matrix<f64, Const<2>, Const<2>, ...>`',
      '               found struct `Matrix<f64, Const<3>, Const<3>, ...>`',
    ],
    moral:
      'nalgebra carries matrix dimensions in the type. A measurement-noise matrix pasted into an innovation covariance cannot compile, which is exactly the mistake that silently produces a plausible-looking filter that is wrong.',
  },
  {
    id: 'match',
    title: 'A mode the supervisor forgot',
    chapter: 'Ch. 26',
    slug: 'ch26-capstone',
    code: 'E0004',
    wrong: 'match self.mode { Explore => .., Navigate{..} => .., Relocalize => .., Done => .. }',
    rustc: [
      'error[E0004]: non-exhaustive patterns: `Mode::Recover(_)` not covered',
      '   --> crates/capstone/src/tasks/supervisor.rs:203:11',
      '    |',
      '203 |     match self.mode {',
      '    |           ^^^^^^^^^ pattern `Mode::Recover(_)` not covered',
      '    |',
      'note: `Mode` defined here',
      '    = help: ensure the match covers every variant, or add a `_ => {}` arm',
    ],
    moral:
      'Adding one state to the mode machine forced every decision site to be revisited. A stringly-typed mode, or a catch-all arm, would have let the new state fall through to "do nothing" — which is what a robot in an unhandled state does off a loading dock.',
  },
  {
    id: 'borrow',
    title: 'Two tasks mutating one map',
    chapter: 'Ch. 13',
    slug: 'ch13-occupancy-grids',
    code: 'E0499',
    wrong: 'esdf.rebuild(&mut map); map.integrate(&scan, &pose); // both alive at once',
    rustc: [
      'error[E0499]: cannot borrow `map` as mutable more than once at a time',
      '   --> crates/capstone/src/mission.rs:96:26',
      '    |',
      ' 95 |     let layer = esdf.rebuild(&mut map);',
      '    |                              -------- first mutable borrow occurs here',
      ' 96 |     map.integrate(&scan, &pose);',
      '    |     ^^^^^^^^^^^^^ second mutable borrow occurs here',
      ' 97 |     controller.follow(layer);',
      '    |                       ----- first borrow later used here',
    ],
    moral:
      'The borrow checker refused a data race that, on a threaded native build, would have shown up as a costmap containing half of one scan — intermittently, on one machine in four.',
  },
  {
    id: 'send',
    title: 'A generator that cannot cross a thread',
    chapter: 'Ch. 8',
    slug: 'ch08-nonparametric-filters',
    code: 'E0277',
    wrong: 'particles.par_iter_mut().for_each(|p| *p = sample(p, &mut rng));',
    rustc: [
      'error[E0277]: `Rc<RefCell<SmallRng>>` cannot be sent between threads safely',
      '   --> crates/capstone/src/tasks/supervisor.rs:271:22',
      '    |',
      '271 |     particles.par_iter_mut().for_each(|p| *p = sample(p, &mut rng));',
      '    |                              ^^^^^^^^ `Rc<RefCell<SmallRng>>` cannot be sent between threads safely',
      '    |',
      '    = help: within this closure, the trait `Send` is not implemented for `Rc<RefCell<SmallRng>>`',
      '    = note: required for `&Rc<RefCell<SmallRng>>` to implement `Send`',
    ],
    moral:
      'Sharing one RNG across rayon workers would have destroyed the property this whole book rests on: a seeded run reproduces exactly. The compiler asked for one generator per worker, deterministically split, before it would let the program exist.',
  },
];

interface Cost {
  title: string;
  body: string;
}

const COSTS: Cost[] = [
  {
    title: 'Graph code fights the borrow checker',
    body: 'A pose graph is a cyclic structure with mutation everywhere, which is precisely what Rust makes awkward. Every graph in this book therefore stores indices rather than references — `petgraph` node ids, `Vec<PoseNode>` offsets — and pays for it in readability and in a class of index bug the compiler no longer catches. The safety was moved, not created.',
  },
  {
    title: 'Compile times are the price of monomorphisation',
    body: 'Const-generic matrices and deeply generic filters instantiate a great deal of code. A clean build of the workspace is minutes, not seconds; incremental rebuilds during widget work are the ones that matter and they stay tolerable. If you are used to editing a Python filter and rerunning it in under a second, this is the adjustment.',
  },
  {
    title: 'WASM has no threads by default',
    body: 'The native build runs each task on its own thread over a channel. The browser build cannot, so the scheduler is cooperative and round-robin. That makes the browser demo prove real-time *throughput*, not real-time *scheduling* — a distinction the chapter is careful about, because they are not the same claim.',
  },
  {
    title: 'The ecosystem has gaps, and they move',
    body: 'Sparse solvers, factor graphs, and Lie-group libraries all exist in Rust and all are younger than their C++ counterparts. Pinning versions is not optional. Every ecosystem claim in this book is dated August 2026 and should be re-checked before it is trusted; that section will rot faster than any other page here.',
  },
  {
    title: 'The simulator grades its own homework',
    body: 'Trajectory RMSE against ground truth exists only because we own the world. On hardware there is no truth column: evaluation becomes held-out maps, loop-closure precision and recall, repeatability across runs, and disagreement between two passes over the same corridor — the map-consistency number Chapter 16 introduced for exactly this reason.',
  },
];

export function RetrospectiveScorecard() {
  const [open, setOpen] = useState<string>('match');
  const v = VIGNETTES.find((x) => x.id === open) as Vignette;

  return (
    <WidgetFrame
      id="w26.4"
      title="Retrospective Scorecard"
      teaches="A language choice is a trade, not a win: name what the type system caught and what it charged, at the same size."
      caption={
        <>
          Left: five mistakes that never reached a test run, each shown as the diagnostic{' '}
          <span className="font-mono">rustc</span> prints for it. Right: what that cost, at equal
          length and with no apologies. The honest summary is narrow but real — Rust removed a
          specific family of bugs (frames, dimensions, unhandled states, aliased mutation,
          accidental nondeterminism) that in robotics are expensive and hard to reproduce, and
          charged for it in ergonomics, build time, and ecosystem maturity.
        </>
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 lg:divide-x lg:divide-fd-border">
        {/* ---- Panel A: what the compiler caught ---- */}
        <div className="p-3">
          <p className="eyebrow mb-2">A · caught at compile time</p>
          <div className="flex flex-wrap gap-1.5">
            {VIGNETTES.map((x) => (
              <button
                key={x.id}
                type="button"
                onClick={() => setOpen(x.id)}
                aria-pressed={open === x.id}
                className={`rounded-sm border px-2 py-1 font-mono text-[0.68rem] transition-colors ${
                  open === x.id
                    ? 'border-fd-primary bg-fd-primary text-fd-primary-foreground'
                    : 'border-fd-border bg-fd-card hover:bg-fd-accent'
                }`}
              >
                {x.code}
              </button>
            ))}
          </div>

          <h4 className="mt-3 font-display text-sm font-semibold">
            {v.title}{' '}
            <Link href={`/chapters/${v.slug}`} className="font-mono text-[0.7rem] font-normal text-fd-primary hover:underline">
              {v.chapter}
            </Link>
          </h4>

          <pre className="mt-2 overflow-x-auto rounded-sm border border-fd-border bg-fd-muted/40 p-2 font-mono text-[0.66rem] leading-snug">
            <code style={{ color: 'var(--pr-truth)' }}>{v.wrong}</code>
          </pre>

          <pre className="mt-1.5 overflow-x-auto rounded-sm border border-fd-border bg-fd-muted/40 p-2 font-mono text-[0.64rem] leading-snug">
            {v.rustc.map((line, i) => (
              <div
                key={i}
                style={{
                  color: line.startsWith('error')
                    ? 'var(--pr-prediction)'
                    : line.trimStart().startsWith('=') || line.startsWith('note:')
                      ? 'var(--pr-measurement)'
                      : undefined,
                }}
              >
                {line}
              </div>
            ))}
          </pre>

          <p className="mt-2 font-prose text-[0.84rem] leading-snug">{v.moral}</p>
        </div>

        {/* ---- Panel B: what it cost ---- */}
        <div className="border-t border-fd-border p-3 lg:border-t-0">
          <p className="eyebrow mb-2">B · what it cost</p>
          <dl className="space-y-2.5">
            {COSTS.map((c) => (
              <div key={c.title}>
                <dt className="font-display text-[0.85rem] font-semibold" style={{ color: 'var(--pr-prediction)' }}>
                  {c.title}
                </dt>
                <dd className="font-prose text-[0.82rem] leading-snug">{c.body}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </WidgetFrame>
  );
}
