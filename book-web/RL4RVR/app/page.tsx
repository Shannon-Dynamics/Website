import Link from 'next/link';
import { ArrowRight, Cpu, Eye, Sigma } from 'lucide-react';
import { CHAPTERS, PARTS, ROBOTS } from '@/lib/chapters';
import { PageBanner, bookCrumb } from '@/components/shannon/PageBanner';
import { BOOK_PDF } from '@/lib/shannon';

const LAYERS = [
  {
    icon: Sigma,
    letter: 'F',
    title: 'Foundation',
    color: 'var(--series-1)',
    body: 'Complete mathematical formalism. Definitions, theorems, and derivations carried through to the last line — including the convergence conditions robots routinely violate, stated plainly rather than buried.',
  },
  {
    icon: Eye,
    letter: 'C',
    title: 'Conceptual',
    color: 'var(--series-3)',
    body: 'Every hard idea gets a visual you can manipulate. Drag γ and watch the horizon stretch; crank Δt until the integrator explodes; slide λ from 0 to 1 and rediscover Monte Carlo.',
  },
  {
    icon: Cpu,
    letter: 'P',
    title: 'Practical',
    color: 'var(--series-2)',
    body: 'Every algorithm implemented in Rust with the best current crates — burn for learning, rapier for physics, egui for dashboards — as code that trains natively and demos in the browser.',
  },
];

export default function HomePage() {
  return (
    <>
      <PageBanner
        crumb={bookCrumb({ label: 'REINFORCEMENT LEARNING FOR ROBOTICS' })}
        title="Reinforcement Learning for Robotics"
        sub="The FCP way — from multi-armed bandits to a quadruped that learns to walk, in Rust."
      />

      <div className="mx-auto max-w-5xl px-4">
      {/* The thesis */}
      <section className="py-14 sm:py-16">
        <p className="sd-kicker">Shannon Press · 2026 · {CHAPTERS.length} chapters</p>
        <h2 className="mt-4 max-w-3xl text-[clamp(1.6rem,4vw,2.3rem)] font-bold leading-[1.15] tracking-[-0.025em] text-ink">
          Foundation · Conceptual · Practical
        </h2>
        <p className="mt-5 max-w-2xl text-[16.5px] leading-relaxed text-ink-secondary">
          Most reinforcement learning texts prove theorems about environments that robots do not
          live in. This book keeps the mathematics complete, then insists on saying which
          assumptions break the moment a real machine touches the ground — and what practitioners do
          about it.
        </p>

        <div className="mt-8 flex flex-wrap gap-3">
          <Link href="/chapters/why-rl-for-robotics" className="sd-btn pri">
            Start reading
            <ArrowRight size={13} />
          </Link>
          <Link href="/chapters" className="sd-btn">
            Browse all {CHAPTERS.length} chapters
          </Link>
          <a className="sd-btn" href={BOOK_PDF} target="_blank" rel="noopener">
            Download PDF <span aria-hidden="true">↓</span>
          </a>
        </div>
      </section>

      {/* The method */}
      <section className="border-t border-hairline py-12">
        <h2 className="sd-kicker sd-kicker-accent">
          The method
        </h2>
        <p className="mt-1.5 max-w-2xl text-[19px] font-semibold tracking-tight text-ink">
          Three layers, interleaved in every chapter — never one without the others.
        </p>

        <div className="mt-6 grid gap-4 md:grid-cols-3">
          {LAYERS.map((l) => {
            const Icon = l.icon;
            return (
              <div
                key={l.letter}
                className="rounded-xl border border-hairline bg-surface p-5"
                style={{ borderTop: `3px solid ${l.color}` }}
              >
                <div className="flex items-center gap-2">
                  <span
                    className="grid h-7 w-7 place-items-center rounded-md text-[13px] font-bold text-white"
                    style={{ background: l.color }}
                  >
                    {l.letter}
                  </span>
                  <Icon size={15} style={{ color: l.color }} aria-hidden />
                  <h3 className="text-[15px] font-semibold tracking-tight text-ink">{l.title}</h3>
                </div>
                <p className="mt-2.5 text-[14px] leading-relaxed text-ink-secondary">{l.body}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* The cast */}
      <section className="border-t border-hairline py-12">
        <h2 className="sd-kicker sd-kicker-accent">
          The cast
        </h2>
        <p className="mt-1.5 max-w-2xl text-[19px] font-semibold tracking-tight text-ink">
          Four robots carry every idea in the book, so abstractions always land somewhere physical.
        </p>

        <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {Object.values(ROBOTS).map((r) => (
            <div key={r.name} className="rounded-xl border border-hairline bg-surface p-4">
              <p className="text-[15px] font-semibold tracking-tight text-ink">{r.name}</p>
              <p className="mt-0.5 text-[12.5px] text-ink-muted">{r.kind}</p>
              <p className="mt-2 text-[12.5px] leading-relaxed text-ink-secondary">{r.thread}</p>
              <p className="mt-2 text-[11.5px] text-ink-muted">Enters in Chapter {r.intro}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Parts */}
      <section className="border-t border-hairline py-12">
        <h2 className="sd-kicker sd-kicker-accent">
          The arc
        </h2>
        <div className="mt-5 space-y-2">
          {PARTS.map((part) => (
            <div
              key={part.id}
              className="flex flex-wrap items-baseline gap-x-3 gap-y-1 rounded-xl border border-hairline bg-surface px-4 py-3"
            >
              <span className="sd-kicker sd-kicker-accent">
                Part {part.id}
              </span>
              <span className="text-[15px] font-semibold tracking-tight text-ink">
                {part.title}
              </span>
              <span className="text-[12.5px] text-ink-muted">
                Chapters {part.chapters[0]}–{part.chapters[part.chapters.length - 1]}
              </span>
              <span className="w-full text-[13px] italic text-ink-secondary">{part.tagline}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Grounding */}
      <section className="border-t border-hairline py-12 pb-20">
        <h2 className="sd-kicker sd-kicker-accent">
          Built on
        </h2>
        <p className="mt-1.5 max-w-2xl text-[15px] leading-relaxed text-ink-secondary">
          Four works form the spine: Sutton &amp; Barto&apos;s <em>Reinforcement Learning: An
          Introduction</em> for the mathematics, Kober, Bagnell &amp; Peters&apos; 2013 survey for
          the robotics bridge, Tang et al.&apos;s 2024 survey of real-world successes for the modern
          taxonomy, and Akinola&apos;s lectures for pedagogy. Where those stop, this book continues —
          PPO and SAC as workhorses, world models, offline RL, teacher–student sim-to-real, and
          foundation-model frontiers — always grounded in the same spine.
        </p>
      </section>
      </div>
    </>
  );
}
