import type { Metadata } from 'next';
import { VISUAL_LANGUAGE } from '@/lib/theme';
import { PageBanner, bookCrumb } from '@/components/shannon/PageBanner';

export const metadata: Metadata = {
  title: 'Method',
  description:
    'How this book is built: the FCP method, the visual language, the toolchain, and the standard of evidence.',
};

const CRATES = [
  { domain: 'Deep learning', crate: 'burn', why: 'Pure Rust, multi-backend, trains natively and runs in the browser via WGPU.' },
  { domain: 'Tensors & geometry', crate: 'ndarray, nalgebra', why: 'ML-shaped data and SE(3)/Riccati geometry respectively.' },
  { domain: 'Physics', crate: 'rapier2d / rapier3d', why: 'Deterministic option, WASM-compatible — simulations run client-side.' },
  { domain: 'Robot models', crate: 'urdf-rs, k', why: 'URDF parsing plus serial-chain forward/inverse kinematics and Jacobians.' },
  { domain: '3D scenes', crate: 'bevy + bevy_rapier', why: 'WASM-capable 3D for locomotion and manipulation demonstrations.' },
  { domain: 'Dashboards', crate: 'egui + egui_plot', why: 'Immediate-mode training dashboards that compile to the browser.' },
  { domain: 'Parallelism', crate: 'rayon', why: 'Vectorized environment rollout farms.' },
  { domain: 'Optimization', crate: 'cmaes', why: 'Policy search over DMP weights and system-identification fits.' },
];

export default function AboutPage() {
  return (
    <>
      <PageBanner
        small
        crumb={bookCrumb(
          { label: 'REINFORCEMENT LEARNING FOR ROBOTICS', href: '/' },
          { label: 'METHOD' },
        )}
        title="How this book is built"
        sub="The FCP method, the visual language, the toolchain, and the standard of evidence it holds itself to."
      />

      <div className="mx-auto max-w-3xl px-4 py-14 pb-20">
      <section>
        <h2 className="text-[19px] font-semibold tracking-tight text-ink">The FCP method</h2>
        <p className="mt-2 text-[15.5px] leading-relaxed text-ink-secondary">
          Three layers run through every chapter, interleaved rather than separated. The{' '}
          <strong className="text-ink">Foundation</strong> layer states and derives the mathematics
          in full — if a proof is genuinely out of scope, the theorem is stated precisely and the
          proof is cited, never waved at. The <strong className="text-ink">Conceptual</strong> layer
          gives every hard idea something you can manipulate, because a parameter you have dragged
          is a parameter you understand. The <strong className="text-ink">Practical</strong> layer
          implements the algorithm in Rust, seeded and tested, so the reader ends with working code
          rather than pseudocode.
        </p>
      </section>

      <section className="mt-9">
        <h2 className="text-[19px] font-semibold tracking-tight text-ink">The visual language</h2>
        <p className="mt-2 text-[15.5px] leading-relaxed text-ink-secondary">
          One encoding scheme runs book-wide, so a color means the same thing in Chapter 20 as it
          did in Chapter 4. The palette is validated for colorblind separation, lightness banding
          and contrast in both light and dark themes — not chosen by eye.
        </p>
        <div className="mt-4 overflow-hidden rounded-xl border border-hairline">
          <table className="w-full text-[13.5px]">
            <thead>
              <tr className="bg-surface-sunken">
                <th className="px-3 py-2 text-left font-semibold text-ink">Quantity</th>
                <th className="px-3 py-2 text-left font-semibold text-ink">Encoding</th>
              </tr>
            </thead>
            <tbody>
              {VISUAL_LANGUAGE.map((v) => (
                <tr key={v.role} className="border-t border-hairline">
                  <td className="px-3 py-2 text-ink-secondary">{v.role}</td>
                  <td className="px-3 py-2 text-ink-secondary">{v.encoding}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-[13.5px] leading-relaxed text-ink-muted">
          Every chart also carries a table view, so the numbers are readable without relying on
          color at all, and every interactive widget degrades to a captioned static figure.
        </p>
      </section>

      <section className="mt-9">
        <h2 className="text-[19px] font-semibold tracking-tight text-ink">The Rust toolchain</h2>
        <p className="mt-2 text-[15.5px] leading-relaxed text-ink-secondary">
          Teaching code lives in the book&apos;s own crates rather than depending on third-party RL
          libraries — building the gym-style abstractions is itself curriculum. External crates are
          used where they are genuinely best in class:
        </p>
        <div className="mt-4 overflow-hidden rounded-xl border border-hairline">
          <table className="w-full text-[13.5px]">
            <thead>
              <tr className="bg-surface-sunken">
                <th className="px-3 py-2 text-left font-semibold text-ink">Domain</th>
                <th className="px-3 py-2 text-left font-semibold text-ink">Crate</th>
                <th className="px-3 py-2 text-left font-semibold text-ink">Why</th>
              </tr>
            </thead>
            <tbody>
              {CRATES.map((c) => (
                <tr key={c.domain} className="border-t border-hairline">
                  <td className="px-3 py-2 text-ink-secondary">{c.domain}</td>
                  <td className="px-3 py-2">
                    <code className="font-mono text-[12.5px] text-ink">{c.crate}</code>
                  </td>
                  <td className="px-3 py-2 text-ink-muted">{c.why}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-9">
        <h2 className="text-[19px] font-semibold tracking-tight text-ink">
          The standard of evidence
        </h2>
        <p className="mt-2 text-[15.5px] leading-relaxed text-ink-secondary">
          Claims about what reinforcement learning achieves on real robots are graded against Tang
          et al.&apos;s levels of real-world success, from L0 (simulation only) to L5 (shipping in a
          product). It is an unusually honest rubric for a field that often reports its best single
          trial, and this book applies it to its own capstone as strictly as to the literature.
        </p>
      </section>
      </div>
    </>
  );
}
