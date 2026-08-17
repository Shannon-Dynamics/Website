import { AlertTriangle, BookOpen, Cpu, Eye, Info, Lightbulb, Sigma } from 'lucide-react';
import { cn } from '@/lib/utils';

type Variant = 'foundation' | 'conceptual' | 'practical' | 'insight' | 'warning' | 'note' | 'robot';

const VARIANTS: Record<
  Variant,
  { label: string; icon: React.ElementType; accent: string; tint: string }
> = {
  foundation: {
    label: 'Foundation',
    icon: Sigma,
    accent: 'var(--series-1)',
    tint: 'color-mix(in srgb, var(--series-1) 7%, transparent)',
  },
  conceptual: {
    label: 'Conceptual',
    icon: Eye,
    accent: 'var(--series-3)',
    tint: 'color-mix(in srgb, var(--series-3) 8%, transparent)',
  },
  practical: {
    label: 'Practical',
    icon: Cpu,
    accent: 'var(--series-2)',
    tint: 'color-mix(in srgb, var(--series-2) 8%, transparent)',
  },
  insight: {
    label: 'Key idea',
    icon: Lightbulb,
    accent: 'var(--series-4)',
    tint: 'color-mix(in srgb, var(--series-4) 10%, transparent)',
  },
  warning: {
    label: 'Where robots break this',
    icon: AlertTriangle,
    accent: 'var(--status-critical)',
    tint: 'color-mix(in srgb, var(--status-critical) 8%, transparent)',
  },
  note: {
    label: 'Note',
    icon: Info,
    accent: 'var(--text-muted)',
    tint: 'color-mix(in srgb, var(--text-muted) 8%, transparent)',
  },
  robot: {
    label: 'Robot thread',
    icon: BookOpen,
    accent: 'var(--series-7)',
    tint: 'color-mix(in srgb, var(--series-7) 8%, transparent)',
  },
};

/**
 * The FCP layer markers. Every callout carries an icon AND a text label, so the
 * layer is never signalled by color alone.
 */
export function Callout({
  variant = 'note',
  title,
  children,
  className,
}: {
  variant?: Variant;
  title?: string;
  children: React.ReactNode;
  className?: string;
}) {
  const spec = VARIANTS[variant];
  const Icon = spec.icon;

  return (
    <aside
      className={cn('my-6 rounded-xl border border-hairline px-4 py-3.5', className)}
      style={{ borderLeft: `3px solid ${spec.accent}`, background: spec.tint }}
    >
      <p
        className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.07em]"
        style={{ color: spec.accent }}
      >
        <Icon size={13} aria-hidden />
        {title ?? spec.label}
      </p>
      <div className="text-[15px] leading-relaxed text-ink [&>p:last-child]:mb-0">{children}</div>
    </aside>
  );
}

/** Boxed theorem / definition / lemma with a numbered label. */
export function Theorem({
  kind = 'Theorem',
  n,
  title,
  children,
}: {
  kind?: 'Theorem' | 'Definition' | 'Lemma' | 'Proposition' | 'Corollary';
  n?: string;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="my-6 rounded-xl border border-hairline bg-surface-raised px-4 py-3.5">
      <p className="mb-1.5 text-[12px] font-semibold tracking-tight text-ink">
        <span className="text-series-1">
          {kind}
          {n ? ` ${n}` : ''}
        </span>
        {title ? <span className="ml-1.5 font-normal text-ink-secondary">— {title}</span> : null}
      </p>
      <div className="text-[15px] leading-relaxed text-ink [&>p:last-child]:mb-0">{children}</div>
    </div>
  );
}

/** Collapsible proof — full derivations are the book's contract, but they fold. */
export function Proof({ children, title = 'Proof' }: { children: React.ReactNode; title?: string }) {
  return (
    <details className="group my-5 rounded-xl border border-hairline bg-surface px-4 py-3">
      <summary className="cursor-pointer list-none text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-secondary transition-colors hover:text-ink">
        <span className="mr-1.5 inline-block transition-transform group-open:rotate-90">▸</span>
        {title}
      </summary>
      <div className="mt-3 border-t border-hairline pt-3 text-[15px] leading-relaxed text-ink [&>p:last-child]:mb-0">
        {children}
      </div>
    </details>
  );
}
