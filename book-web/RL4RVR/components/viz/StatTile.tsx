import { cn } from '@/lib/utils';

/**
 * A single headline number. Sometimes the right answer is *not* a chart — a
 * scalar the reader must track (current Δ, episode count, clip fraction) reads
 * better as a stat tile than as a one-point plot.
 */
export function StatTile({
  label,
  value,
  unit,
  hint,
  status,
  mono = true,
  className,
}: {
  label: string;
  value: string | number;
  unit?: string;
  hint?: string;
  /** Reserved status meaning — always paired with the text label below. */
  status?: 'good' | 'warning' | 'critical';
  mono?: boolean;
  className?: string;
}) {
  const statusColor =
    status === 'good'
      ? 'var(--status-good)'
      : status === 'warning'
        ? 'var(--status-warning)'
        : status === 'critical'
          ? 'var(--status-critical)'
          : undefined;

  return (
    <div className={cn('rounded-lg border border-hairline bg-surface px-3 py-2.5', className)}>
      <p className="text-[10.5px] font-medium uppercase tracking-[0.06em] text-ink-muted">
        {label}
      </p>
      <p
        className={cn(
          'mt-0.5 text-[19px] font-semibold leading-tight tracking-tight text-ink',
          mono && 'tabular',
        )}
        style={statusColor ? { color: statusColor } : undefined}
      >
        {typeof value === 'number' ? formatValue(value) : value}
        {unit ? <span className="ml-1 text-[12px] font-normal text-ink-muted">{unit}</span> : null}
      </p>
      {hint ? <p className="mt-0.5 text-[11px] leading-snug text-ink-muted">{hint}</p> : null}
    </div>
  );
}

function formatValue(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  if (v === 0) return '0';
  if (Math.abs(v) < 0.001) return v.toExponential(1);
  return v.toFixed(3);
}

/** A row of stat tiles — the dashboard's KPI strip. */
export function StatRow({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-3 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">{children}</div>
  );
}
