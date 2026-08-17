'use client';

import { Table2, LineChart as LineIcon } from 'lucide-react';
import { useState } from 'react';
import { cn } from '@/lib/utils';

export interface TableView {
  columns: string[];
  rows: (string | number)[][];
}

/**
 * Standard chrome around every chart in the book.
 *
 * Provides the **table view** the dataviz method requires as the relief
 * channel: any chart whose fills sit below 3:1 on the surface must let the
 * reader read the numbers instead. The toggle is always available, not just
 * where it is strictly obligated.
 */
export function ChartFrame({
  title,
  subtitle,
  caption,
  legend,
  controls,
  table,
  height = 300,
  children,
  className,
}: {
  title?: string;
  subtitle?: string;
  caption?: string;
  legend?: React.ReactNode;
  controls?: React.ReactNode;
  table?: TableView;
  height?: number;
  children: React.ReactNode;
  className?: string;
}) {
  const [showTable, setShowTable] = useState(false);

  return (
    <figure className={cn('my-6 rounded-xl border border-hairline bg-surface', className)}>
      {(title || controls || table) && (
        <div className="flex flex-wrap items-start justify-between gap-3 border-b border-hairline px-4 py-3">
          <div className="min-w-0">
            {title ? (
              <h4 className="text-[13.5px] font-semibold tracking-tight text-ink">{title}</h4>
            ) : null}
            {subtitle ? (
              <p className="mt-0.5 text-[12px] leading-snug text-ink-muted">{subtitle}</p>
            ) : null}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {controls}
            {table ? (
              <button
                type="button"
                onClick={() => setShowTable((v) => !v)}
                className="flex items-center gap-1 rounded-md border border-hairline px-2 py-1 text-[11px] text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
                aria-pressed={showTable}
              >
                {showTable ? <LineIcon size={12} /> : <Table2 size={12} />}
                {showTable ? 'Chart' : 'Table'}
              </button>
            ) : null}
          </div>
        </div>
      )}

      {legend ? <div className="px-4 pt-3">{legend}</div> : null}

      {showTable && table ? (
        <div className="scroll-x thin-scroll px-4 py-3" style={{ maxHeight: height + 40 }}>
          <table className="w-full text-[12.5px]">
            <thead>
              <tr>
                {table.columns.map((c) => (
                  <th
                    key={c}
                    className="tabular border-b-2 border-baseline px-2 py-1.5 text-left font-semibold text-ink"
                  >
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row, i) => (
                <tr key={i}>
                  {row.map((cell, j) => (
                    <td
                      key={j}
                      className="tabular border-b border-hairline px-2 py-1 text-ink-secondary"
                    >
                      {typeof cell === 'number' ? cell.toFixed(3) : cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ height }} className="px-1 py-1">
          {children}
        </div>
      )}

      {caption ? (
        <figcaption className="border-t border-hairline px-4 py-2.5 text-[12.5px] leading-relaxed text-ink-muted">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}

/**
 * Legend — always present for ≥2 series. Identity is carried by a colored mark
 * beside text in an ink token, never by coloring the label itself.
 */
export function Legend({
  items,
}: {
  items: Array<{ label: string; color: string; dashed?: boolean }>;
}) {
  if (items.length < 2) return null;
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1.5">
      {items.map((it) => (
        <li key={it.label} className="flex items-center gap-1.5 text-[11.5px] text-ink-secondary">
          <span
            aria-hidden
            className="inline-block h-[3px] w-4 rounded-full"
            style={{
              background: it.dashed
                ? `repeating-linear-gradient(90deg, ${it.color} 0 4px, transparent 4px 7px)`
                : it.color,
            }}
          />
          {it.label}
        </li>
      ))}
    </ul>
  );
}
