import katex from 'katex';

import { katexMacros } from '@/lib/katex-macros';

export interface NotationRow {
  /** LaTeX source, rendered to HTML at build time. */
  sym: string;
  meaning: string;
  note?: string;
}

export interface NotationTableProps {
  rows: NotationRow[];
}

/**
 * Symbols are typeset during the static export, so no KaTeX reaches the
 * browser. `throwOnError: false` keeps a bad symbol from failing the build —
 * it renders in KaTeX's error color instead, which is loud enough to catch.
 */
function renderSymbol(sym: string): string {
  return katex.renderToString(sym, {
    throwOnError: false,
    strict: 'ignore',
    macros: katexMacros(),
  });
}

/**
 * Server Component by construction: it imports KaTeX and must never become a
 * client component, or the macro table and renderer ship to the reader.
 */
export function NotationTable({ rows }: NotationTableProps) {
  const hasNotes = rows.some((row) => Boolean(row.note));

  return (
    <div className="my-8 overflow-x-auto">
      {/* `table!` overrides the global `#nd-page table { display: block }`
          scroll hack; this table scrolls inside the wrapper above instead. */}
      <table className="table! w-full border-collapse text-start">
        <caption className="sr-only">Notation used in this chapter</caption>
        <thead>
          <tr className="border-b border-fd-border">
            <th scope="col" className="eyebrow w-[9rem] py-2 pe-4 text-start align-bottom">
              Symbol
            </th>
            <th scope="col" className="eyebrow py-2 pe-4 text-start align-bottom">
              Meaning
            </th>
            {hasNotes ? (
              <th scope="col" className="eyebrow w-[30%] py-2 text-start align-bottom">
                Note
              </th>
            ) : null}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.sym}
              className="border-b border-fd-border/60 align-baseline last:border-b-0"
            >
              <td className="py-2 pe-4 whitespace-nowrap">
                <span
                  className="text-fd-foreground"
                  dangerouslySetInnerHTML={{ __html: renderSymbol(row.sym) }}
                />
              </td>
              <td className="py-2 pe-4 font-prose text-[0.95rem] leading-snug text-fd-foreground">
                {row.meaning}
              </td>
              {hasNotes ? (
                <td className="py-2 font-ui text-[0.8rem] leading-snug text-fd-muted-foreground">
                  {row.note}
                </td>
              ) : null}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
