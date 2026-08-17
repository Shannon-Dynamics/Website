'use client';

import { Check, Copy } from 'lucide-react';
import { useRef, useState } from 'react';

/**
 * Wraps every fenced code block: adds the language tag, an optional caption
 * and a copy button. Highlighting itself is done at build time by Shiki
 * (rehype-pretty-code), so no syntax-highlighting JS ships to the reader.
 */
export function Pre({
  children,
  ...props
}: React.HTMLAttributes<HTMLPreElement> & { 'data-language'?: string }) {
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);
  const language = props['data-language'];

  const copy = async () => {
    const text = ref.current?.innerText ?? '';
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable — the code is still selectable */
    }
  };

  return (
    <div className="group relative my-5 overflow-hidden rounded-xl border border-hairline bg-surface-sunken">
      <div className="flex items-center justify-between border-b border-hairline px-3 py-1.5">
        <span className="font-mono text-[10.5px] uppercase tracking-[0.08em] text-ink-muted">
          {language ?? 'code'}
        </span>
        <button
          type="button"
          onClick={copy}
          className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-ink-muted transition-colors hover:bg-surface hover:text-ink"
          aria-label={copied ? 'Copied' : 'Copy code'}
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre ref={ref} {...props} className="thin-scroll">
        {children}
      </pre>
    </div>
  );
}

/**
 * A named Rust listing — the "P" layer's standard presentation. `caption`
 * states what the snippet does and which crate it belongs to.
 */
export function RustSnippet({
  title,
  caption,
  children,
}: {
  title?: string;
  caption?: string;
  children: React.ReactNode;
}) {
  return (
    <figure className="my-6">
      {title ? (
        <figcaption className="mb-1.5 flex items-center gap-2 text-[12px] font-semibold text-ink">
          <span className="rounded bg-series-2 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-white">
            Rust
          </span>
          {title}
        </figcaption>
      ) : null}
      {children}
      {caption ? (
        <figcaption className="mt-1.5 text-[12.5px] leading-relaxed text-ink-muted">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}
