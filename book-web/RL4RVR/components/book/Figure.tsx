/**
 * Static figure wrapper — every interactive widget must also degrade to a
 * captioned static figure (accessibility and print). Use for inline SVG
 * diagrams authored directly in MDX.
 */
export function Figure({
  caption,
  label,
  children,
  full = false,
}: {
  caption?: string;
  label?: string;
  children: React.ReactNode;
  /** Let the figure use the full column width rather than the reading measure. */
  full?: boolean;
}) {
  return (
    <figure className={`my-6 ${full ? '' : 'mx-auto'}`}>
      <div className="scroll-x thin-scroll rounded-xl border border-hairline bg-surface p-4">
        {children}
      </div>
      {(caption || label) && (
        <figcaption className="mt-2 text-[12.5px] leading-relaxed text-ink-muted">
          {label ? <span className="font-semibold text-ink-secondary">{label}. </span> : null}
          {caption}
        </figcaption>
      )}
    </figure>
  );
}
