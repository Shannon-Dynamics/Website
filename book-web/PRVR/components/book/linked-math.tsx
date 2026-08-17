import type { ReactNode } from 'react';

/**
 * Marks an equation whose colour-coded terms are hover targets.
 *
 * Deliberately a *server* component: the actual wiring is done once per page by
 * <TermLinker>, which finds `.term-*` spans after mount. Making this a client
 * component instead would push every equation it wraps across the server/client
 * boundary, and React would serialise all of that rendered KaTeX into the page
 * a second time for hydration — which, at this book's density of mathematics,
 * was most of the page weight.
 *
 * So this renders nothing but a class, and costs nothing.
 */
export function LinkedMath({ children }: { children: ReactNode }) {
  return <div className="pr-linked-math">{children}</div>;
}

/**
 * An inline role swatch for prose: "the measurement (green) says…".
 * <TermLinker> picks it up by the same class the equation terms use.
 */
export function RoleTag({
  role,
  children,
}: {
  role: 'prior' | 'prediction' | 'measurement' | 'posterior' | 'truth';
  children?: ReactNode;
}) {
  return (
    <span
      className={`pr-role-tag term-${role}`}
      style={{ '--pr-role-ink': `var(--pr-${role})` } as React.CSSProperties}
    >
      <span className="pr-role-dot" aria-hidden="true" />
      {children ?? role}
    </span>
  );
}
