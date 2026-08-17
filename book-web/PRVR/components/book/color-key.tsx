export type ColorRole = 'prior' | 'prediction' | 'measurement' | 'posterior' | 'truth';

export interface ColorKeyProps {
  items?: ColorRole[];
}

/**
 * The book's color code. Values are CSS vars, never literals, because each one
 * is redefined for dark mode in global.css.
 */
const ROLES: Record<ColorRole, { label: string; swatch: string; gloss: string }> = {
  prior: { label: 'Prior', swatch: 'var(--pr-prior)', gloss: 'belief before this step' },
  prediction: {
    label: 'Prediction',
    swatch: 'var(--pr-prediction)',
    gloss: 'belief after motion',
  },
  measurement: {
    label: 'Measurement',
    swatch: 'var(--pr-measurement)',
    gloss: 'observation likelihood',
  },
  posterior: {
    label: 'Posterior',
    swatch: 'var(--pr-posterior)',
    gloss: 'belief after the update',
  },
  truth: { label: 'Truth', swatch: 'var(--pr-truth)', gloss: 'ground truth' },
};

const ALL_ROLES: ColorRole[] = ['prior', 'prediction', 'measurement', 'posterior', 'truth'];

/** Legend chips placed under a figure or simulation so colors can be decoded. */
export function ColorKey({ items = ALL_ROLES }: ColorKeyProps) {
  return (
    <ul
      role="list"
      aria-label="Color key"
      className="not-prose my-3 flex flex-wrap items-center gap-x-4 gap-y-1.5"
    >
      {items.map((item) => {
        const role = ROLES[item];
        return (
          <li key={item} className="flex items-center gap-1.5" title={role.gloss}>
            <span
              aria-hidden="true"
              className="size-2.5 shrink-0 rounded-[1px]"
              style={{ backgroundColor: role.swatch }}
            />
            <span className="font-ui text-[0.6875rem] font-medium uppercase tracking-[0.1em] text-fd-muted-foreground">
              {role.label}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
