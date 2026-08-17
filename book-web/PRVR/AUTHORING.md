# Authoring a chapter

How to turn a `Chapter-NN.md` design doc (in the repo root, one level up) into a chapter of the
web book. Read this in full before writing MDX.

## Where things live

| Path | What |
|---|---|
| `content/chapters/chNN-slug.mdx` | The chapter itself. One file. |
| `components/ch/chNN/*.tsx` | That chapter's interactive widgets. Client components. |
| `lib/` | Shared algorithm library — the simulations run **real implementations**, not fakes. |
| `components/book/*` | Prose components (Overview, Derivation, Algorithm, Exercises, References…). |
| `components/viz/*` | Nivo charts + dashboard shell. |
| `components/sim/*` | `WidgetFrame`, `SimCanvas`, `Transport`, `Slider`, `ControlPanel`. |
| `lib/book-structure.ts` | Canonical chapter numbers, slugs, titles. Never invent a slug. |

## Frontmatter (required)

```yaml
---
title: The Bayes Filter
description: One sentence, shown in search results and on the chapter card.
chapter: 5
part: PART II
partTitle: The Bayes Filter Family
difficulty: Foundational      # Foundational | Intermediate | Advanced
readingTime: 45 min
quote: The robot's belief is not a location. It is a distribution over locations.
quoteAuthor: Sebastian Thrun
quoteSource: Probabilistic Robotics (2005)
---
```

The quote is **required** and must be a real, verifiable statement by a genuine robotics
researcher, with the real source. Paraphrase only if you mark it as a paraphrase. Never invent a
quotation and never attribute an invented line to a real person.

## Chapter skeleton

Follow this order. It is the FCP rhythm: hook → conceptual → foundation → practical → exercises.

```mdx
<Overview goals={[...]} prerequisites={[...]}>
Two or three paragraphs: what this chapter is for, and why the reader should care.
</Overview>

## The problem                      ← the hook, with the first widget
## Building intuition               ← Conceptual: widgets first, math named but not yet derived
## The mathematics                  ← Foundation: definitions, theorems, derivations
## The algorithm                    ← <Algorithm> box
## Implementation in Rust           ← Practical
## Putting it together              ← the integration lab / dashboard
## Exercises
## References
```

## Math

KaTeX with `$…$` and `$$…$$`. Global macros are defined in `lib/katex-macros.ts` — use them:
`\bel`, `\belbar`, `\Normal`, `\E`, `\SEtwo`, `\bplus`, `\bminus`, `\T`, `\norm{}`, `\mat{}`.

Color-code equation terms so they match the figures:

```latex
$$
\belbar(x_t) = \int \htmlClass{term-prediction}{p(x_t \mid u_t, x_{t-1})}\,
               \htmlClass{term-prior}{\bel(x_{t-1})}\, dx_{t-1}
$$
```

Available classes: `term-prior`, `term-prediction`, `term-measurement`, `term-posterior`,
`term-truth`.

Note: JSX *attribute strings* never pass through remark-math, so `title="$x$"` renders literally.
Props that may need math (`Derivation`'s `result`, `Exercise`'s `hint` and `solution`) accept a
node — but the simplest approach is to keep math in the component's children, where MDX processes
it normally.

Long algebra goes in a `<Derivation>`:

```mdx
<Derivation title="Deriving the Kalman gain" result="K_t = \Sigma_t H_t^\mathsf{T} S_t^{-1}">
Step-by-step algebra here. The reader can skip this entirely on a first pass.
</Derivation>
```

## Code

Rust only, in fenced blocks with a title and (optionally) highlighted lines:

````mdx
```rust title="crates/pr-core/src/filters/bayes.rs" {4-7}
pub trait BayesFilter {
    type Control;
    type Measurement;
    fn predict(&mut self, u: &Self::Control);
    fn correct(&mut self, z: &Self::Measurement);
}
```
````

Rules for code:
- It must be real, compilable-in-spirit Rust using the book's crate stack (`nalgebra` 0.35,
  `rand` 0.9, `faer` 0.24, `parry2d` 0.30, `factrs` 0.3, `petgraph` 0.8).
- Show types. `SVector<f64, 3>`, not `Vec<f64>`, when the dimension is fixed.
- Comment the *why*, never the *what*.
- Every chapter has at least three substantial listings: the core type, the algorithm, and a
  worked example with its expected output.

## Widgets

Every widget is a client component in `components/ch/chNN/`, wrapped in `WidgetFrame`, using
`SimCanvas` + `useSimulation` for animation and the real algorithms from `lib/`.

```tsx
'use client';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { Transport, Slider, ControlPanel } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';

export function HallwayBeliefMachine() {
  const sim = useSimulation<State>({ init, step, fps: 12 });
  return (
    <WidgetFrame
      id="w5.1"
      title="Hallway Belief Machine"
      teaches="Sensing sharpens the belief; moving smears it."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      caption={<>What to notice, and what to try changing.</>}
    >
      <SimCanvas world={...} draw={...} deps={[sim.tick]} ariaLabel="..." />
      <ControlPanel>
        <Slider label="Sensor noise σ" role="measurement" value={...} min={0.01} max={1} onChange={...} />
      </ControlPanel>
      <Transport {...sim} onToggle={sim.toggle} onStep={sim.stepOnce} onReset={sim.reset} onReseed={sim.reseed} />
    </WidgetFrame>
  );
}
```

Then import it at the top of the MDX file and place it in the prose.

Widget rules, in priority order:

1. **Autoplay a sensible default.** The reader must learn something without touching anything.
2. **One idea per widget.** If it needs a paragraph to explain the controls, split it.
3. **Foreground one parameter.** Others can exist, but one slider is the point of the widget.
4. **Use the book color code**, always via `var(--pr-*)` — never a literal hex, never a color that
   only reads in one theme.
5. **Name the misconception it kills** in `teaches`.
6. IDs come from the chapter design doc (`w5.1`, `w5.2`, …) and must match the design's manifest.

## Explorable prose

Two devices let the prose itself become interactive. Both degrade correctly in the print edition,
so use them freely.

### `<Scrub>` — a number the reader can drag

```mdx
import { Scrub } from '@/components/book/scrub';

The wheels deliver the commanded step plus an error of about
<Scrub id="ch09.alpha1" value={0.12} min={0.01} max={0.5} step={0.01}
       unit="m" role="prediction" label="motion noise sigma" /> — drag it up
and watch every peak in the belief widen.
```

The `id` is a namespaced key (`chNN.name`). A widget reads the same key:

```tsx
import { useExplorable, setValue } from '@/lib/explorable/store';

const sigma = useExplorable('ch09.alpha1', params.sigma);   // falls back to its own slider
```

Rules:

- **The sentence must still read if nobody drags anything.** The printed value is the value the
  text claims; scrubbing is an invitation to ask "what if", not a blank to fill in.
- Give it a `role` when the quantity belongs to one of the five estimation colours, so the number
  matches the curve it controls.
- If a widget also exposes the value as a slider, have the slider call `setValue` with the same
  key — then the sentence and the panel are two views of one parameter, not two parameters.

### `<LinkedMath>` — point at a term, light up the figure

Wrap any display equation whose terms are colour-coded:

```mdx
<LinkedMath>
$$
\htmlClass{term-posterior}{\bel(x_t)} = \eta\,
\htmlClass{term-measurement}{p(z_t \mid x_t)}\,
\htmlClass{term-prediction}{\belbar(x_t)}
$$
</LinkedMath>
```

Hovering (or tabbing to) a tinted term mutes every other estimation colour in every figure on the
page. **No widget code is required**: `SimCanvas` fades the non-hovered roles in the palette it
hands to `draw`, so any widget that draws its prior in `p.prior` and its measurement in
`p.measurement` — which is all of them — already participates.

Use `<RoleTag role="measurement">the measurement</RoleTag>` to get the same link from a sentence.

## Active exercises

Three components turn the exercises from prose into something the reader does. Outcomes persist in
`localStorage`, per exercise id.

```mdx
<Exercise level="C" difficulty={1} title="Predict, then check">
  <Predict
    id="ch05.e4"
    question="Set motion noise to maximum and sensing off. After twenty steps, what does the belief look like?"
    options={[
      { label: 'A single peak that has drifted', because: 'Prediction is unbiased — it loses certainty, not accuracy.' },
      { label: 'Flat, over the whole corridor', correct: true, because: 'Repeated convolution drives any belief on a loop toward uniform.' },
      { label: 'Three peaks, one per door', because: 'Doors enter only through the measurement model.' },
    ]}
  >
  Follow-up shown once they commit.
  </Predict>
</Exercise>
```

- **`<Predict>`** — the reader must commit before the explanation appears. Every option gets a
  `because`, including the wrong ones: a wrong answer is only useful if it explains itself. Use
  this for every exercise whose text says "predict, then verify".
- **`<CheckAnswer id prompt answer tolerance unit>`** — a numeric box with an explicit tolerance.
  Set the tolerance to what a reader doing the algebra by hand would plausibly write; marking
  0.464 wrong because the answer is 0.4637 teaches arithmetic pedantry, not estimation.
- **`<Hints>`** — children are revealed one at a time. Write them as a staircase: the first names
  the relevant idea, the last is nearly the answer.

## Dashboards

For chapters where the point is *monitoring* an algorithm rather than watching a scene, use the
dashboard components: `Dashboard`, `DashboardPanel`, `StatTile`, and the Nivo charts
(`LineChart`, `BarChart`, `HeatMap`, `ScatterChart`, `NetworkGraph`). Series accept a
`role` prop that applies the book color automatically.

Good dashboard candidates: filter innovation/NEES over time (Ch. 6, 11, 14), particle effective
sample size (Ch. 8, 12), map entropy during exploration (Ch. 13, 24), optimizer cost per
iteration (Ch. 15, 16).

## Exercises

Three to six per chapter, spanning the three passes, using the `level` prop:

```mdx
<Exercises>
  <Exercise level="F" difficulty={2} title="Marginalize the joint">
    Body. Math is fine here.
    <details>…</details>
  </Exercise>
</Exercises>
```

`F` = derive something. `C` = predict what a widget will do, then check. `P` = write Rust.

## References

Every chapter ends with a real bibliography — a mix of the foundational papers and **recent**
work (2020–2026). Verify each one exists before citing it: authors, year, title, venue, and a URL
or DOI. Add a `note` saying why the reference matters.

```mdx
<References>
  <Reference
    authors="Dellaert, F. and Kaess, M."
    year={2017}
    title="Factor Graphs for Robot Perception"
    venue="Foundations and Trends in Robotics 6(1–2)"
    doi="10.1561/2300000043"
    note="The standard modern treatment; Chapter 15 follows its formulation."
  />
</References>
```

**Never cite a paper you have not verified exists.** Fabricated citations are the one failure
mode that would discredit the whole book.

## Voice

Rigorous but narrative-first. Concrete before abstract. Say what breaks and when. Prefer the
active voice and a real example over a general claim. The reader is smart and busy: never pad,
never hedge, and never write "as we saw earlier" without a link.
