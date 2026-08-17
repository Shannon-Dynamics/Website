# Reinforcement Learning for Robotics — The FCP Way

The interactive web book. Twenty-two chapters teaching reinforcement learning for robotics through three interleaved layers: **Foundation** (complete mathematical formalism), **Conceptual** (an interactive simulation for every hard idea), and **Practical** (Rust implementations).

## Running it

```bash
npm install
npm run dev      # http://localhost:3000
```

Requires Node 18.18+ (Next.js 15). Other scripts:

```bash
npm run build      # static export into out/; also the correctness gate
npm run typecheck  # tsc --noEmit
npm run lint
```

## Deployment

### GitHub Pages (automated)

Pushing to `main` builds and publishes the book via
[.github/workflows/deploy.yml](../.github/workflows/deploy.yml). The site lands at:

```
https://shannon-dynamics.github.io/ReinforcementLearning4Robotics/
```

One-time setup in the repository: **Settings → Pages → Build and deployment →
Source: GitHub Actions**. Nothing else is required; the workflow requests the
`pages: write` and `id-token: write` permissions it needs.

Two details that break most Next.js Pages deployments, both handled here:

- **A project site is served from `/<repo>/`, not `/`.** The workflow passes
  `NEXT_PUBLIC_BASE_PATH` from `actions/configure-pages`, and
  [next.config.mjs](next.config.mjs) applies it as both `basePath` and
  `assetPrefix` — the latter matters because the Web Worker chunk is resolved
  against it. Because the value comes from the action, renaming the repo (or
  moving to an `<org>.github.io` user site, where the prefix is empty) needs no
  code change.
- **Jekyll would delete `_next/`,** since it ignores paths starting with an
  underscore. `public/.nojekyll` ships in every export, and the workflow writes
  one too.

To reproduce a Pages build locally:

```bash
npm run build:pages
# then serve it from a matching subdirectory, because paths are absolute:
mkdir -p /tmp/pages && cp -r out /tmp/pages/ReinforcementLearning4Robotics
(cd /tmp/pages && python3 -m http.server 8110)
# open http://127.0.0.1:8110/ReinforcementLearning4Robotics/
```

Serving `out/` directly at the domain root will 404 on every asset — that is
expected, not a bug, because the build was told it lives under a prefix.

### Anywhere else: no server, ever

`npm run build` produces a **fully static site in `out/`** — plain HTML, JS and
CSS. There are no API routes, no server actions, no middleware and no image
optimizer, so nothing needs a Node process or a serverless function at runtime.
Drop `out/` on any static host (S3, Netlify, nginx, a USB stick) and the whole
book works.

Everything the book computes, it computes in the reader's browser:

| Work | Where it happens |
|---|---|
| MDX → HTML, KaTeX math, Shiki highlighting | Build time |
| Value iteration, Q-learning, bandit testbeds, kinematics | Main thread, client |
| Neural training, gait optimization, randomized transfer | **Web Worker**, client |

Verified by serving the export from a subdirectory and confirming in a real
browser that all 22 chapters render, the workers run, client-side navigation
keeps the base path, and no request 404s.

## Layout

```
rl4r-web/
├── app/                      # Next.js App Router
│   ├── page.tsx              # landing
│   ├── about/                # the method, palette, toolchain
│   └── chapters/[slug]/      # chapter renderer (SSG, one route per chapter)
├── content/chapters/         # the book itself — 22 MDX files
├── components/
│   ├── layout/               # header, theme provider, chapter navigation
│   ├── book/                 # MDX component map, callouts, quotes, exercises
│   ├── viz/                  # Nivo chart wrappers + chart chrome
│   └── sim/                  # 25 interactive simulations
└── lib/
    ├── chapters.ts           # the TOC as typed data (mirrors ../TOC.md)
    ├── theme.ts              # the validated palette and Nivo theme
    ├── mdx.ts                # MDX loading and heading extraction
    └── rl/                   # the simulation engine (see below)
```

### Interaction, not just parameters

A slider that redraws a chart is parameter tweaking. These widgets give the
reader something to *do*, with consequences:

| Widget | What you do |
|---|---|
| `BanditTestbed` (Ch 3) | Pull the arms yourself; your regret is raced against UCB and ε-greedy on the same problem |
| `PendleSim` (Ch 2) | Grab the bob and fling it, or knock it, and watch the controller recover |
| `WarehouseEditor` (Ch 4) | Drag-paint walls and slip patches; value iteration re-solves on every edit |
| `GpiDashboard` (Ch 5) | Click a cell to corrupt its value, then step forward and watch GPI repair it |
| `TdDashboard` (Ch 6) | Drive Rusty with the arrow keys, blind, and see the agent overtake your best run |
| `RewardDesigner` (Ch 14) | Compose a reward and watch the optimal policy exploit it |
| `DmpSculptor` (Ch 17) | Draw a demonstration freehand; LWR fits the primitive to your curve |
| `ReacherKinematics` (Ch 13) | Drag the end-effector; IK solves and the manipulability ellipsoid responds |
| `GraspWrench` (Ch 20) | Drag contact points until force closure fails |
| `SharedAutonomy` (Ch 21) | Set the blend and feel task success trade against your sense of control |

### The simulation engine

Almost every interactive runs the real algorithm rather than a canned animation.
The exception is labelled as such in its own interface: `PipelineSwitcher`
(Ch 19) compares navigation architectures whose training cannot run in a
browser, so its curves encode the survey's qualitative findings. Everything
else — including the reward mixer's gait optimizer, the replay/target-network
ablation, the behaviour-cloning drift, the dynamics ensemble and the
randomization transfer — computes what it displays.

[lib/rl/](lib/rl/) contains:

| Module | Contents |
|---|---|
| `random.ts` | Seeded mulberry32 RNG, Gaussian/categorical/Beta sampling |
| `gridworld.ts` | Rusty's warehouse MDP — both the white-box `transitions` and black-box `step` views |
| `dp.ts` | Policy evaluation, policy iteration, value iteration — as generators yielding one snapshot per sweep, so the UI can animate them |
| `bandits.ts` | ε-greedy, UCB1, gradient bandit, Thompson sampling, and the 10-armed testbed harness |
| `td.ts` | MC prediction, TD(0), SARSA, Expected SARSA, Q-learning, Double Q, Sarsa(λ) |
| `pendulum.ts` | Pendle's dynamics with Euler / semi-implicit / RK4 integrators and an energy-shaping controller |
| `nn.ts` | A minimal MLP with SGD + momentum — enough for behaviour cloning and one-step dynamics models |
| `tiles.ts` | Tile coding and semi-gradient SARSA, Chapter 8's workhorse |
| `dynamics.ts` | Learned dynamics ensembles (bootstrap resampled) and a CEM planner |
| `rewardlab.ts` | A gridworld whose reward the reader composes, solved exactly so reward hacking is demonstrably the optimum rather than a training failure |
| `walker.ts` | A planar reduction of Ferris — leg spring-dampers, friction cone, contact scheduling — plus an evolution strategy that optimizes a gait against a reward |

Heavy simulations run in a **Web Worker** ([lib/sim/worker.ts](lib/sim/worker.ts)),
driven by the `useSimulation` hook, so training never blocks the page. Widget
settings are mirrored into the URL by `useWidgetState`, which makes any
configuration shareable and lets prose link to a specific view.

Constants match the chapter designs exactly — the warehouse is 12×9 with `p_slip = 0.2`, `γ = 0.95`, rewards +25 / −1 / −10, as fixed in Chapter 4 and reused through Chapter 7.

## Writing a chapter

Chapters are MDX in `content/chapters/chNN-slug.mdx`. Frontmatter carries the title and the researcher epigraph:

```mdx
---
title: Chapter Title
chapter: 5
quote:
  text: The quotation.
  author: Who said it
  affiliation: Where they work
  source: Which work it comes from
---
```

Components available in MDX without importing (see [components/book/MdxContent.tsx](components/book/MdxContent.tsx)):

- **Structure** — `ChapterOverview`, `Callout`, `Theorem`, `Proof`, `Figure`
- **Code** — `RustSnippet` wrapping a fenced ` ```rust ` block
- **Ending** — `Exercises`, `CodingTask`, `References`
- **Charts** — `LineChart`, `BarChart`, `StatRow`, `StatTile`
- **Simulations** — `RustyDrive`, `SuccessLevels`, `PendleSim`, `ContractionDemo`, `BanditTestbed`, `MdpExplorer`, `GpiDashboard`, `TdDashboard`, `LambdaDial`, `DeadlyTriad`, `ReplayBuffer`, `PolicyGradientLab`, `EntropyDial`, `ModelBiasFan`, `CurseOfDimensionality`, `DomainRandomization`, `CovariateShift`, `DmpSculptor`, `RewardMixer`, `PipelineSwitcher`, `GraspWrench`, `ReacherKinematics`, `SharedAutonomy`, `MissionControl`, `WarehouseEditor`, `RewardDesigner`

Math is KaTeX: `$inline$` and `$$display$$`. Code blocks are highlighted at build time by Shiki, so no highlighting JS ships to the reader.

Adding a chapter means writing the MDX file and adding its entry to `lib/chapters.ts`. The route, navigation, and static generation follow automatically.

## Design rules

**The palette is validated, not chosen.** Both light and dark sets pass the lightness band, chroma floor, colorblind separation (Machado 2009 at severity 1.0), normal-vision floor, and contrast checks. Do not add colors — use the eight categorical slots in order, the sequential ramp for magnitude, and the diverging pair for polarity. Scatter-type charts cap at three series.

**Encoding is consistent book-wide.** Value functions get the sequential ramp; policies get directional arrows in slot 1; TD error, advantage and signed reward get the diverging pair; uncertainty gets a fan or band. A color means the same thing in Chapter 20 as in Chapter 4.

**Accessibility is structural.** Every chart carries a table view. Every callout pairs an icon with a text label, so meaning never rests on color alone. Every simulation has a caption describing what it shows. Wide content scrolls inside its own container; the page body never scrolls horizontally.

**Both themes are selected, not flipped.** The dark palette is its own set of steps chosen against the dark surface. Tokens live in [app/globals.css](app/globals.css); nothing else defines a color.

## Relationship to the design docs

The parent directory holds the book's design: `TOC.md` is the authoritative structure, and `Chapter-1.md` … `Chapter-22.md` are the per-chapter design documents specifying sections, widgets, Rust plans and exercises. `CLAUDE.md` holds the development conventions.

This web app implements those designs. When they disagree, the design docs are the source of truth for *what* a chapter covers; this app is the source of truth for *how* it is presented.
