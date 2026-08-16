#!/usr/bin/env python3
"""Generate the three dedicated Library pages from index.html.

Head (all CSS), the pattern <defs>, the nav, the CTA banner, the footer and the
shared scripts are lifted straight out of index.html so the sub-pages can
never drift from the main page's design system. Re-run after editing index.html.

(Sourced from index.html, the promoted live homepage, not index-v3.html — the
draft variant kept alongside it. index.html also no longer has a #work
section, so these sub-pages naturally lose the "What we do" nav link too.)
"""
import re, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path("/Users/tamariaartpinkan/Documents/Shannon/web-upgrade")
src = (SRC / "index.html").read_text()

# ── shared chunks ───────────────────────────────────────────────────────────
head = src[src.index("<head>"):src.index("</head>") + len("</head>")]

defs = src[src.index('<svg width="0" height="0"') if '<svg width="0" height="0"' in src
           else src.index("<svg", src.index("<body")):src.index("</svg>", src.index("<body")) + 6]

nav = src[src.index('<header id="nav">'):src.index("</header>") + 9]
# sub-pages live beside index.html, so in-page anchors must be qualified
nav = re.sub(r'href="#(work|capabilities|showcase|library|ecosystem)"',
             r'href="index.html#\1"', nav)
# match on the leading class token (nav-logo), not the full class string — an
# exact-string .replace() here silently no-ops the moment index.html's own
# class list changes shape, which is exactly what broke the logo link
old_logo_count = nav.count('href="#" class="nav-logo')
nav = re.sub(r'href="#"(\s+class="nav-logo)', r'href="index.html"\1', nav)
assert old_logo_count == 1, f"expected exactly one nav-logo href=\"#\" in the source header, found {old_logo_count}"

cta_marker = "<!-- ═══════════ CTA BANNER" # exact suffix drifts (e.g. "— contact form") — match the stable prefix
cta_start = src.index(cta_marker)
foot_end = src.index("</footer>") + 9
cta_footer = src[cta_start:foot_end]

scripts_head = ('    <script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>\n'
                '    <script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/ScrollTrigger.min.js"></script>\n'
                '    <script src="https://cdn.jsdelivr.net/npm/@studio-freight/lenis@1.0.42/dist/lenis.min.js"></script>')


def block(start_marker, end_marker=None):
    a = src.index(start_marker)
    b = src.index(end_marker, a) if end_marker else len(src)
    return src[a:b]


js_core = block("/* ---------- smooth scroll ---------- */", "/* ---------- capability tabs ---------- */")
js_halftone = block("/* ---------- animated halftone moiré texture + magnetic hover lens ---------- */",
                    "    </script>")

# ── page-specific styles ────────────────────────────────────────────────────
PAGE_CSS = """
        /* ═══ sub-page banner ═══ */
        /* .tex tex-bottom (defined in the shared sheet) lays the legibility
           scrim between the dot canvas and this copy */
        .lp-banner { position: relative; height: 46vh; min-height: 340px; background: #10181b; }
        .lp-banner-in { position: relative; max-width: 1240px; margin: 0 auto; height: 100%; padding: 0 24px;
                        display: flex; flex-direction: column; justify-content: flex-end; padding-bottom: 46px; }
        .lp-crumb { display: flex; align-items: center; gap: 10px; margin-bottom: 18px;
                    font-family: "JetBrains Mono", monospace; font-size: 10px; letter-spacing: .2em; color: rgba(255,255,255,.72); }
        .lp-crumb a { color: rgba(255,255,255,.72); transition: color .25s; }
        .lp-crumb a:hover { color: #fff; }
        .lp-crumb span.on { color: #A9C3FF; }
        .lp-title { color: #fff; font-size: clamp(30px, 4vw, 52px); font-weight: 500; letter-spacing: -.02em; line-height: 1.08; max-width: 18ch; }
        .lp-sub { color: rgba(255,255,255,.82); font-size: 14px; line-height: 1.7; max-width: 56ch; margin-top: 16px; }
        .lp-body { background: #fff; padding: 0 24px 24px; }
        .lp-wrap { max-width: 1240px; margin: 0 auto; }
        .lp-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; padding: 30px 0 26px;
                  font-family: "JetBrains Mono", monospace; font-size: 10px; letter-spacing: .18em; color: rgba(16,24,27,.42); }
        .lp-count { margin-left: auto; }

        /* filter chips */
        .lp-filter { border: 1px solid rgba(16,24,27,.16); border-radius: 9999px; padding: 7px 15px;
                     font-family: "JetBrains Mono", monospace; font-size: 9.5px; letter-spacing: .16em;
                     color: rgba(16,24,27,.55); transition: background .25s, color .25s, border-color .25s; }
        .lp-filter:hover { border-color: rgba(76,126,255,.5); color: #4C7EFF; }
        .lp-filter[aria-pressed="true"] { background: #10181b; border-color: #10181b; color: #fff; }

        /* ═══ publications list ═══ */
        .pp-row { display: grid; gap: 6px 28px; padding: 26px 14px; border-top: 1px solid rgba(16,24,27,.12);
                  transition: background .3s, padding-left .3s; }
        .pp-row:hover { background: rgba(16,24,27,.025); padding-left: 22px; }
        .pp-row[hidden] { display: none; }
        @media (min-width: 900px) { .pp-row { grid-template-columns: 92px 132px 1fr 96px; align-items: start; } }
        .pp-year { font-family: "JetBrains Mono", monospace; font-size: 11px; letter-spacing: .14em; color: rgba(16,24,27,.35); padding-top: 3px; }
        .pp-tag { font-family: "JetBrains Mono", monospace; font-size: 9px; letter-spacing: .14em;
                  border: 1px solid rgba(76,126,255,.35); color: #4C7EFF; border-radius: 9999px;
                  padding: 3px 9px; justify-self: start; white-space: nowrap; }
        .pp-h { font-size: 17px; font-weight: 600; letter-spacing: -.01em; line-height: 1.35; }
        .pp-by { font-size: 12px; color: rgba(16,24,27,.5); margin-top: 5px; }
        .pp-ex { font-size: 13px; line-height: 1.7; color: rgba(16,24,27,.62); margin-top: 11px; max-width: 68ch; }
        .pp-go { font-size: 12.5px; color: #4C7EFF; white-space: nowrap; padding-top: 3px; }
        .pp-row:hover .pp-go { text-decoration: underline; text-underline-offset: 4px; }

        /* ═══ books grid ═══ */
        .bk-grid { display: grid; gap: 40px 30px; padding-bottom: 20px; }
        @media (min-width: 640px)  { .bk-grid { grid-template-columns: repeat(2, 1fr); } }
        /* 4 not 3 — each cover was ~393px at 3-up, big enough to dominate the
           page; 4-up brings it to ~288px, roughly the same ~27% cut the detail-
           page cover took (218px -> 160px) so the two pages feel consistent */
        @media (min-width: 1000px) { .bk-grid { grid-template-columns: repeat(4, 1fr); } }
        .bk { display: flex; flex-direction: column; height: 100%; }
        /* face-on here — the -46deg shelf tilt belongs to the drifting rail on
           the home page, not to a catalogue you are meant to read */
        .bk .book-cover {
            position: relative; display: block; width: 100%; aspect-ratio: 3 / 4.2; height: auto;
            border-radius: 3px 8px 8px 3px; transform: none; overflow: hidden;
            box-shadow: 0 22px 44px -26px rgba(16,24,27,.75);
            transition: transform .5s cubic-bezier(.76,0,.24,1), box-shadow .5s ease;
        }
        .bk .book-cover:hover { transform: translateY(-6px); box-shadow: 0 30px 54px -26px rgba(16,24,27,.6); }
        .bk .book-cover::after { content: ''; position: absolute; top: 0; bottom: 0; left: 0; width: 17px; z-index: 3;
            background: linear-gradient(90deg, rgba(0,0,0,.34), rgba(0,0,0,.05) 70%, rgba(255,255,255,.1)); }
        .bk .book-cover-t { left: 30px; right: 18px; bottom: 20px; font-size: 14px; font-weight: 600; line-height: 1.35; }
        /* grow so the meta line lands on the same baseline across a row,
           however long each blurb runs */
        .bk-meta { padding-top: 16px; display: flex; flex-direction: column; flex: 1; }
        .bk-t { font-size: 14.5px; font-weight: 600; letter-spacing: -.01em; }
        .bk-t a { color: inherit; }
        .bk-t a:hover { color: #4C7EFF; }
        .bk-a { font-size: 12px; color: rgba(16,24,27,.5); margin-top: 3px; }
        .bk-d { font-size: 12.5px; line-height: 1.65; color: rgba(16,24,27,.6); margin-top: 10px; }
        .bk-f { display: flex; align-items: center; gap: 14px; margin-top: auto; padding-top: 14px;
                font-family: "JetBrains Mono", monospace; font-size: 9.5px; letter-spacing: .14em; color: rgba(16,24,27,.4); }
        .bk-f a { margin-left: auto; color: #4C7EFF; }
        .bk-f a:hover { text-decoration: underline; text-underline-offset: 3px; }

        /* ═══ open-source cards ═══ */
        .os-h { font-family: "JetBrains Mono", monospace; font-size: 10px; letter-spacing: .2em;
                color: rgba(16,24,27,.42); padding: 34px 0 16px; }
        .os-grid { display: grid; gap: 14px; }
        @media (min-width: 800px) { .os-grid { grid-template-columns: repeat(2, 1fr); } }
        .os-card { position: relative; border: 1px solid rgba(16,24,27,.12); border-radius: 14px;
                   padding: 22px 24px 20px; background: #fff; display: flex; flex-direction: column;
                   transition: border-color .35s, box-shadow .35s, transform .35s cubic-bezier(.76,0,.24,1); }
        .os-card:hover { border-color: rgba(76,126,255,.34); box-shadow: 0 20px 40px -28px rgba(16,24,27,.5); transform: translateY(-3px); }
        .os-name { font-family: "JetBrains Mono", monospace; font-size: 13px; font-weight: 500; letter-spacing: .02em; }
        .os-d { font-size: 12.5px; line-height: 1.7; color: rgba(16,24,27,.62); margin-top: 10px; }
        .os-f { display: flex; align-items: center; gap: 14px; margin-top: auto; padding-top: 18px;
                font-family: "JetBrains Mono", monospace; font-size: 9.5px; letter-spacing: .14em; color: rgba(16,24,27,.4); }
        .os-f a { margin-left: auto; color: #4C7EFF; }
        .os-f a:hover { text-decoration: underline; text-underline-offset: 3px; }
        .os-dot { width: 7px; height: 7px; border-radius: 9999px; background: #4C7EFF; flex-shrink: 0; }

        /* sibling links out to the other two Library pages */
        .lp-more { display: grid; gap: 14px; padding: 46px 0 10px; border-top: 1px solid rgba(16,24,27,.12); margin-top: 40px; }
        @media (min-width: 700px) { .lp-more { grid-template-columns: 1fr 1fr; } }
        .lp-more a { display: flex; align-items: baseline; gap: 14px; padding: 20px 22px;
                     border: 1px solid rgba(16,24,27,.12); border-radius: 14px;
                     transition: border-color .3s, background .3s, padding-left .3s; }
        .lp-more a:hover { border-color: rgba(76,126,255,.34); background: rgba(76,126,255,.03); padding-left: 30px; }
        .lp-more b { font-size: 15px; font-weight: 600; }
        .lp-more small { display: block; font-size: 12px; color: rgba(16,24,27,.55); margin-top: 4px; font-weight: 400; }
        .lp-more span { font-family: "JetBrains Mono", monospace; font-size: 10px; letter-spacing: .2em; color: #4C7EFF; }
        .lp-more em { margin-left: auto; font-style: normal; color: rgba(16,24,27,.3); }
"""

PAGES = {
    "publications": dict(num="LIB.01", label="PUBLICATIONS", title="Papers, preprints and field notes",
                         sub="Everything the team has written down — peer-reviewed work on simulation fidelity and "
                             "agentic autonomy, alongside the engineering notes we keep while building simulated machines."),
    "books": dict(num="LIB.02", label="BOOKS", title="Long-form from Shannon Press",
                  sub="Books written by the Shannon team on Rust and robotics — "
                      "the long-form counterpart to the papers and notes."),
    "open-source": dict(num="LIB.03", label="OPEN SOURCE", title="Repositories, guides and API reference",
                        sub="The Rust we have published, and the documentation that goes with every Shannon system. "
                            "Open by default — financial and safety tooling should be inspectable."),
}
SIBLINGS = {
    "publications": [("books", "LIB.02", "Books", "Long-form from Shannon Press"),
                     ("open-source", "LIB.03", "Open Source", "Repositories, guides and API reference")],
    "books": [("publications", "LIB.01", "Publications", "Papers, preprints and field notes"),
              ("open-source", "LIB.03", "Open Source", "Repositories, guides and API reference")],
    "open-source": [("publications", "LIB.01", "Publications", "Papers, preprints and field notes"),
                    ("books", "LIB.02", "Books", "Long-form from Shannon Press")],
}

# ── content ─────────────────────────────────────────────────────────────────
PUBS = [
    ("2026", "PAPER", "High-Fidelity Contact Dynamics for Sim-to-Real Transfer", "Shannon Robotics Group",
     "Contact is where simulation usually lies. We model compliant contact with measured material response instead of "
     "tuned stiffness, and show policies trained in the simulator holding their grasp success rate on hardware without retuning."),
    ("2026", "PAPER", "Deterministic Replay for Multi-Body Simulation at 0.5ms", "Shannon Robotics Group",
     "A solver that produces bit-identical trajectories across machines and thread counts, so a failure seen in CI can be "
     "reproduced exactly on a workstation. We describe the fixed-point accumulation scheme and its cost."),
    ("2025", "PREPRINT", "Agentic Autonomy Under Partial Observability", "Shannon Dynamics · RantAI",
     "A planner that reasons about what it cannot see. We give the agent an explicit budget for information-gathering "
     "actions and measure how that budget trades against task completion time across 400 simulated warehouse layouts."),
    ("2025", "PREPRINT", "Domain Randomisation Budgets and the 2.1% Gap", "Shannon Robotics Group",
     "Randomising everything is wasteful and randomising too little does not transfer. We fit a per-parameter sensitivity "
     "model and allocate randomisation where it actually moves the reality gap."),
    ("2025", "NOTE", "What Breaks When You Simulate a Factory", "Field notes from the team",
     "Everything fits in CAD and nothing fits on the floor. A write-up of the six assumptions that failed first when we "
     "rehearsed a real production line — and the validation step we now run before any asset enters the simulator."),
    ("2025", "NOTE", "Reading a Sim-to-Real Report Honestly", "Field notes from the team",
     "Most transfer numbers are quoted without the conditions that produced them. The checklist we apply to our own "
     "results before they leave the building, and the three ways a good number can still mislead."),
    ("2024", "NOTE", "Rust in the Control Loop", "Field notes from the team",
     "Why the borrow checker earns its keep at 1 kHz. Allocation discipline, deterministic teardown, and the places we "
     "still reach for unsafe — with the latency numbers that justified moving the loop off C++."),
    ("2024", "PAPER", "GPU-Batched Environments for Industrial Manipulation", "Shannon Robotics Group",
     "Scheduling 256 environments per node without the tail latency that usually eats the gains. Memory layout, "
     "synchronisation points, and where the batch size stops paying for itself."),
    ("2024", "NOTE", "Assets Are the Hard Part", "Field notes from the team",
     "A simulator is only as good as what you put in it. How we validate imported CAD and URDF, what we reject, and the "
     "asset library conventions that keep a scene reproducible a year later."),
]

# slug is the last field — must match an existing book-<slug>.html detail page
BOOKS = [
    ("Probabilistic Robotics via Rust", "Shannon Press", "bc-blue bc-v2", "2026 · 26 CHAPTERS",
     "An interactive book on probabilistic robotics — Bayes filters, Kalman and particle filters, localization, "
     "occupancy grids, and SLAM as sparse least squares — culminating in a rover that maps a floorplan it has never seen.",
     "probabilistic-robotics-via-rust"),
    ("Reinforcement Learning for Robotics", "Shannon Press", "bc-ink bc-v3", "2026 · 22 CHAPTERS",
     "Reinforcement learning for robotics the FCP way — full mathematical foundations, interactive simulations, and "
     "production Rust, from multi-armed bandits through PPO/SAC to a quadruped that learns to walk.",
     "reinforcement-learning-for-robotics"),
]

REPOS = [
    ("octomap-rs", "RUST", "APACHE-2.0",
     "Octree occupancy mapping in Rust — probabilistic updates, ray casting, and serialisation, with no allocation on "
     "the update path.", "https://github.com/Shannon-Dynamics", "Repository ↗"),
    ("scan3d-rs", "RUST", "APACHE-2.0",
     "3D scan reconstruction — point-cloud registration through to a watertight mesh the simulator can load directly as "
     "a collision asset.", "https://github.com/Shannon-Dynamics", "Repository ↗"),
]
DOCS = [
    ("shannon-sdk / docs", "DOCUMENTATION", "VERSIONED",
     "Guides, tutorials and quickstarts — from importing a URDF to running your first closed-loop rehearsal against a "
     "simulated cell.", "#", "Read the docs ↗"),
    ("api-reference", "REFERENCE", "AUTO-GENERATED",
     "Typed interfaces for every Shannon system — generated from source, so the reference and the shipped binary never "
     "drift apart.", "#", "Browse API ↗"),
]


def pubs_body():
    chips = "".join(
        f'<button type="button" class="lp-filter" data-filter="{v}" aria-pressed="{"true" if v == "ALL" else "false"}">{v}</button>'
        for v in ("ALL", "PAPER", "PREPRINT", "NOTE"))
    rows = "\n".join(f"""                    <article class="pp-row" data-type="{t}" data-reveal>
                        <span class="pp-year">{y}</span>
                        <span class="pp-tag">{t}</span>
                        <div>
                            <h2 class="pp-h">{title}</h2>
                            <p class="pp-by">{by}</p>
                            <p class="pp-ex">{ex}</p>
                        </div>
                        <a class="pp-go" href="#">Read ↗</a>
                    </article>""" for y, t, title, by, ex in PUBS)
    return f"""                <div class="lp-bar" data-reveal>
                    {chips}
                    <span class="lp-count" data-count>{len(PUBS)} ENTRIES</span>
                </div>
                <div data-pp-list>
{rows}
                </div>"""


def books_body():
    cards = "\n".join(f"""                    <article class="bk" data-reveal>
                        <a href="book-{slug}.html" class="book-cover {cls}"><span class="book-cover-t">{t}</span></a>
                        <div class="bk-meta">
                            <h2 class="bk-t"><a href="book-{slug}.html">{t}</a></h2>
                            <p class="bk-a">{a}</p>
                            <p class="bk-d">{d}</p>
                            <p class="bk-f"><span>{meta}</span><a href="book-{slug}.html">Details ↗</a></p>
                        </div>
                    </article>""" for t, a, cls, meta, d, slug in BOOKS)
    return f"""                <div class="lp-bar" data-reveal>
                    <span>SHANNON PRESS</span>
                    <span class="lp-count">{len(BOOKS)} TITLES</span>
                </div>
                <div class="bk-grid">
{cards}
                </div>"""


def oss_body():
    def cards(items):
        return "\n".join(f"""                    <article class="os-card" data-reveal>
                        <p class="os-name"><span class="os-dot" style="display:inline-block;margin-right:9px"></span>{n}</p>
                        <p class="os-d">{d}</p>
                        <p class="os-f"><span>{a}</span><span>{b}</span><a href="{href}"{
                        ' target="_blank" rel="noopener"' if href.startswith("http") else ""}>{cta}</a></p>
                    </article>""" for n, a, b, d, href, cta in items)
    return f"""                <div class="lp-bar" data-reveal>
                    <span>OPEN BY DEFAULT</span>
                    <span class="lp-count"><a href="https://github.com/Shannon-Dynamics" target="_blank" rel="noopener"
                        style="color:#4C7EFF">github.com/Shannon-Dynamics ↗</a></span>
                </div>
                <h2 class="os-h" data-reveal>REPOSITORIES</h2>
                <div class="os-grid">
{cards(REPOS)}
                </div>
                <h2 class="os-h" data-reveal>DOCUMENTATION</h2>
                <div class="os-grid">
{cards(DOCS)}
                </div>"""


BODIES = {"publications": pubs_body, "books": books_body, "open-source": oss_body}

FILTER_JS = """
        /* ---------- publications filter ---------- */
        (function () {
            const chips = [...document.querySelectorAll("[data-filter]")];
            const rows = [...document.querySelectorAll(".pp-row")];
            const count = document.querySelector("[data-count]");
            if (!chips.length) return;
            chips.forEach(chip => chip.addEventListener("click", () => {
                const want = chip.dataset.filter;
                chips.forEach(c => c.setAttribute("aria-pressed", String(c === chip)));
                let shown = 0;
                rows.forEach(r => {
                    const on = want === "ALL" || r.dataset.type === want;
                    r.hidden = !on;
                    if (on) shown++;
                });
                if (count) count.textContent = shown + (shown === 1 ? " ENTRY" : " ENTRIES");
                ScrollTrigger.refresh();
            }));
        })();
"""


def page(slug):
    meta = PAGES[slug]
    sibs = "\n".join(
        f"""                    <a href="library-{s}.html">
                        <span>{n}</span>
                        <b>{name}<small>{desc}</small></b>
                        <em>↗</em>
                    </a>""" for s, n, name, desc in SIBLINGS[slug])
    title_txt = re.sub("<[^>]+>", "", meta["title"]).replace("&amp;", "&")
    doc_head = head.replace(
        "<title>Shannon Dynamics — Simulation-First Robotics Engineering</title>",
        f"<title>{meta['label'].title()} — Library — Shannon Dynamics</title>")
    doc_head = doc_head.replace("</style>", PAGE_CSS + "    </style>")
    js = js_core + (FILTER_JS if slug == "publications" else "") + js_halftone
    return f"""<!DOCTYPE html>
<html lang="en">
{doc_head}

<body class="bg-slatebg antialiased overflow-x-hidden">
{defs}

{nav}

    <div class="w-full">
        <div class="overflow-hidden">

            <!-- ═══════════ PAGE BANNER ═══════════ -->
            <div class="lp-banner tex tex-bottom">
                <canvas class="halftone absolute inset-0" data-speed="1.6"></canvas>
                <div class="lp-banner-in">
                    <p class="lp-crumb">
                        <a href="index.html">SHANNON</a> /
                        <a href="index.html#library">LIBRARY</a> /
                        <span class="on">{meta['label']}</span>
                    </p>
                    <h1 class="lp-title">{meta['title']}</h1>
                    <p class="lp-sub">{meta['sub']}</p>
                </div>
            </div>

            <!-- ═══════════ CONTENT ═══════════ -->
            <div class="lp-body">
                <div class="lp-wrap">
{BODIES[slug]()}

                    <!-- ── the other two Library pages ── -->
                    <nav class="lp-more" data-reveal aria-label="More from the Library">
{sibs}
                    </nav>
                </div>
            </div>

{cta_footer}
        </div>
    </div>

    <!-- ═══════════ SCRIPTS ═══════════ -->
{scripts_head}

    <script>
{js}    </script>
</body>

</html>
"""


for slug in PAGES:
    out = SRC / f"library-{slug}.html"
    out.write_text(page(slug))
    print("wrote", out.name, len(out.read_text()), "bytes")
