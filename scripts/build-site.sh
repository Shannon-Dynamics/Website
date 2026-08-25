#!/usr/bin/env bash
#
# Assemble the marketing site into dist/.
#
# The site is hand-written HTML with no build step of its own, so this only
# gathers the files that are meant to be public. It exists because the two books
# live in this repository too (book-web/, one Vercel project each) and their
# sources and node_modules must never be published — naming what ships is surer
# than trying to exclude what does not.
#
set -euo pipefail

cd "$(dirname "$0")/.."

rm -rf dist
mkdir -p dist/assets

# The utility layer. This used to be compiled in the visitor's browser by
# cdn.tailwindcss.com on every page load; it is compiled once here instead, and
# only the classes the pages actually use survive.
npx tailwindcss -i styles/tailwind.css -o dist/assets/tailwind.css --minify

# Pages.
cp index.html library-books.html library-open-source.html library-publications.html dist/
# The two book landing pages the interactive editions replaced. Still reachable
# by direct link, so they keep shipping.
cp book-probabilistic-robotics-via-rust.html book-reinforcement-learning-for-robotics.html dist/

# Assets: logos, the demo reels the homepage frames, the covers, the PDF editions.
cp -R src video-demo book-cover book-pdf dist/

# macOS leaves these in any directory it has previewed.
find dist -name '.DS_Store' -delete

echo "dist/ assembled:"
du -sh dist
find dist -type f | wc -l | xargs echo "files:"
