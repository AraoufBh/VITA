#!/usr/bin/env bash
# Build the deliverables: paper-latex.zip (Overleaf-ready LaTeX project) and vita-tsa-code.zip (repository).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/dist}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
# LaTeX project: every input file, flat paths as referenced by main.tex
TMP="$(mktemp -d)"
mkdir -p "$TMP/paper-latex"
cd "$ROOT/paper"
cp main.tex figstyle.tex references.bib "$TMP/paper-latex/"
cp -r figures generated data "$TMP/paper-latex/"

cd "$TMP"
rm -f "$OUT/paper-latex.zip"
zip -qr "$OUT/paper-latex.zip" paper-latex
# code repository (raw data are downloaded by `make data`; caches of LP results are included)
cd "$ROOT/.."
rm -f "$OUT/vita-tsa-code.zip"
zip -qr "$OUT/vita-tsa-code.zip" vita-tsa \
  -x "vita-tsa/data/raw/*.csv" "vita-tsa/dist/*" "vita-tsa/logs/*" "vita-tsa/paper/*.aux" "vita-tsa/paper/*.log" \
     "vita-tsa/paper/*.fls" "vita-tsa/paper/*.fdb_latexmk" "vita-tsa/paper/*.blg" "vita-tsa/paper/*.out" \
     "vita-tsa/**/__pycache__/*" "vita-tsa/.pytest_cache/*" "vita-tsa/paper/build/*"
rm -rf "$TMP"
ls -l "$OUT"
