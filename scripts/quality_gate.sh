#!/usr/bin/env bash
# Quality gate of the deliverables (Section 17 of the study protocol). Run from the repository root after
# `make assets`, the final compilation and `scripts/package.sh dist`.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="${1:-$ROOT/dist}"
PY=python3
fail=0
step() { echo; echo "== $*"; }

step "1. unit tests"
(cd "$ROOT" && $PY -m pytest -q tests) || fail=1

step "2. clean compilation of paper-latex.zip"
T="$(mktemp -d)"; (cd "$T" && unzip -q "$DIST/paper-latex.zip" && cd paper-latex && \
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.out 2>&1) || { echo "compile failed"; fail=1; }
LOG="$T/paper-latex/main.log"
if grep -q "^!" "$LOG"; then echo "LaTeX errors"; fail=1; fi
if grep -E -q "undefined|Citation .* undefined|Reference .* undefined" "$LOG"; then grep -E "undefined" "$LOG"; fail=1; fi
python3 - "$LOG" <<'EOF' || fail=1
import re, sys
log = open(sys.argv[1], errors="ignore").read()
bad = [float(x) for x in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log) if float(x) > 1.0]
print(f"overfull boxes above 1 pt: {len(bad)}")
sys.exit(1 if bad else 0)
EOF
pages=$(pdfinfo "$T/paper-latex/main.pdf" | awk '/^Pages/{print $2}')
echo "pages: $pages"; if [ "$pages" -lt 10 ] || [ "$pages" -gt 12 ]; then fail=1; fi
if pdftotext "$T/paper-latex/main.pdf" - | grep -q "??"; then echo "?? in PDF"; fail=1; fi

step "3. lint"
(cd "$ROOT" && $PY scripts/lint_paper.py paper) || fail=1

step "4. references"
(cd "$ROOT" && $PY scripts/check_refs.py paper) || fail=1

step "5. numbers"
(cd "$ROOT" && $PY scripts/check_numbers.py) || fail=1

step "7. header and author block"
pdftotext -l 1 "$T/paper-latex/main.pdf" - | head -12

rm -rf "$T"
echo; if [ $fail -eq 0 ]; then echo "QUALITY GATE: automatic checks passed"; else echo "QUALITY GATE: FAILURES"; fi
exit $fail
