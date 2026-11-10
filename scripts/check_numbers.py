"""Recompute every macro of paper/generated/numbers.tex from results/ and fail on any mismatch.

The assets are regenerated into a temporary directory, so the paper files are not touched.
usage: python scripts/check_numbers.py
"""
import importlib.util
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vita import export  # noqa: E402


def parse(path):
    txt = Path(path).read_text()
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", txt, flags=re.M))


def main():
    paper = parse(ROOT / "paper" / "generated" / "numbers.tex")
    with tempfile.TemporaryDirectory() as tmp:
        export.GEN = Path(tmp) / "generated"
        export.DAT = Path(tmp) / "data"
        spec = importlib.util.spec_from_file_location("make_assets", ROOT / "scripts" / "make_assets.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.GEN = export.GEN
        mod.main()
        fresh = parse(export.GEN / "numbers.tex")
    bad = [k for k in set(paper) | set(fresh) if paper.get(k) != fresh.get(k)]
    for k in sorted(bad):
        print(f"mismatch {k}: paper={paper.get(k)} recomputed={fresh.get(k)}")
    if bad:
        return 1
    print(f"numbers ok: {len(paper)} macros equal to the values recomputed from results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
