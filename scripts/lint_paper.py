"""Style lint of the paper: em-dashes, forbidden patterns and banned vocabulary (Section 13.4).

usage: python scripts/lint_paper.py [paper_dir]   (exit code 1 on any violation)
"""
import re
import subprocess
import sys
from pathlib import Path

BANNED = ["delve", "tapestry", "realm", "landscape", "pivotal", "crucial", "crucially", "paramount", "seamless",
          "seamlessly", "leverage", "leverages", "leveraging", "harness", "unlock", "unleash", "empower", "elevate",
          "foster", "bolster", "underscore", "underscores", "showcase", "showcases", "intricate", "multifaceted",
          "nuanced", "meticulous", "meticulously", "groundbreaking", "cutting-edge", "game-changer", "revolutionize",
          "transformative", "holistic", "myriad", "plethora", "notably", "importantly", "interestingly", "remarkably",
          "vital", "utilize", "utilizes", "journey", "embark", "navigate"]
PHRASES = ["it is worth noting", "it is important to note", "plays a crucial role", "paves the way", "shed light",
           "a testament to", "ever-evolving", "in today's", "in conclusion", "in summary"]
CAPS = {"furthermore": 2, "moreover": 2, "additionally": 2, "novel": 2, "robust": 2, "comprehensive": 1,
        "state-of-the-art": 2}
FORBIDDEN = ["TODO", "TBD", "FIXME", "XX", "lorem", "placeholder", "??", "Journal of LaTeX Class Files", "github"]


def body_text(pdf: Path) -> tuple[str, str]:
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
    # text before the References heading
    idx = txt.rfind("REFERENCES")
    if idx < 0:
        idx = txt.rfind("References")
    body = txt[:idx] if idx > 0 else txt
    return txt, re.sub(r"-\n\s*", "", body)  # join hyphenated line breaks


def main(paper_dir="paper"):
    pdir = Path(paper_dir)
    errors = []
    files = [p for p in pdir.rglob("*") if p.suffix in (".tex", ".bib") and "build" not in p.parts]
    for p in files:
        s = p.read_text(encoding="utf-8")
        if "—" in s:
            errors.append(f"{p}: U+2014 em-dash")
        if "---" in s:
            errors.append(f"{p}: triple hyphen")
        for pat in FORBIDDEN:
            hay = s if pat not in ("github", "lorem", "placeholder") else s.lower()
            if pat in hay:
                errors.append(f"{p}: forbidden pattern '{pat}'")
    pdf = pdir / "main.pdf"
    if pdf.exists():
        full, body = body_text(pdf)
        if "—" in full:
            errors.append("main.pdf: U+2014 em-dash in text")
        for pat in FORBIDDEN:
            hay = full if pat not in ("github", "lorem", "placeholder") else full.lower()
            if pat in hay:
                errors.append(f"main.pdf: forbidden pattern '{pat}'")
        low = body.lower()
        for w in BANNED:
            n = len(re.findall(rf"(?<![a-z-]){re.escape(w)}(?![a-z])", low))
            if n:
                errors.append(f"banned word '{w}' x{n}")
        for ph in PHRASES:
            if ph in low:
                errors.append(f"banned phrase '{ph}'")
        for w, cap in CAPS.items():
            n = len(re.findall(rf"(?<![a-z-]){re.escape(w)}(?![a-z])", low))
            if n > cap:
                errors.append(f"'{w}' x{n} exceeds cap {cap}")
        n = low.count("to the best of our knowledge")
        if n > 1:
            errors.append("'to the best of our knowledge' more than once")
        for m in re.finditer(r"significant(ly)?", low):
            ctx = low[max(0, m.start() - 250): m.end() + 250]
            if not any(k in ctx for k in ("wilcoxon", "holm", "p =", "p<", "p <", "p-value", "test")):
                errors.append(f"'significant' without a nearby statistical test: ...{low[m.start()-60:m.end()+40]}...")
        if "!" in re.sub(r"\s", "", body.split("Abstract")[-1])[:0]:
            pass
    else:
        errors.append("main.pdf missing")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"lint ok: {len(files)} source files and main.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
