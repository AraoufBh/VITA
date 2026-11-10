"""Add `note = {doi: \\href{https://doi.org/<doi>}{<doi>}}` to every bibliography entry with a doi field.

IEEEtran.bst (v1.14) does not print the doi field; the note makes every reference end with its DOI.
usage: python scripts/add_doi_notes.py paper/references.bib
"""
import re
import sys
from pathlib import Path


def main(path):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    out = []
    for entry in re.split(r"(?=^@)", s, flags=re.M):
        m = re.search(r"doi\s*=\s*\{([^}]*)\}", entry)
        if entry.startswith("@") and m and "note" not in entry:
            doi = m.group(1).strip()
            note = "note={doi: \\href{https://doi.org/%s}{%s}}" % (doi, doi.replace("_", "\\_"))
            entry = entry.rstrip()
            assert entry.endswith("}"), entry[:60]
            entry = entry[:-1].rstrip() + ", " + note + "}\n"
        out.append(entry)
    p.write_text("".join(out), encoding="utf-8")
    print(f"doi notes added in {p}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "paper/references.bib")
