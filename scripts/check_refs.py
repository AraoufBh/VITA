"""Reference checks: at most 30 entries in main.bbl, each printed with its DOI, each cited, DOIs
syntactically valid; if doi.org or Crossref is reachable, resolve every DOI and compare titles.

usage: python scripts/check_refs.py [paper_dir]
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def main(paper_dir="paper"):
    pdir = Path(paper_dir)
    bbl = (pdir / "main.bbl").read_text(encoding="utf-8")
    bib = (pdir / "references.bib").read_text(encoding="utf-8")
    tex = "".join(p.read_text(encoding="utf-8") for p in pdir.rglob("*.tex") if "generated" not in p.parts)
    items = re.findall(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})", bbl, flags=re.S)
    items = [(k, v) for k, v in items if not k.startswith("IEEEexample")]
    errors = []
    if len(items) > 30:
        errors.append(f"{len(items)} references (> 30)")
    cited = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", tex):
        cited.update(c.strip() for c in m.group(1).split(","))
    entries = {}
    for e in re.split(r"(?=^@)", bib, flags=re.M):
        m = re.match(r"@(\w+)\{([^,]+),", e)
        if m and m.group(1).lower() != "ieeetranbstctl":
            entries[m.group(2).strip()] = e
    for key in entries:
        if key not in cited:
            errors.append(f"uncited entry {key}")
    titles = {}
    for key, body in items:
        dm = re.search(r"doi:\s*\\href\{https://doi.org/([^}]+)\}", body)
        if not dm:
            errors.append(f"{key}: no printed DOI")
            continue
        doi = dm.group(1)
        if not DOI_RE.match(doi):
            errors.append(f"{key}: invalid DOI {doi}")
        tm = re.search(r"title=\{(.*?)\}, (journal|howpublished)", entries.get(key, ""), flags=re.S)
        titles[key] = (doi, tm.group(1) if tm else "")
    reachable = False
    try:
        urllib.request.urlopen("https://api.crossref.org/works?rows=0", timeout=10)
        reachable = True
    except Exception:  # noqa: BLE001
        print("Crossref and doi.org unreachable from this environment: online title check skipped")
    if reachable:
        for key, (doi, title) in titles.items():
            try:
                with urllib.request.urlopen(f"https://api.crossref.org/works/{doi}", timeout=20) as r:
                    t = json.load(r)["message"]["title"][0]
                if norm(t)[:40] != norm(re.sub(r"[{}\\]", "", title))[:40]:
                    errors.append(f"{key}: title mismatch ({t})")
            except Exception as ex:  # noqa: BLE001
                errors.append(f"{key}: DOI {doi} not resolved ({type(ex).__name__})")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"refs ok: {len(items)} references, all printed with a syntactically valid DOI and cited")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
