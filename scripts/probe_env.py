"""Probe the execution environment (Python, CPU, RAM, disk, TeX, host reachability)."""
import os
import platform
import shutil
import subprocess
import urllib.request

HOSTS = ["data.open-power-system-data.org", "raw.githubusercontent.com", "pypi.org", "doi.org", "api.crossref.org",
         "download.pytorch.org"]


def reach(host):
    try:
        with urllib.request.urlopen(f"https://{host}/", timeout=15) as r:
            return f"reachable (HTTP {r.status})"
    except urllib.error.HTTPError as e:
        return f"reachable (HTTP {e.code})"
    except Exception as e:  # noqa: BLE001
        return f"unreachable ({type(e).__name__})"


def main():
    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"cpu count {os.cpu_count()}")
    try:
        mem = int(open("/proc/meminfo").read().split("\n")[0].split()[1]) / 1024 ** 2
        print(f"ram {mem:.1f} GiB")
    except OSError:
        pass
    du = shutil.disk_usage(".")
    print(f"disk free {du.free / 1024 ** 3:.0f} GiB")
    for tool in ["pdflatex", "bibtex", "latexmk", "pdftotext", "pdftoppm", "kpsewhich"]:
        print(f"{tool}: {shutil.which(tool)}")
    if shutil.which("kpsewhich"):
        out = subprocess.run(["kpsewhich", "IEEEtran.cls"], capture_output=True, text=True).stdout.strip()
        print(f"IEEEtran.cls: {out or 'missing'}")
    for h in HOSTS:
        print(f"{h}: {reach(h)}")


if __name__ == "__main__":
    main()
