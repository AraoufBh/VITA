"""Compute (and cache) the full-resolution design optimum of every year and carbon price."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vita.data import load_data  # noqa: E402
from vita.evaluate import Evaluator  # noqa: E402
from vita.utils import load_config, set_single_thread_env  # noqa: E402

if __name__ == "__main__":
    set_single_thread_env()
    cfg = load_config("B")
    data, _ = load_data(cfg)
    ev = Evaluator(cfg, data)
    prices = [float(p) for p in sys.argv[1].split(",")] if len(sys.argv) > 1 else cfg["case"]["co2_prices"]
    years = [int(y) for y in sys.argv[2].split(",")] if len(sys.argv) > 2 else cfg["data"]["years"]
    for c in prices:
        for y in years:
            t0 = time.time()
            o = ev.oracle(y, c)
            print(f"oracle {y} {c}: TAC={o['tac']:.0f} co2={o['co2']:.0f} unserved={o['unserved']:.3f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
