"""Summarize development-fold VITA runs from saved models (best validation regret per seed)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vita.utils import RESULTS  # noqa: E402

rows = {}
for p in sorted((RESULTS / "models").glob("vita_F2018_K*_s*_full_*.pt")):
    r = torch.load(p, weights_only=False)
    hp = r["hp"]
    sig = tuple(sorted((k, str(v)) for k, v in hp.items()))
    rows.setdefault((r["K"], sig), []).append((r["seed"], r["best_val_regret"], r["best_iter"],
                                               [round(h["val_regret"], 3) for h in r["history"]], r["train_time"]))
base = None
for (K, sig), runs in rows.items():
    d = dict(sig)
    key = {k: d.get(k) for k in ("loss_norm", "proto_init", "st_steps", "lambda_rec", "alpha", "lr_mu", "lr_theta", "anchors_from_baselines", "m", "encoder", "rho_onpolicy", "R", "tau_end")}
    regs = [r[1] for r in sorted(runs)]
    print(f"K={K} n={len(runs)} mean={sum(regs)/len(regs):.4f} regs={[round(x, 3) for x in regs]} iters={[r[2] for r in sorted(runs)]} t={[int(r[4]) for r in sorted(runs)]}")
    print("    ", key)
    for r in sorted(runs):
        print("      seed", r[0], r[3])
