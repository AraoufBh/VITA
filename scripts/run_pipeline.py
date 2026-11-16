"""Run the VITA study stage by stage.  Every result is cached under results/, so stages resume.

usage: python scripts/run_pipeline.py STAGE [--jobs N] [--overrides JSON] [--tag NAME]
stages: data oracle dev-baselines anchors dev train baselines evaluate ablations transfer all
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from vita import pipeline as PL  # noqa: E402
from vita.utils import RESULTS, dump_json, log  # noqa: E402

LOG = RESULTS / "logs" / "pipeline.log"


def stage_data():
    from vita.data import write_checksums

    cfg, data, ev = PL.ctx()
    write_checksums()
    log(f"data tier {cfg['tier']}: years {sorted(data)}; info {PL._STATE['info']['filled_hours']}", LOG)


def stage_oracle(jobs):
    cfg, data, ev = PL.ctx()
    tasks = [(y, c) for c in cfg["case"]["co2_prices"] for y in cfg["data"]["years"]]
    PL.run_parallel(_oracle, tasks, jobs)


def _oracle(y, c):
    _, _, ev = PL.ctx()
    o = ev.oracle(y, c)
    log(f"oracle {y} {c}: TAC={o['tac']:.0f} t={o['time']:.0f}s", LOG)
    return o["tac"]


def stage_dev_baselines(jobs, Ks):
    cfg, _, _ = PL.ctx()
    f = PL.dev_fold(cfg)
    tasks = [(m, f, f["val"], cfg["case"]["co2_central"], K) for K in Ks for m in PL.BASELINES]
    res = PL.run_parallel(PL.baseline_task, tasks, jobs)
    for r in res:
        log(f"dev baseline {r['method']} K={r['K']}: val regret {r['metrics']['regret']:.4f}", LOG)


def stage_anchors(jobs, Ks, folds):
    cfg, _, _ = PL.ctx()
    tasks = []
    for f in folds:
        for y in f["train"]:
            for K in Ks:
                tasks.append((f, y, K))
    # baseline designs of the training years first (used as anchors), then the anchor sets
    btasks = sorted({(m, f["name"], y, K) for (f, y, K) in tasks for m in cfg["vita"]["anchors_from_baselines"]})
    fmap = {f["name"]: f for f in folds}
    PL.run_parallel(PL.baseline_task, [(m, fmap[fn], y, cfg["case"]["co2_central"], K) for (m, fn, y, K) in btasks], jobs)
    PL.run_parallel(PL.anchor_task, tasks, jobs)
    log(f"anchors done for {len(tasks)} (fold, year, K)", LOG)


def _vita_eval_val(f, K, seed, variant, overrides):
    cfg, _, _ = PL.ctx()
    return PL.vita_eval_task(f, K, seed, variant, f["val"], cfg["case"]["co2_central"], overrides=overrides)


def stage_dev(jobs, Ks, seeds, overrides, tag):
    cfg, _, _ = PL.ctx()
    f = PL.dev_fold(cfg)
    tasks = [(f, K, s, "full", overrides) for K in sorted(Ks, reverse=True) for s in seeds]
    PL.run_parallel(PL.vita_task, tasks, jobs)
    res = PL.run_parallel(_vita_eval_val, tasks, jobs)
    out = {}
    for K in Ks:
        regs = [r["metrics"]["regret"] for r in res if r["K"] == K]
        base = {m: PL.baseline_task(m, f, f["val"], cfg["case"]["co2_central"], K)["metrics"]["regret"]
                for m in ["KM", "KMed", "HC", "HCX", "DP", "AE", "CO"]}
        best = min(base.values())
        out[K] = dict(vita_mean=float(np.mean(regs)), vita=regs, best_baseline=best,
                      best_name=min(base, key=base.get), baselines=base, ratio=float(np.mean(regs) / best))
        log(f"DEV[{tag}] K={K}: VITA {np.mean(regs):.4f} {np.round(regs, 4)} | best {out[K]['best_name']} "
            f"{best:.4f} | ratio {out[K]['ratio']:.3f}", LOG)
    dump_json(dict(tag=tag, overrides=overrides, result=out), RESULTS / "dev" / f"dev_{tag}.json")


def stage_train(jobs, Ks, seeds, folds, overrides, variants=("full",)):
    tasks = [(f, K, s, v, overrides) for K in sorted(Ks, reverse=True) for v in variants for f in folds for s in seeds]
    PL.run_parallel(PL.vita_task, tasks, jobs)
    log(f"trained {len(tasks)} VITA models ({variants})", LOG)


def stage_baselines(jobs, Ks, folds, prices):
    tasks = [(m, f, f["test"], c, K) for K in Ks for c in prices for f in folds for m in PL.BASELINES]
    tasks.sort(key=lambda t: t[0] != "IX")  # IX (three full solves) first for load balance
    PL.run_parallel(PL.baseline_task, tasks, jobs)
    log(f"baselines evaluated: {len(tasks)}", LOG)


def stage_evaluate(jobs, Ks, seeds, folds, prices, overrides, variants=("full",)):
    tasks = [(f, K, s, v, f["test"], c, "superposition", False, overrides)
             for K in Ks for v in variants for c in prices for f in folds for s in seeds]
    PL.run_parallel(PL.vita_eval_task, tasks, jobs)
    log(f"VITA evaluations: {len(tasks)}", LOG)


def _abl_eval(f, s, var, kw):
    v = "full" if var in ("A5", "A6") else var
    return PL.vita_eval_task(f, 8, s, v, f["test"], 150.0, **kw)


def stage_ablations(jobs, folds, seeds, base):
    tasks = [(f, 8, s, v, None) for v in ("A3", "A4", "A2", "A1") for f in folds for s in seeds]
    PL.run_parallel(PL.vita_task, tasks, jobs)
    ev = [(f, s, var, kw) for f in folds for s in seeds for var, kw in
          (("A1", {}), ("A2", {}), ("A3", {}), ("A4", {}), ("A5", {"link": "cyclic"}), ("A6", {"snap": True}))]
    PL.run_parallel(_abl_eval, ev, jobs)
    bt = [(base, f, f["test"], 150.0, 8, "cyclic") for f in folds] + [(base, f, f["test"], 150.0, 8) for f in folds]
    PL.run_parallel(PL.baseline_task, bt, jobs)
    log(f"ablations done ({len(tasks)} trainings)", LOG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--K", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--overrides", default="{}")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--variants", default="full")
    ap.add_argument("--folds", default=None)
    ap.add_argument("--prices", default=None)
    a = ap.parse_args()
    cfg, _, _ = PL.ctx()
    st = PL.experiment_settings(cfg)
    Ks = [int(k) for k in a.K.split(",")] if a.K else st["K_grid"]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else st["seeds"]
    ov = json.loads(a.overrides) or None
    folds = PL.folds(cfg)
    if a.folds:
        folds = [f for f in folds if f["name"] in a.folds.split(",")]
    t0 = time.time()
    if a.stage == "data":
        stage_data()
    elif a.stage == "oracle":
        stage_oracle(a.jobs)
    elif a.stage == "dev-baselines":
        stage_dev_baselines(a.jobs, Ks)
    elif a.stage == "anchors":
        stage_anchors(a.jobs, Ks, folds)
    elif a.stage == "dev":
        stage_dev(a.jobs, Ks, seeds, ov, a.tag)
    elif a.stage == "train":
        stage_train(a.jobs, Ks, seeds, folds, ov, tuple(a.variants.split(",")))
    elif a.stage == "baselines":
        stage_baselines(a.jobs, Ks, folds, [float(c) for c in a.prices.split(",")] if a.prices else st["prices"])
    elif a.stage == "ablations":
        stage_ablations(a.jobs, folds, [int(x) for x in (a.seeds or ",".join(map(str, cfg["experiment"].get("ablation_seeds", seeds)))).split(",")],
                        cfg["experiment"].get("ablation_baseline", "KMed"))
    elif a.stage == "evaluate":
        stage_evaluate(a.jobs, Ks, seeds, folds, st["prices"], ov, tuple(a.variants.split(",")))
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    log(f"stage {a.stage} finished in {time.time() - t0:.0f}s", LOG)
