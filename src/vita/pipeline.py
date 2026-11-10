"""Experiment orchestration: anchors, baselines, VITA training, deployment and evaluation.

Every task writes its result to ``results/`` (pickle) and is skipped when the file exists, so
interrupted runs resume.  Tasks run in parallel joblib processes, one thread each.
"""
from __future__ import annotations

import functools
import time

import numpy as np
from scipy.stats import qmc

from . import baselines as B
from . import lp_models as L
from .data import Standardizer, load_data
from .evaluate import Evaluator, design_metrics, fossil_reference_co2
from .utils import RESULTS, cache_get, cache_put, load_config, log, set_single_thread_env, stable_hash

BASELINES = ["KM", "KMed", "HC", "HCX", "DP", "AE", "CO", "IX"]
LOG = RESULTS / "logs" / "pipeline.log"

_STATE = {}


def ctx(tier="B"):
    """Per-process cached configuration, data and evaluator."""
    if "cfg" not in _STATE:
        set_single_thread_env()
        cfg = load_config(tier)
        data, info = load_data(cfg)
        _STATE.update(cfg=cfg, data=data, info=info, ev=Evaluator(cfg, data))
    return _STATE["cfg"], _STATE["data"], _STATE["ev"]


def folds(cfg):
    return [dict(name=f"F{f['test']}", train=f["train"], val=f["val"], test=f["test"]) for f in cfg["splits"]["folds"]]


def dev_fold(cfg):
    d = cfg["splits"]["dev"]
    return dict(name=f"F{2018}", train=d["train"], val=d["val"], test=2018)


def standardizer(fold):
    cfg, data, _ = ctx()
    return Standardizer([data[y].X for y in fold["train"]])


def experiment_settings(cfg):
    e = cfg["experiment"]
    return dict(K_grid=e["K_grid"], prices=cfg["case"]["co2_prices"], seeds=e["seeds"], J0=cfg["vita"]["J0"])


# ----------------------------------------------------------------------------- AE encoder
def ae_encoder(fold, seed=0):
    """Encoder of the AE baseline: VITA's encoder trained on reconstruction only (fold training years)."""
    import torch

    from .vita import VITAMap, pretrain_autoencoder

    cfg, data, _ = ctx()
    st = standardizer(fold)
    key = stable_hash(["ae", fold["train"], seed, cfg["vita"]["m"], cfg["vita"]["channels"], cfg["vita"]["pretrain_epochs"]])
    p = RESULTS / "models" / f"ae_{key[:12]}.pt"
    m = VITAMap(len(data[fold["train"][0]].feature_names), 2, cfg["vita"]["m"], cfg["vita"]["channels"])
    if p.exists():
        m.load_state_dict(torch.load(p))
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        pretrain_autoencoder(m, [st(data[y].X) for y in fold["train"]], cfg["vita"]["pretrain_epochs"], seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), p)
    return m.encoder.eval()


# ----------------------------------------------------------------------------- baselines
def baseline_task(method, fold, year, price, K, link="superposition"):
    """Aggregate ``year`` with a baseline, solve the aggregated design LP, evaluate at full resolution."""
    cfg, data, ev = ctx()
    key = stable_hash(["base", method, fold["train"], year, float(price), K, link, cfg["tier"]])
    res = cache_get(key, "baseline")
    if res is not None:
        return res
    par = ev.par(price)
    X = data[year].X
    st = standardizer(fold)
    Z = st(X)
    enc = ae_encoder(fold) if method == "AE" else None
    co = None
    if method == "CO":
        ckey = stable_hash(["co_days", year, float(price), cfg["tier"], par.signature()])
        co = cache_get(ckey, "co")
        if co is None:
            t0 = time.perf_counter()
            co = dict(xs=B.single_day_designs(X, cfg["case"], par, cfg["tier"]), t=time.perf_counter() - t0)
            cache_put(ckey, co, "co")
    agg = B.run_baseline(method, X, Z, st, K, cfg["case"], par, cfg["tier"], enc, None if co is None else co["xs"])
    if method == "CO":
        agg.time += co["t"]  # the 365 single-day LPs are part of the deployment cost
    res = evaluate_aggregation(agg, year, price, link)
    res.update(method=method, fold=fold["name"], year=year, price=float(price), K=K, link=link)
    cache_put(key, res, "baseline")
    return res


def evaluate_aggregation(agg, year, price, link="superposition"):
    from .vita import aggregated_design

    cfg, data, ev = ctx()
    par = ev.par(price)
    des = aggregated_design(agg.P, agg.w, agg.sigma, cfg["case"], par, cfg["tier"], link)
    full = ev.full_eval(year, price, des["x"])
    orc = ev.oracle(year, price)
    co2_ref = fossil_reference_co2(data[year].X, cfg["case"], par, cfg["tier"])
    m = design_metrics(des, full, orc, par, co2_ref)
    m["runtime"] = agg.time + des["lp_time"] + agg.full_time
    m["agg_time"] = agg.time
    m["lp_time"] = des["lp_time"]
    m["full_solves"] = agg.full_solves
    out = dict(metrics=m, x=des["x"], w=agg.w, sigma=agg.sigma, P=agg.P)
    if link == "superposition":
        out["S_h2"], out["S_tes"] = des["S_h2"], des["S_tes"]
    return out


# ----------------------------------------------------------------------------- anchors
def lhs_designs(fold, year, J0, seed):
    cfg, data, ev = ctx()
    price = cfg["case"]["co2_central"]
    xstar = ev.oracle(year, price)["x"]
    xmax = np.max([ev.oracle(y, price)["x"] for y in fold["train"]], axis=0)
    zero = xstar < 0.05
    s = np.where(xmax >= 0.05, 0.2 * xmax, cfg["vita"]["zero_cap_default"])
    u = qmc.LatinHypercube(d=len(xstar), seed=seed).random(J0)
    lo, hi = cfg["vita"]["lhs_low"], cfg["vita"]["lhs_high"]
    X = np.where(zero, u * s, xstar * (lo + (hi - lo) * u))
    par = ev.par(price)
    return np.minimum(X, np.where(np.isfinite(par.x_ub), par.x_ub, np.inf))


def anchor_task(fold, year, K):
    """Anchor set of one training year: LHS designs plus the KM, HC and HC+X designs at this K."""
    cfg, data, ev = ctx()
    price = cfg["case"]["co2_central"]
    J0 = cfg["vita"]["J0"]
    key = stable_hash(["anchors", fold["train"], year, K, J0, cfg["tier"]])
    res = cache_get(key, "anchors")
    if res is not None:
        return res
    designs = [("lhs%d" % j, x) for j, x in enumerate(lhs_designs(fold, year, J0, seed=1000 + year))]
    for mth in cfg["vita"]["anchors_from_baselines"]:
        b = baseline_task(mth, fold, year, price, K)
        designs.append((mth, b["x"]))
    out = []
    for name, x in designs:
        e = ev.full_eval(year, price, x)
        out.append(dict(id=f"{name}", x=np.asarray(x, float), V=float(e["V"]), g=np.asarray(e["g"]), rho=1.0))
    cache_put(key, out, "anchors")
    return out


# ----------------------------------------------------------------------------- VITA
def _full_eval(year, x, price):
    _, _, ev = ctx()
    return ev.full_eval(year, price, x)


def _log_tag(tag, msg):
    log(f"{tag} {msg}", LOG)


def vita_hparams(cfg, overrides=None):
    hp = dict(cfg["vita"])
    if overrides:
        hp.update(overrides)
    return hp


def _vita_hp(cfg, variant, overrides):
    hp = vita_hparams(cfg, overrides)
    if variant == "A1":
        hp["value_loss"] = False
    elif variant == "A2":
        hp["alpha"] = 0.0
    elif variant == "A3":
        hp["on_policy"] = False
    elif variant == "A4":
        hp["encoder"] = "none"
    return hp


def vita_key(fold, K, seed, variant="full", overrides=None):
    cfg, _, _ = ctx()
    hp = _vita_hp(cfg, variant, overrides)
    key = stable_hash(["vita", fold["train"], fold["val"], K, seed, variant, "superposition",
                       {k: v for k, v in sorted(hp.items())}, cfg["tier"]])
    return key, RESULTS / "models" / f"vita_{fold['name']}_K{K}_s{seed}_{variant}_{key[:10]}.pt", hp


def vita_task(fold, K, seed, variant="full", overrides=None):
    """Train VITA (or an ablation variant) on one fold; returns the trained state and history."""
    import torch

    from .vita import train_vita

    cfg, data, ev = ctx()
    key, p, hp = vita_key(fold, K, seed, variant, overrides)
    link = "superposition"
    if p.exists():
        return torch.load(p, weights_only=False)
    price = cfg["case"]["co2_central"]
    st = standardizer(fold)
    years = list(fold["train"]) + [fold["val"]]
    anchors = {y: anchor_task(fold, y, K) for y in fold["train"]} if hp.get("value_loss", True) else \
        {y: [] for y in fold["train"]}
    task = dict(hp=hp, K=K, seed=seed, case=cfg["case"], par=ev.par(price), tier=cfg["tier"],
                X={y: data[y].X for y in years}, Z={y: st(data[y].X) for y in years},
                train_years=list(fold["train"]), val_year=fold["val"], anchors=anchors,
                oracle_val=ev.oracle(fold["val"], price)["tac"],
                full_eval=functools.partial(_full_eval, price=price), link=link,
                std_mean=st.mean, std_std=st.std,
                log=functools.partial(_log_tag, f"[{fold['name']} {variant} {key[:6]}]"))
    res = train_vita(task)
    res.update(fold=fold["name"], K=K, seed=seed, variant=variant, hp=hp)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(res, p)
    return res


def vita_eval_task(fold, K, seed, variant, year, price, link="superposition", snap=False, overrides=None):
    """Deploy a trained VITA map on a target instance (zero full-resolution solves) and evaluate."""
    from .vita import deploy, load_model

    cfg, data, ev = ctx()
    tkey, _, _ = vita_key(fold, K, seed, variant, overrides)
    key = stable_hash(["vitaeval", tkey, year, float(price), link, snap])
    res = cache_get(key, "vitaeval")
    if res is not None:
        return res
    tv = vita_task(fold, K, seed, variant, overrides)
    model = load_model(tv["state"], tv["model_kwargs"])
    st = standardizer(fold)
    X = data[year].X
    agg = deploy(model, X, st(X), K, snap=snap)
    res = evaluate_aggregation(agg, year, price, link)
    res.update(method="VITA" if variant == "full" else f"VITA-{variant}", fold=fold["name"], year=year,
               price=float(price), K=K, seed=seed, variant=variant, link=link, snap=snap,
               train_time=tv["train_time"], best_iter=tv["best_iter"])
    cache_put(key, res, "vitaeval")
    return res


def run_parallel(fn, tasks, n_jobs=2):
    """Run tasks in fresh loky worker processes; workers are shut down afterwards so that idle
    workers do not keep the memory of persistent solver instances."""
    from joblib import Parallel, delayed
    from joblib.externals.loky import get_reusable_executor

    if n_jobs == 1:
        return [fn(*t) for t in tasks]
    out = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(delayed(fn)(*t) for t in tasks)
    get_reusable_executor().shutdown(wait=True)
    return out
