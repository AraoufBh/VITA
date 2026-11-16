"""Time-series aggregation baselines (same standardized features, same aggregated model).

Every method returns an :class:`Aggregation` (representative days in raw units, weights,
day-to-cluster sequence).  tsam normalizes each column to [0, 1] internally; column weights
equal to each feature's range in standardized units undo that normalization, so every method
measures distances on the same z-scored features.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import lp_models as L

warnings.filterwarnings("ignore", module="tsam")


@dataclass
class Aggregation:
    P: np.ndarray            # (K, 24, F) raw representative profiles
    w: np.ndarray            # (K,) weights, sum 365
    sigma: np.ndarray        # (365,) cluster of each day
    time: float = 0.0        # aggregation runtime [s] (deployment)
    full_solves: int = 0     # full-resolution solves needed at deployment
    full_time: float = 0.0   # time spent in those solves [s]
    extra: dict = field(default_factory=dict)

    @property
    def K(self):
        return len(self.w)


def means_from_labels(X, labels, K):
    P = np.stack([X[labels == k].mean(0) for k in range(K)])
    w = np.bincount(labels, minlength=K).astype(float)
    return P, w


def medoids_from_labels(X, Z, labels, K):
    """Medoid of each cluster in standardized input space (minimum sum of squared distances)."""
    Zf = Z.reshape(len(Z), -1)
    meds = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        D = ((Zf[idx, None, :] - Zf[None, idx, :]) ** 2).sum(-1)
        meds.append(idx[np.argmin(D.sum(1))])
    meds = np.array(meds)
    return X[meds], np.bincount(labels, minlength=K).astype(float), meds


def _tsam(Z, K, method, representation, seed=0):
    import tsam
    from tsam import ClusterConfig, KMeans

    nd, H, F = Z.shape
    cols = [f"f{i}" for i in range(F)]
    df = pd.DataFrame(Z.reshape(-1, F), index=pd.date_range("2001-01-01", periods=nd * H, freq="h"), columns=cols)
    flat = Z.reshape(-1, F)
    rng_ = flat.max(0) - flat.min(0)
    weights = {c: float(max(r, 1e-9)) for c, r in zip(cols, rng_)}
    if method == "kmeans":
        meth = KMeans(n_init=10, random_state=seed)
    else:
        meth = method
    res = tsam.aggregate(df, n_clusters=K, period_duration=24, cluster=ClusterConfig(method=meth, representation=representation),
                         weights=weights, preserve_column_means=False)
    rep = res.cluster_representatives.values.reshape(K, H, F)
    labels = np.asarray(res.cluster_assignments, int)
    counts = np.bincount(labels, minlength=K).astype(float)
    return rep, labels, counts


def agg_km(X, Z, st, K, seed=0):
    t0 = time.perf_counter()
    rep, labels, w = _tsam(Z, K, "kmeans", "mean", seed)
    return Aggregation(st.inverse(rep), w, labels, time.perf_counter() - t0)


def agg_hc(X, Z, st, K):
    t0 = time.perf_counter()
    rep, labels, w = _tsam(Z, K, "hierarchical", "medoid")
    return Aggregation(st.inverse(rep), w, labels, time.perf_counter() - t0)


def agg_dp(X, Z, st, K):
    t0 = time.perf_counter()
    rep, labels, w = _tsam(Z, K, "hierarchical", "distribution")
    return Aggregation(st.inverse(rep), w, labels, time.perf_counter() - t0)


def agg_kmed(X, Z, st, K, seed=0):
    import kmedoids

    t0 = time.perf_counter()
    Zf = Z.reshape(len(Z), -1)
    D = np.sqrt(((Zf[:, None, :] - Zf[None, :, :]) ** 2).sum(-1))
    res = kmedoids.fasterpam(D, K, random_state=seed, n_cpu=1)
    labels = np.asarray(res.labels, int)
    meds = np.asarray(res.medoids, int)
    P = X[meds]
    w = np.bincount(labels, minlength=K).astype(float)
    return Aggregation(P, w, labels, time.perf_counter() - t0)


def extreme_days(X, tier="B"):
    """Indices of the three extreme days: max LT heat (Tier A) or min PV (Tier B substitute),
    min PV plus wind, max price (daily means), distinct days chosen in this order."""
    pv, wd, pr = X[..., 0].mean(1), X[..., 1].mean(1), X[..., 2].mean(1)
    crit = []
    if tier == "A":
        crit.append(np.argsort(-X[..., 3].mean(1)))
    else:
        crit.append(np.argsort(pv))
    crit.append(np.argsort(pv + wd))
    crit.append(np.argsort(-pr))
    chosen = []
    for order in crit:
        for d in order:
            if d not in chosen:
                chosen.append(int(d))
                break
    return chosen


def agg_hcx(X, Z, st, K, tier="B"):
    """HC with K-3 clusters plus 3 extreme days added as new cluster centers; days closer to an
    extreme day than to their own medoid join it (tsam 'new_cluster' semantics)."""
    t0 = time.perf_counter()
    rep, labels, _ = _tsam(Z, K - 3, "hierarchical", "medoid")
    ext = extreme_days(X, tier)
    Zf = Z.reshape(len(Z), -1)
    centers = list(rep.reshape(K - 3, -1)) + [Zf[d] for d in ext]
    new = labels.copy()
    for d in range(len(Z)):
        if d in ext:
            new[d] = K - 3 + ext.index(d)
            continue
        best = ((Zf[d] - centers[labels[d]]) ** 2).sum()
        for j, e in enumerate(ext):
            de = ((Zf[d] - Zf[e]) ** 2).sum()
            if de < best:
                best, new[d] = de, K - 3 + j
    P = np.concatenate([st.inverse(rep), X[ext]], axis=0)
    w = np.bincount(new, minlength=K).astype(float)
    return Aggregation(P, w, new, time.perf_counter() - t0, extra={"extreme_days": ext})


def single_day_designs(X, case, par, tier="B"):
    """CO baseline: design LP of every single day repeated over the year (cyclic storage)."""
    xs = np.zeros((len(X), L.NX))
    for d in range(len(X)):
        inp = L.inputs_from_features(X[d:d + 1], case, par, tier)
        lp = L.HubLP(inp, [365.0], par, mode="design", link="cyclic")
        xs[d] = lp.summary(lp.solve())["x"]
    return xs


def agg_co(X, Z, st, K, case, par, tier="B", xs=None):
    from sklearn.cluster import AgglomerativeClustering

    t0 = time.perf_counter()
    if xs is None:
        xs = single_day_designs(X, case, par, tier)
    t_lp = time.perf_counter() - t0
    mx = np.abs(xs).max(0)
    keep = mx > 1e-6
    Y = xs[:, keep] / mx[keep]
    labels = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Y)
    P, w, meds = medoids_from_labels(X, Z, labels, K)
    return Aggregation(P, w, labels, time.perf_counter() - t0, extra={"single_day_lp_time": t_lp, "medoids": meds})


def agg_ae(X, Z, st, K, encoder, seed=0):
    """Autoencoder embedding (encoder trained on reconstruction only) plus k-means, cluster means."""
    import torch
    from sklearn.cluster import KMeans

    t0 = time.perf_counter()
    with torch.no_grad():
        E = encoder(torch.tensor(Z, dtype=torch.float64).permute(0, 2, 1)).numpy()
    labels = KMeans(K, n_init=10, random_state=seed).fit_predict(E)
    P, w = means_from_labels(X, labels, K)
    return Aggregation(P, w, labels, time.perf_counter() - t0)


def agg_ix(X, Z, st, K, case, par, tier="B", n_iter=3):
    """Iterative extreme periods: HC with K-3 clusters, then 3 rounds of design, full-resolution
    evaluation and inclusion of the day with the largest unserved energy (else largest daily
    cost gap between the full-resolution model and its representative day)."""
    t0 = time.perf_counter()
    base = agg_hc(X, Z, st, K - n_iter)
    P, w, sigma = base.P.copy(), base.w.copy(), base.sigma.copy()
    full_t, nfull = 0.0, 0
    added = []
    for it in range(n_iter):
        inp = L.inputs_from_features(P, case, par, tier)
        lpd = L.HubLP(inp, w, par, mode="design", link="superposition", sigma=sigma)
        sd = lpd.solve()
        xh = lpd.summary(sd)["x"]
        per_day = lpd.period_costs(sd)  # unweighted per-day cost of each representative day
        t1 = time.perf_counter()
        ev = L.solve_full_oper(X, xh, case, par, tier, method="simplex", daily=True)
        full_t += time.perf_counter() - t1
        nfull += 1
        un = ev["daily_unserved"].copy()
        un[added] = -1
        if un.max() > 1e-6:
            d = int(np.argmax(un))
        else:
            gap = np.abs(ev["daily_cost"] - per_day[sigma])
            gap[added] = -1
            d = int(np.argmax(gap))
        added.append(d)
        k_old = sigma[d]
        w[k_old] -= 1.0
        P = np.concatenate([P, X[d:d + 1]], axis=0)
        w = np.append(w, 1.0)
        sigma = sigma.copy()
        sigma[d] = len(w) - 1
    return Aggregation(P, w, sigma, time.perf_counter() - t0, full_solves=nfull, full_time=full_t,
                       extra={"added_days": added})


def run_baseline(name, X, Z, st, K, case, par, tier="B", encoder=None, co_designs=None, seed=0):
    if name == "KM":
        return agg_km(X, Z, st, K, seed)
    if name == "KMed":
        return agg_kmed(X, Z, st, K, seed)
    if name == "HC":
        return agg_hc(X, Z, st, K)
    if name == "HCX":
        return agg_hcx(X, Z, st, K, tier)
    if name == "DP":
        return agg_dp(X, Z, st, K)
    if name == "AE":
        return agg_ae(X, Z, st, K, encoder, seed)
    if name == "CO":
        return agg_co(X, Z, st, K, case, par, tier, co_designs)
    if name == "IX":
        return agg_ix(X, Z, st, K, case, par, tier)
    raise ValueError(name)
