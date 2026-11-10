"""Aggregation maps: weights, representatives, sequences and fixed K."""
import numpy as np
import torch
from conftest import random_features

from vita import baselines as B
from vita.data import Standardizer
from vita.vita import VITAMap, aggregate_hard, aggregate_soft, deploy


def year(rng, nd=365):
    X = random_features(rng, nd)
    return X, Standardizer([X])


def test_soft_aggregation_weights_and_means(rng):
    X, st = year(rng, 40)
    m = VITAMap(4, 5, m=8, ch=8)
    A, _ = m.soft_assign(torch.as_tensor(st(X)), tau=0.5)
    P, w = aggregate_soft(A, torch.as_tensor(X))
    assert abs(w.sum().item() - 40) < 1e-9
    ref = (A.detach().numpy().T @ X.reshape(40, -1)) / w.detach().numpy()[:, None]
    assert np.allclose(P.detach().numpy().reshape(5, -1), ref)


def test_hard_assignment_keeps_K(rng):
    X, st = year(rng, 50)
    m = VITAMap(4, 12, m=8, ch=8)
    with torch.no_grad():
        m.mu.zero_()
        m.mu[0] = 100.0  # most prototypes collapse: empty clusters must be refilled
    agg = deploy(m, X, st(X), 12)
    assert len(np.unique(agg.sigma)) == 12
    assert agg.w.sum() == 50 and np.all(agg.w >= 1)
    P, w = aggregate_hard(agg.sigma, X, 12)
    assert np.allclose(P, agg.P)


def test_baselines_consistent(cfg, par, rng):
    X, st = year(rng)
    Z = st(X)
    for name in ["KM", "KMed", "HC", "HCX", "DP"]:
        agg = B.run_baseline(name, X, Z, st, 8, cfg["case"], par)
        assert agg.P.shape == (8, 24, 4)
        assert abs(agg.w.sum() - 365) < 1e-9
        assert np.array_equal(np.bincount(agg.sigma, minlength=8).astype(float), agg.w)
    hcx = B.agg_hcx(X, Z, st, 8)
    for j, d in enumerate(hcx.extra["extreme_days"]):
        assert np.allclose(hcx.P[5 + j], X[d])
    km = B.agg_km(X, Z, st, 8)
    for k in range(8):
        assert np.allclose(km.P[k], X[km.sigma == k].mean(0), atol=1e-8)
