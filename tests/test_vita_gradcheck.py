"""Gradient check through the complete aggregation map: encoder -> soft assignment ->
representative days and weights -> aggregated LP value (exact LP-dual backward)."""
import numpy as np
import torch
from conftest import random_features

from vita.data import Standardizer
from vita.vita import AggValue, VITAMap, YearSolver, aggregate_soft


def test_gradcheck_full_map(cfg, par):
    torch.manual_seed(0)
    rng = np.random.default_rng(7)
    nd, K = 20, 3
    X = random_features(rng, nd)
    st = Standardizer([X])
    Z = torch.as_tensor(st(X))
    Xt = torch.as_tensor(X)
    model = VITAMap(4, K, m=4, ch=4)
    with torch.no_grad():
        model.mu.copy_(model.embed(Z)[[0, 7, 13]])
    designs = [np.array([30.0, 20.0, 25.0, 6.0, 12.0, 20.0, 4.0, 18.0, 8.0, 5.0, 150.0]),
               np.array([10.0, 40.0, 5.0, 2.0, 14.0, 4.0, 7.0, 10.0, 12.0, 15.0, 60.0])]
    tau = 0.7
    with torch.no_grad():
        A0, _ = model.soft_assign(Z, tau)
    sigma = A0.argmax(1).numpy()

    def loss():
        A, _ = model.soft_assign(Z, tau)
        P, w = aggregate_soft(A, Xt)
        solver = YearSolver(cfg["case"], par, "B")  # fresh models: cold, deterministic solves
        vals = AggValue.apply(P, w, solver, sigma, [((i, 0), x) for i, x in enumerate(designs)])
        return vals.sum() / 1e5

    model.zero_grad()
    loss().backward()
    params = [("mu", model.mu)] + [(n, p) for n, p in model.encoder.named_parameters()]
    checked, worst = 0, 0.0
    h = 1e-6
    for name, p in params:
        flat = p.data.view(-1)
        g = p.grad.view(-1)
        for i in rng.choice(flat.numel(), size=min(3, flat.numel()), replace=False):
            old = flat[i].item()
            with torch.no_grad():
                flat[i] = old + h
                lp_ = loss().item()
                flat[i] = old - h
                lm_ = loss().item()
                flat[i] = old
                l0 = loss().item()
            fwd, bwd = (lp_ - l0) / h, (l0 - lm_) / h
            if abs(fwd - bwd) > 1e-4 * max(abs(fwd), abs(bwd), 1e-3):
                continue  # kink of the piecewise-linear value function
            fd = (lp_ - lm_) / (2 * h)
            an = g[i].item()
            if max(abs(fd), abs(an)) < 1e-6:
                continue
            rel = abs(fd - an) / max(abs(fd), abs(an))
            worst = max(worst, rel)
            checked += 1
    assert checked >= 10
    assert worst < 1e-3, worst
