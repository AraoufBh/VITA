"""Central finite-difference checks of every LP sensitivity formula (Section 6.5).

A coordinate is checked only at nondegenerate points, detected as agreement of the forward
and backward one-sided differences; at least half of the sampled coordinates of every
formula must be checkable and every checked coordinate must match to 1e-4 (relative).
"""
import numpy as np
import pytest
from conftest import random_features

from vita import lp_models as L
from vita.costs import DESIGN_NAMES

RTOL = 1e-4


def make_lp(cfg, par, P, w, x, link, sigma, override=None):
    inp = L.inputs_from_features(P, cfg["case"], par, "B")
    if override:
        for k, (idx, val) in override.items():
            inp[k] = inp[k].copy()
            inp[k][idx] = val
    return L.HubLP(inp, w, par, mode="oper", x=x, link=link, sigma=sigma)


def fd_check(fun, theta0, analytic, h):
    vp, vm, v0 = fun(theta0 + h), fun(theta0 - h), fun(theta0)
    fwd, bwd = (vp - v0) / h, (v0 - vm) / h
    scale = max(abs(fwd), abs(bwd), 1.0)
    if abs(fwd - bwd) > 1e-7 * max(abs(v0), 1.0) / h + 1e-6 * scale:
        return None  # kink: degenerate point in this direction
    fd = (vp - vm) / (2 * h)
    return abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-3)


def instance(cfg, par, rng, link):
    nd, K = 6, 3
    Xd = random_features(rng, nd)
    sigma = np.array([0, 1, 2, 0, 1, 2]) if link == "superposition" else None
    if link == "superposition":
        P = np.stack([Xd[sigma == k].mean(0) for k in range(K)])
        w = np.bincount(sigma, minlength=K).astype(float) * 365 / nd
    else:
        P, w = Xd[:K], np.full(K, 365.0 / K)
    x = np.array([20.0, 15.0, 30.0, 6.0, 9.0, 40.0, 4.0, 18.0, 6.0, 5.0, 120.0])
    return P, w, x, sigma


@pytest.mark.parametrize("link", ["superposition", "cyclic"])
def test_input_sensitivities(cfg, par, rng, link):
    P, w, x, sigma = instance(cfg, par, rng, link)
    lp = make_lp(cfg, par, P, w, x, link, sigma)
    sol = lp.solve()
    g = lp.input_gradients(sol)
    checked, total = 0, 0
    for key in ("d_el", "d_L", "d_H", "cf_pv", "cf_wd", "price", "cop_L"):
        base = L.inputs_from_features(P, cfg["case"], par, "B")[key]
        idxs = [(int(rng.integers(0, P.shape[0])), int(rng.integers(0, 24))) for _ in range(6)]
        for idx in idxs:
            t0 = base[idx]
            if key.startswith("cf") and (t0 < 1e-3 or t0 > 0.999):
                continue
            h = 1e-4 * max(abs(t0), 1.0)

            def V(t, key=key, idx=idx):
                return make_lp(cfg, par, P, w, x, link, sigma, {key: (idx, t)}).solve()["obj"]

            err = fd_check(V, t0, g[key][idx], h)
            total += 1
            if err is not None:
                checked += 1
                assert err < RTOL, (key, idx, err, g[key][idx])
    for k in range(P.shape[0]):
        def Vw(t, k=k):
            w2 = w.copy()
            w2[k] = t
            return make_lp(cfg, par, P, w2, x, link, sigma).solve()["obj"]

        err = fd_check(Vw, w[k], g["w"][k], 1e-4 * w[k])
        total += 1
        if err is not None:
            checked += 1
            assert err < RTOL, ("w", k, err)
    assert checked >= total // 2


@pytest.mark.parametrize("link", ["superposition", "cyclic"])
def test_capacity_subgradient(cfg, par, rng, link):
    P, w, x, sigma = instance(cfg, par, rng, link)
    lp = make_lp(cfg, par, P, w, x, link, sigma)
    sol = lp.solve()
    g = lp.capacity_subgradient(sol)
    checked = 0
    for i, name in enumerate(DESIGN_NAMES):
        def V(t, i=i):
            x2 = x.copy()
            x2[i] = t
            return make_lp(cfg, par, P, w, x2, link, sigma).solve()["obj"]

        err = fd_check(V, x[i], g[i], 1e-4 * max(x[i], 1.0))
        if err is not None:
            checked += 1
            assert err < RTOL, (name, err, g[i])
    assert checked >= len(DESIGN_NAMES) // 2


def test_generic_lp_duality(cfg, par, rng):
    """dV/db = lambda, dV/dc_j = y_j and dV/dA_ij = -lambda_i y_j on the raw LP data."""
    P, w, x, sigma = instance(cfg, par, rng, "superposition")
    lp = make_lp(cfg, par, P, w, x, "superposition", sigma)
    sol = lp.solve()
    y, lam = sol["y"], sol["rd"]

    def resolve(mod):
        lp2 = make_lp(cfg, par, P, w, x, "superposition", sigma)
        mod(lp2)
        return lp2.solve()["obj"]

    # right-hand side of equality rows
    rows = lp.rows("el")[:4]
    checked = 0
    for r in rows:
        b0 = lp.rlo[r]

        def Vb(t, r=r):
            def mod(l2):
                l2.rlo[r] = t
                l2.rhi[r] = t
            return resolve(mod)

        err = fd_check(Vb, b0, lam[r], 1e-4 * max(abs(b0), 1))
        if err is not None:
            checked += 1
            assert err < RTOL
    # cost coefficients of import and gas boiler columns
    for j in list(lp.C("gp")[:3]) + list(lp.C("qgb")[:3]):
        c0 = lp.c[j]

        def Vc(t, j=j):
            def mod(l2):
                l2.c[j] = t
            return resolve(mod)

        err = fd_check(Vc, c0, y[j], 1e-4 * max(abs(c0), 1))
        if err is not None:
            checked += 1
            assert err < RTOL
    # matrix coefficients: heat pump input in the electricity balance and electrolyzer efficiency
    A = lp.A.tocsr()
    for (r, j) in [(lp.rows("el")[i], lp.C("qL")[i]) for i in range(3)] + \
                  [(lp.rows("h2")[i], lp.C("e")[i]) for i in range(3)]:
        a0 = A[r, j]

        def Va(t, r=r, j=j):
            def mod(l2):
                M = l2.A.tolil()
                M[r, j] = t
                l2.A = M.tocsc()
            return resolve(mod)

        err = fd_check(Va, a0, -lam[r] * y[j], 1e-5 * max(abs(a0), 1))
        if err is not None:
            checked += 1
            assert err < RTOL
    assert checked >= 8
