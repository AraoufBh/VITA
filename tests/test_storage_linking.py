"""Inter-period storage linking (superposition formulation, Kotzur et al. 2018)."""
import numpy as np
from conftest import random_features

from vita import lp_models as L


def test_identity_clustering_matches_chronological(cfg, par, rng):
    """With every day its own representative, the superposition model equals the chronological
    cyclic model (battery set to zero, since it is intra-day cyclic by construction)."""
    nd = 7
    Xd = random_features(rng, nd)
    x = np.array([25.0, 20.0, 0.0, 0.0, 10.0, 60.0, 5.0, 15.0, 8.0, 6.0, 300.0])
    full = L.HubLP(L.full_inputs(Xd, cfg["case"], par, "B"), [1.0], par, mode="oper", x=x, link="cyclic")
    v_full = full.solve()["obj"]
    agg = L.HubLP(L.inputs_from_features(Xd, cfg["case"], par, "B"), np.ones(nd), par, mode="oper", x=x,
                  link="superposition", sigma=np.arange(nd))
    v_agg = agg.solve()["obj"]
    assert abs(v_full - v_agg) <= 1e-7 * abs(v_full)


def test_identity_clustering_design(cfg, par, rng):
    nd = 5
    Xd = random_features(rng, nd)
    par0 = par.with_co2(150.0)
    par0.kappa = par0.kappa.copy()
    par0.kappa[2] = 1e9  # forbid batteries (intra-day cyclic by construction)
    full = L.HubLP(L.full_inputs(Xd, cfg["case"], par0, "B"), [1.0], par0, mode="design", link="cyclic")
    agg = L.HubLP(L.inputs_from_features(Xd, cfg["case"], par0, "B"), np.ones(nd), par0, mode="design",
                  link="superposition", sigma=np.arange(nd))
    a, b = full.solve()["obj"], agg.solve()["obj"]
    assert abs(a - b) <= 1e-7 * abs(a)


def test_reconstructed_soc_feasible(cfg, par, rng):
    """With several days per cluster, the reconstructed state of charge
    (1-lambda)^(h+1) S_d + intra_{sigma_d,h} stays within [0, E] for every day and hour."""
    nd, K = 30, 4
    Xd = random_features(rng, nd)
    sigma = rng.integers(0, K, nd)
    sigma[:K] = np.arange(K)
    P = np.stack([Xd[sigma == k].mean(0) for k in range(K)])
    w = np.bincount(sigma, minlength=K).astype(float)
    lp = L.HubLP(L.inputs_from_features(P, cfg["case"], par, "B"), w, par, mode="design",
                 link="superposition", sigma=sigma)
    sol = lp.solve()
    xs = sol["y"][lp.xcol0:lp.xcol0 + L.NX]
    for st, sv, lam, xn in (("h2", "sh", par.lam_h2, "E_h2"), ("tes", "sz", par.lam_tes, "E_tes")):
        S = sol["y"][lp.S[st]["S"]]
        intra = lp.get(sol, sv)
        hh = np.arange(24)
        soc = (1 - lam) ** (hh[None, :] + 1) * S[:, None] + intra[sigma]
        E = xs[L.ID[xn]]
        assert soc.min() >= -1e-6 and soc.max() <= E + 1e-6
        # chronological consistency of the inter-day states
        Snext = (1 - lam) ** 24 * S + intra[sigma, 23]
        assert np.allclose(Snext, np.roll(S, -1), atol=1e-6)
