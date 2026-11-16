"""Basic properties of the hub LPs on small instances."""
import numpy as np
from conftest import random_features

from vita import lp_models as L
from vita.costs import crf


def test_crf_and_kappa(par):
    assert abs(crf(0.07, 25) - 0.0858105) < 1e-6
    # PV: 383.7312 EUR/kW, FOM 2.4757 %, 40 a
    assert abs(par.kappa[0] - 383731.2 * (crf(0.07, 40) + 0.024757)) < 1e-6
    assert par.kappa.shape == (11,)


def test_design_then_operation_consistent(cfg, par, rng):
    Xd = random_features(rng, 4)
    inp = L.full_inputs(Xd, cfg["case"], par, "B")
    d = L.HubLP(inp, [365 / 4], par, mode="design", link="cyclic")
    sd = d.solve()
    s = d.summary(sd)
    o = L.HubLP(inp, [365 / 4], par, mode="oper", x=s["x"], link="cyclic")
    so = o.solve()
    assert abs(par.kappa @ s["x"] + so["obj"] - sd["obj"]) <= 1e-7 * abs(sd["obj"])
    assert s["unserved"] < 1e-6


def test_split_design_equivalent(cfg, par, rng):
    Xd = random_features(rng, 3)
    inp = L.full_inputs(Xd, cfg["case"], par, "B")
    a = L.HubLP(inp, [1.0], par, mode="design").solve()["obj"]
    b = L.HubLP(inp, [1.0], par, mode="design", split=True).solve()["obj"]
    assert abs(a - b) <= 1e-7 * abs(a)


def test_unserved_keeps_feasible(cfg, par, rng):
    Xd = random_features(rng, 2)
    inp = L.full_inputs(Xd, cfg["case"], par, "B")
    o = L.HubLP(inp, [1.0], par, mode="oper", x=np.zeros(11), link="cyclic")
    so = o.solve()
    s = o.summary(so)
    assert s["unserved"] > 0 and so["obj"] > 0


def test_weekday_demand_mapping(cfg, par):
    P = np.zeros((2, 24, 4))
    P[0, :, 3] = 1.0
    inp = L.inputs_from_features(P, cfg["case"], par, "B")
    assert inp["d_el"][0, 6] == 20.0 and inp["d_el"][0, 5] == 12.0 and inp["d_el"][1, 12] == 12.0
    assert inp["d_H"][0, 21] == 30.0 and inp["d_H"][0, 22] == 10.0
    assert inp["d_L"][0, 10] == 12.0 and inp["d_L"][1, 10] == 6.0
