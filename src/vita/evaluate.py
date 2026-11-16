"""Full-resolution evaluation of designs (cached on disk) and evaluation metrics."""
from __future__ import annotations

import numpy as np

from . import lp_models as L
from .costs import build_params
from .utils import cache_get, cache_put, stable_hash


class Evaluator:
    """Oracle designs and full-resolution operational evaluations, keyed by a hash of inputs."""

    def __init__(self, cfg: dict, data: dict):
        self.cfg, self.data = cfg, data
        self.case, self.tier = cfg["case"], cfg["tier"]
        self._par = {}

    def par(self, c_co2: float):
        c = float(c_co2)
        if c not in self._par:
            self._par[c] = build_params(self.cfg, c)
        return self._par[c]

    def _key(self, kind, year, c_co2, x=None):
        X = self.data[year].X
        return stable_hash([kind, self.tier, int(year), float(c_co2), self.par(c_co2).signature(),
                            np.round(X, 10), None if x is None else np.round(np.asarray(x, float), 9)])

    def oracle(self, year: int, c_co2: float) -> dict:
        key = self._key("oracle", year, c_co2)
        res = cache_get(key, "oracle")
        if res is None:
            res = L.solve_full_design(self.data[year].X, self.case, self.par(c_co2), self.tier, method="simplex")
            cache_put(key, res, "oracle")
        return res

    def full_eval(self, year: int, c_co2: float, x, daily: bool = False) -> dict:
        x = np.maximum(np.asarray(x, float), 0.0)
        key = self._key("eval" + ("d" if daily else ""), year, c_co2, x)
        res = cache_get(key, "eval")
        if res is None:
            res = L.solve_full_oper(self.data[year].X, x, self.case, self.par(c_co2), self.tier,
                                    method="simplex", daily=daily)
            cache_put(key, res, "eval")
        return res


def fossil_reference_co2(X, case, par, tier) -> float:
    """Emissions if all high-temperature heat came from the gas boiler (normalization of the CO2 metric)."""
    inp = L.full_inputs(X, case, par, tier)
    return float(inp["d_H"].sum() / par.eta_gb * par.eps_gas)


def design_metrics(agg_design: dict, ev: dict, oracle: dict, par, co2_ref: float) -> dict:
    """Metrics of Section 9.3 for one aggregated design evaluated at full resolution."""
    xh, xs = agg_design["x"], oracle["x"]
    tac_full = ev["tac"]
    k = par.kappa
    return dict(
        regret=(tac_full - oracle["tac"]) / oracle["tac"] * 100.0,
        cost_err=abs(agg_design["tac"] - tac_full) / tac_full * 100.0,
        cap_dev=float(np.sum(k * np.abs(xh - xs)) / np.sum(k * xs) * 100.0),
        unserved=ev["unserved"] / ev["demand"] * 100.0,
        co2_err=abs(ev["co2"] - oracle["co2"]) / co2_ref * 100.0,
        tac_full=tac_full, tac_agg=agg_design["tac"], co2_full=ev["co2"], co2_agg=agg_design["co2"],
    )
