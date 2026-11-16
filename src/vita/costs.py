"""Techno-economic parameters from PyPSA technology-data v0.13.0 (projection year 2030).

All monetary values are converted to EUR of 2020 with the inflation rate of the
configuration (2 %/a) using the ``currency_year`` column; capital costs are
annualized with the capital recovery factor plus fixed O&M.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .utils import DATA_RAW

# technology-data keys used by the hub model
TECH_KEYS = {
    "pv": "solar-utility",
    "wind": "onwind",
    "bat_e": "battery storage",
    "bat_p": "battery inverter",
    "ely": "electrolysis",
    "h2": "hydrogen storage tank type 1 including compressor",
    "hpL": "central air-sourced heat pump",
    "hpH": "industrial heat pump high temperature",
    "eb": "electric boiler steam",
    "gb": "gas boiler steam",
    "tes": "central water tank storage",
    "gas": "gas",
}

# verified fallback (identical to v0.13.0, used only if the CSV is unavailable)
FALLBACK = [
    ("solar-utility", "investment", 383.7312, 2020), ("solar-utility", "FOM", 2.4757, 2020),
    ("solar-utility", "lifetime", 40, 2020),
    ("onwind", "investment", 1095.8533, 2015), ("onwind", "FOM", 1.2167, 2015),
    ("onwind", "VOM", 1.4286, 2015), ("onwind", "lifetime", 30, 2015),
    ("battery storage", "investment", 151.0028, 2020), ("battery storage", "lifetime", 25, 2020),
    ("battery inverter", "investment", 170.1440, 2020), ("battery inverter", "FOM", 0.3375, 2020),
    ("battery inverter", "efficiency", 0.96, 2020), ("battery inverter", "lifetime", 10, 2020),
    ("electrolysis", "investment", 1500.0, 2020), ("electrolysis", "FOM", 4.0, 2020),
    ("electrolysis", "efficiency", 0.6217, 2020), ("electrolysis", "efficiency-heat", 0.2228, 2020),
    ("electrolysis", "lifetime", 25, 2020),
    ("hydrogen storage tank type 1 including compressor", "investment", 47.7573, 2020),
    ("hydrogen storage tank type 1 including compressor", "FOM", 1.1133, 2020),
    ("hydrogen storage tank type 1 including compressor", "lifetime", 30, 2020),
    ("central air-sourced heat pump", "investment", 906.0988, 2015),
    ("central air-sourced heat pump", "FOM", 0.2336, 2015),
    ("central air-sourced heat pump", "VOM", 2.6561, 2015),
    ("central air-sourced heat pump", "efficiency", 3.2, 2015),
    ("central air-sourced heat pump", "lifetime", 25, 2015),
    ("industrial heat pump high temperature", "investment", 941.1019, 2019),
    ("industrial heat pump high temperature", "FOM", 0.0931, 2019),
    ("industrial heat pump high temperature", "VOM", 3.2224, 2019),
    ("industrial heat pump high temperature", "efficiency", 3.05, 2019),
    ("industrial heat pump high temperature", "lifetime", 20, 2019),
    ("electric boiler steam", "investment", 70.49, 2019), ("electric boiler steam", "FOM", 1.4571, 2019),
    ("electric boiler steam", "VOM", 0.8811, 2019), ("electric boiler steam", "efficiency", 0.99, 2019),
    ("electric boiler steam", "lifetime", 25, 2019),
    ("gas boiler steam", "investment", 45.7727, 2019), ("gas boiler steam", "FOM", 4.18, 2019),
    ("gas boiler steam", "VOM", 1.007, 2019), ("gas boiler steam", "efficiency", 0.93, 2019),
    ("gas boiler steam", "lifetime", 25, 2019),
    ("central water tank storage", "investment", 3.0361, 2020),
    ("central water tank storage", "FOM", 1.0, 2015),
    ("central water tank storage", "energy to power ratio", 60.3448, 2020),
    ("central water tank storage", "lifetime", 40, 2020),
    ("gas", "fuel", 24.568, 2010), ("gas", "CO2 intensity", 0.198, np.nan),
]

DESIGN_NAMES = ["P_pv", "P_wd", "E_bat", "P_bat", "P_ely", "E_h2", "Q_L", "Q_H", "Q_eb", "Q_gb", "E_tes"]
DESIGN_LABELS = ["PV", "Wind", "Battery energy", "Battery power", "Electrolyzer", "H2 tank",
                 "LT heat pump", "HT heat pump", "Electric boiler", "Gas boiler", "Thermal tank"]


def load_costs_table() -> pd.DataFrame:
    p = DATA_RAW / "costs_2030.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(FALLBACK, columns=["technology", "parameter", "value", "currency_year"])


def _get(df, tech, par, default=None):
    s = df[(df.technology == tech) & (df.parameter == par)]
    if s.empty:
        if default is None:
            raise KeyError(f"{tech}/{par} missing")
        return default, np.nan
    return float(s.value.iloc[0]), float(s.currency_year.iloc[0])


def crf(r: float, n: float) -> float:
    return r / (1.0 - (1.0 + r) ** (-n))


@dataclass
class HubParams:
    """All constants of the hub model (MW, MWh, EUR of 2020, per year)."""

    kappa: np.ndarray            # annualized capital cost per design unit [EUR/(MW or MWh)/a]
    x_ub: np.ndarray             # upper bounds of the design variables (potentials)
    vom_wd: float
    vom_L: float
    vom_H: float
    vom_eb: float
    vom_gb: float
    c_gas: float                 # EUR/MWh_th fuel
    eps_gas: float               # tCO2/MWh_th
    c_co2: float                 # EUR/t
    eta_gb: float
    eta_eb: float
    eta_inv: float
    eta_ely: float
    eta_ely_th: float
    cop_H: float
    cop_L_nom: float
    tes_ratio: float             # energy-to-power ratio [h]
    lam_tes: float               # hourly standing loss of the thermal tank
    lam_h2: float = 0.0
    grid: float = 40.0
    imp_add: float = 30.0
    exp_sub: float = 2.0
    c_u: float = 3000.0
    d_h2: float = 8.0
    table: dict = field(default_factory=dict)

    @property
    def c_gb_fuel(self) -> float:
        """Fuel plus carbon cost per MWh of steam from the gas boiler."""
        return (self.c_gas + self.c_co2 * self.eps_gas) / self.eta_gb

    def with_co2(self, c_co2: float) -> "HubParams":
        import copy

        p = copy.copy(self)
        p.c_co2 = float(c_co2)
        return p

    def signature(self) -> tuple:
        return (tuple(np.round(self.kappa, 8)), tuple(np.round(self.x_ub, 6)), self.vom_wd, self.vom_L,
                self.vom_H, self.vom_eb, self.vom_gb, self.c_gas, self.eps_gas, self.c_co2, self.eta_gb,
                self.eta_eb, self.eta_inv, self.eta_ely, self.eta_ely_th, self.cop_H, self.cop_L_nom,
                self.tes_ratio, self.lam_tes, self.lam_h2, self.grid, self.imp_add, self.exp_sub,
                self.c_u, self.d_h2)


def build_params(cfg: dict, c_co2: float | None = None) -> HubParams:
    df = load_costs_table()
    infl = cfg["economics"]["inflation"]
    r = cfg["economics"]["discount_rate"]
    base_year = cfg["economics"]["currency_year"]

    def eur(v, cy):
        if np.isnan(cy):
            return v
        return v * (1.0 + infl) ** (base_year - cy)

    def ann(key, unit_factor=1000.0):
        tech = TECH_KEYS[key]
        inv, cy = _get(df, tech, "investment")
        fom, _ = _get(df, tech, "FOM", default=0.0)
        life, _ = _get(df, tech, "lifetime")
        capex = eur(inv, cy) * unit_factor  # per MW or MWh
        k = capex * (crf(r, life) + fom / 100.0)
        return k, dict(tech=tech, investment=inv, currency_year=cy, fom=fom, lifetime=life,
                       capex_eur2020=capex, annualized=k)

    table = {}
    kap = []
    for name, key in zip(DESIGN_NAMES, ["pv", "wind", "bat_e", "bat_p", "ely", "h2", "hpL", "hpH", "eb", "gb", "tes"]):
        k, row = ann(key)
        kap.append(k)
        table[name] = row

    def vom(key):
        v, cy = _get(df, TECH_KEYS[key], "VOM", default=0.0)
        return eur(v, cy)

    gas_fuel, gcy = _get(df, "gas", "fuel")
    eps, _ = _get(df, "gas", "CO2 intensity")
    sc = cfg["case"]
    x_ub = np.full(len(DESIGN_NAMES), np.inf)
    x_ub[0] = sc["pv_max"]
    x_ub[1] = sc["wind_max"]
    tau_h = sc["tes_time_constant_days"] * 24.0
    p = HubParams(
        kappa=np.array(kap),
        x_ub=x_ub,
        vom_wd=vom("wind"), vom_L=vom("hpL"), vom_H=vom("hpH"), vom_eb=vom("eb"), vom_gb=vom("gb"),
        c_gas=eur(gas_fuel, gcy), eps_gas=eps,
        c_co2=sc["co2_central"] if c_co2 is None else c_co2,
        eta_gb=_get(df, TECH_KEYS["gb"], "efficiency")[0],
        eta_eb=_get(df, TECH_KEYS["eb"], "efficiency")[0],
        eta_inv=_get(df, TECH_KEYS["bat_p"], "efficiency")[0],
        eta_ely=_get(df, TECH_KEYS["ely"], "efficiency")[0],
        eta_ely_th=_get(df, TECH_KEYS["ely"], "efficiency-heat")[0],
        cop_H=_get(df, TECH_KEYS["hpH"], "efficiency")[0],
        cop_L_nom=_get(df, TECH_KEYS["hpL"], "efficiency")[0],
        tes_ratio=_get(df, TECH_KEYS["tes"], "energy to power ratio")[0],
        lam_tes=1.0 - np.exp(-1.0 / tau_h),
        grid=sc["grid_limit"], imp_add=sc["import_adder"], exp_sub=sc["export_discount"],
        c_u=sc["unserved_penalty"], d_h2=sc["h2_demand"],
        table=table,
    )
    p.table["_gas"] = dict(fuel=gas_fuel, currency_year=gcy, fuel_eur2020=p.c_gas, eps=eps)
    return p
