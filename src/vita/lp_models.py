"""Sparse LP models of the sector-coupled industrial hub, solved with HiGHS.

One builder covers every model of the study:
* full-resolution design / operational LP: one period of 8760 h, cyclic storage;
* aggregated design / operational LP: K representative days with weights w_k, intra-day
  cyclic battery and, for the hydrogen and thermal tanks, the superposition formulation of
  Kotzur et al. (2018) with the day sequence sigma (``link='superposition'``);
* aggregated model without inter-period linking (every storage intra-day cyclic,
  ``link='cyclic'``), used by ablation A5 and by the single-day LPs of the CO baseline.

The bound "0 <= (1-lambda)^h S_d + intra_{sigma_d,h} <= E for all d, h" is written exactly
through per-cluster auxiliaries Smax_k >= S_d and Smin_k <= S_d (d in cluster k), which is
an exact projection because (1-lambda)^h > 0.
"""
from __future__ import annotations

import time

import highspy
import numpy as np
import scipy.sparse as sp

from .costs import DESIGN_NAMES, HubParams

OPV = ["gp", "gm", "ppv", "pwd", "bc", "bd", "sb", "e", "r", "hc", "hd", "sh", "qL", "qH", "qeb", "qgb",
       "zc", "zd", "sz", "uel", "uL", "uH", "uh2"]
IV = {n: i for i, n in enumerate(OPV)}
ID = {n: i for i, n in enumerate(DESIGN_NAMES)}
INF = highspy.kHighsInf
NX = len(DESIGN_NAMES)


def peak_mask(H: int, window=(6, 22)) -> np.ndarray:
    hod = np.arange(H) % 24
    return ((hod >= window[0]) & (hod < window[1])).astype(float)


def inputs_from_features(P: np.ndarray, case: dict, par: HubParams, tier: str) -> dict:
    """Map raw feature profiles P (NP, H, F) to every time-varying model input."""
    NP, H, _ = P.shape
    m = peak_mask(H, case["peak_hours"])[None, :]
    if tier == "B":
        wk = P[..., 3]
        return dict(cf_pv=P[..., 0], cf_wd=P[..., 1], price=P[..., 2],
                    d_el=case["el_base"] + case["el_peak_add"] * wk * m,
                    d_H=case["ht_base"] + case["ht_peak_add"] * wk * m,
                    d_L=case["lt_base"] + case["lt_peak_add"] * wk * m,
                    cop_L=np.full((NP, H), par.cop_L_nom))
    wk = P[..., 5]
    return dict(cf_pv=P[..., 0], cf_wd=P[..., 1], price=P[..., 2], d_L=P[..., 3], cop_L=P[..., 4],
                d_el=case["el_base"] + case["el_peak_add"] * wk * m,
                d_H=case["ht_base"] + case["ht_peak_add"] * wk * m)


def feature_grad(g: dict, P: np.ndarray, case: dict, tier: str) -> np.ndarray:
    """Chain rule from input sensitivities to dV/dP (same shape as P)."""
    NP, H, F = P.shape
    m = peak_mask(H, case["peak_hours"])[None, :]
    G = np.zeros_like(P)
    G[..., 0] = g["cf_pv"]
    G[..., 1] = g["cf_wd"]
    G[..., 2] = g["price"]
    dwk = (case["el_peak_add"] * g["d_el"] + case["ht_peak_add"] * g["d_H"]) * m
    if tier == "B":
        G[..., 3] = dwk + case["lt_peak_add"] * g["d_L"] * m
    else:
        G[..., 3] = g["d_L"]
        G[..., 4] = g["cop_L"]
        G[..., 5] = dwk
    return G


class HubLP:
    """Builds and solves one LP instance; keeps index maps for result extraction."""

    def __init__(self, inp: dict, w, par: HubParams, mode: str = "oper", x=None, link: str = "cyclic",
                 sigma=None, split: bool = False):
        assert mode in ("oper", "design") and link in ("cyclic", "superposition")
        self.inp = {k: np.atleast_2d(np.asarray(v, float)) for k, v in inp.items()}
        self.NP, self.H = self.inp["price"].shape
        self.N = self.NP * self.H
        self.w = np.asarray(w, float).reshape(self.NP)
        self.par, self.mode, self.link = par, mode, link
        self.x = None if x is None else np.asarray(x, float)
        self.sigma = None if sigma is None else np.asarray(sigma, int)
        # split=True replaces each design column of a full-year design LP by daily copies linked by
        # equalities; this removes dense columns and speeds up the interior point method.
        self.split = bool(split) and mode == "design" and self.NP == 1 and self.H % 24 == 0
        self._build()

    # ------------------------------------------------------------------ indices
    def C(self, v: str) -> np.ndarray:
        return IV[v] * self.N + np.arange(self.N)

    def XD(self, name: str) -> int:
        return self.xcol0 + ID[name]

    # ------------------------------------------------------------------ build
    def _build(self):
        par, NP, H, N = self.par, self.NP, self.H, self.N
        sup = self.link == "superposition"
        design = self.mode == "design"
        ncol = len(OPV) * N
        if sup:
            self.ND = len(self.sigma)
            self.S = {}
            for st in ("h2", "tes"):
                self.S[st] = dict(S=ncol + np.arange(self.ND), Smax=ncol + self.ND + np.arange(NP),
                                  Smin=ncol + self.ND + NP + np.arange(NP))
                ncol += self.ND + 2 * NP
        self.xcol0 = ncol
        if design:
            ncol += NX
        if self.split:
            self.nday = self.H // 24
            self.xs0 = ncol
            ncol += NX * self.nday
        self.ncol = ncol
        inp = self.inp
        wt = np.repeat(self.w, H)
        # unweighted hourly cost coefficients (needed for dV/dw)
        c0 = np.zeros(len(OPV) * N)
        c0[self.C("gp")] = inp["price"].ravel() + par.imp_add
        c0[self.C("gm")] = -(inp["price"].ravel() - par.exp_sub)
        c0[self.C("pwd")] = par.vom_wd
        c0[self.C("qL")] = par.vom_L
        c0[self.C("qH")] = par.vom_H
        c0[self.C("qeb")] = par.vom_eb
        c0[self.C("qgb")] = par.vom_gb + par.c_gb_fuel
        for u in ("uel", "uL", "uH", "uh2"):
            c0[self.C(u)] = par.c_u
        self.c0 = c0
        c = np.zeros(ncol)
        c[: len(c0)] = c0 * np.tile(wt, len(OPV))
        lo = np.zeros(ncol)
        hi = np.full(ncol, INF)
        hi[self.C("gp")] = par.grid
        hi[self.C("gm")] = par.grid
        if sup:
            lo[self.C("sh")] = -INF
            lo[self.C("sz")] = -INF
        if design:
            c[self.xcol0:self.xcol0 + NX] = par.kappa
            hi[self.xcol0:self.xcol0 + NX] = np.where(np.isfinite(par.x_ub), par.x_ub, INF)
        else:
            x = self.x
            hi[self.C("ppv")] = inp["cf_pv"].ravel() * x[ID["P_pv"]]
            hi[self.C("pwd")] = inp["cf_wd"].ravel() * x[ID["P_wd"]]
            for v in ("bc", "bd"):
                hi[self.C(v)] = x[ID["P_bat"]]
            hi[self.C("sb")] = x[ID["E_bat"]]
            hi[self.C("e")] = x[ID["P_ely"]]
            hi[self.C("qL")] = x[ID["Q_L"]]
            hi[self.C("qH")] = x[ID["Q_H"]]
            hi[self.C("qeb")] = x[ID["Q_eb"]]
            hi[self.C("qgb")] = x[ID["Q_gb"]]
            for v in ("zc", "zd"):
                hi[self.C(v)] = x[ID["E_tes"]] / par.tes_ratio
            if not sup:
                hi[self.C("sh")] = x[ID["E_h2"]]
                hi[self.C("sz")] = x[ID["E_tes"]]
        self.c, self.clo, self.chi = c, lo, hi

        R, Cc, V, rlo, rhi = [], [], [], [], []
        nrow = [0]
        blocks = {}

        def block(name, n, lo_, hi_):
            s = nrow[0]
            nrow[0] += n
            rlo.append(np.broadcast_to(np.asarray(lo_, float), (n,)).copy())
            rhi.append(np.broadcast_to(np.asarray(hi_, float), (n,)).copy())
            blocks[name] = (s, n)
            return s + np.arange(n)

        def add(rows, cols, vals):
            rows = np.asarray(rows)
            R.append(rows)
            Cc.append(np.broadcast_to(np.asarray(cols), rows.shape))
            V.append(np.broadcast_to(np.asarray(vals, float), rows.shape))

        hh = np.tile(np.arange(H), NP)
        pp = np.repeat(np.arange(NP), H)
        prev = pp * H + (hh - 1) % H
        r = block("el", N, inp["d_el"].ravel(), inp["d_el"].ravel())
        for v, k in (("ppv", 1), ("pwd", 1), ("gp", 1), ("bd", 1), ("uel", 1), ("gm", -1), ("bc", -1), ("e", -1)):
            add(r, self.C(v), k)
        add(r, self.C("qL"), -1.0 / inp["cop_L"].ravel())
        add(r, self.C("qH"), -1.0 / par.cop_H)
        add(r, self.C("qeb"), -1.0 / par.eta_eb)
        r = block("lt", N, inp["d_L"].ravel(), inp["d_L"].ravel())
        for v, k in (("qL", 1), ("r", 1), ("zd", 1), ("zc", -1), ("uL", 1)):
            add(r, self.C(v), k)
        r = block("ht", N, inp["d_H"].ravel(), inp["d_H"].ravel())
        for v in ("qH", "qeb", "qgb", "uH"):
            add(r, self.C(v), 1)
        r = block("h2", N, par.d_h2, par.d_h2)
        add(r, self.C("e"), par.eta_ely)
        for v, k in (("hd", 1), ("hc", -1), ("uh2", 1)):
            add(r, self.C(v), k)
        r = block("rec", N, -INF, 0.0)
        add(r, self.C("r"), 1)
        add(r, self.C("e"), -par.eta_ely_th)
        r = block("bat", N, 0.0, 0.0)
        add(r, self.C("sb"), 1)
        add(r, self.C("sb")[prev], -1)
        add(r, self.C("bc"), -par.eta_inv)
        add(r, self.C("bd"), 1.0 / par.eta_inv)
        for st, sv, cv, dv, lam in (("h2d", "sh", "hc", "hd", par.lam_h2), ("tesd", "sz", "zc", "zd", par.lam_tes)):
            r = block(st, N, 0.0, 0.0)
            add(r, self.C(sv), 1)
            add(r, self.C(cv), -1)
            add(r, self.C(dv), 1)
            if sup:
                m = hh > 0
                add(r[m], self.C(sv)[prev[m]], -(1.0 - lam))
            else:
                add(r, self.C(sv)[prev], -(1.0 - lam))
        if design:
            if self.split:
                day_of = hh // 24

                def X(name):
                    return self.xs0 + ID[name] * self.nday + day_of
            else:
                X = self.XD
            r = block("l_pv", N, -INF, 0.0)
            add(r, self.C("ppv"), 1)
            add(r, X("P_pv"), -inp["cf_pv"].ravel())
            r = block("l_wd", N, -INF, 0.0)
            add(r, self.C("pwd"), 1)
            add(r, X("P_wd"), -inp["cf_wd"].ravel())
            for nm, v, xn, k in (("l_bc", "bc", "P_bat", 1.0), ("l_bd", "bd", "P_bat", 1.0), ("l_sb", "sb", "E_bat", 1.0),
                                 ("l_e", "e", "P_ely", 1.0), ("l_qL", "qL", "Q_L", 1.0), ("l_qH", "qH", "Q_H", 1.0),
                                 ("l_qeb", "qeb", "Q_eb", 1.0), ("l_qgb", "qgb", "Q_gb", 1.0),
                                 ("l_zc", "zc", "E_tes", 1.0 / par.tes_ratio), ("l_zd", "zd", "E_tes", 1.0 / par.tes_ratio)):
                r = block(nm, N, -INF, 0.0)
                add(r, self.C(v), 1)
                add(r, X(xn), -k)
            if not sup:
                for nm, v, xn in (("l_sh", "sh", "E_h2"), ("l_sz", "sz", "E_tes")):
                    r = block(nm, N, -INF, 0.0)
                    add(r, self.C(v), 1)
                    add(r, X(xn), -1)
        if sup:
            ND, sig = self.ND, self.sigma
            ad = np.arange(ND)
            for st, sv, lam, xn in (("h2", "sh", par.lam_h2, "E_h2"), ("tes", "sz", par.lam_tes, "E_tes")):
                ell = 1.0 - lam
                S = self.S[st]
                intra = self.C(sv)
                r = block(f"inter_{st}", ND, 0.0, 0.0)
                add(r, S["S"][(ad + 1) % ND], 1)
                add(r, S["S"][ad], -ell ** H)
                add(r, intra[sig * H + H - 1], -1)
                r = block(f"smax_{st}", ND, 0.0, INF)
                add(r, S["Smax"][sig], 1)
                add(r, S["S"][ad], -1)
                r = block(f"smin_{st}", ND, 0.0, INF)
                add(r, S["S"][ad], 1)
                add(r, S["Smin"][sig], -1)
                Ecap = 0.0 if design else self.x[ID[xn]]
                r = block(f"up_{st}", N, -INF, Ecap)
                add(r, S["Smax"][pp], ell ** (hh + 1))
                add(r, intra, 1)
                if design:
                    add(r, self.XD(xn), -1)
                r = block(f"up0_{st}", NP, -INF, Ecap)
                add(r, S["Smax"], 1)
                if design:
                    add(r, self.XD(xn), -1)
                r = block(f"low_{st}", N, 0.0, INF)
                add(r, S["Smin"][pp], ell ** (hh + 1))
                add(r, intra, 1)
        if self.split:
            nd = self.nday
            for i in range(NX):
                r = block(f"split{i}", nd, 0.0, 0.0)
                cols_d = self.xs0 + i * nd + np.arange(nd)
                add(r[:1], self.xcol0 + i, 1)
                add(r[:1], cols_d[:1], -1)
                add(r[1:], cols_d[:-1], 1)
                add(r[1:], cols_d[1:], -1)
            self.chi[self.xs0:] = INF
        self.nrow = nrow[0]
        self.blocks = blocks
        self.rlo = np.concatenate(rlo)
        self.rhi = np.concatenate(rhi)
        A = sp.csc_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(Cc))), shape=(self.nrow, ncol))
        A.sum_duplicates()
        self.A = A

    def rows(self, name: str) -> np.ndarray:
        s, n = self.blocks[name]
        return s + np.arange(n)

    # ------------------------------------------------------------------ solve
    def solve(self, method: str = "simplex", tol: float = 1e-7, time_limit: float | None = None,
              basis: bool = False) -> dict:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("random_seed", 0)
        if method in ("ipm", "ipm_nox"):
            h.setOptionValue("solver", "ipm")
            h.setOptionValue("run_crossover", "on" if method == "ipm" else "off")
        else:
            h.setOptionValue("solver", "simplex")
            h.setOptionValue("simplex_strategy", 1)  # dual simplex
        h.setOptionValue("primal_feasibility_tolerance", tol)
        h.setOptionValue("dual_feasibility_tolerance", tol)
        if time_limit:
            h.setOptionValue("time_limit", float(time_limit))
        lp = highspy.HighsLp()
        lp.num_col_ = self.ncol
        lp.num_row_ = self.nrow
        lp.col_cost_ = self.c
        lp.col_lower_ = self.clo
        lp.col_upper_ = self.chi
        lp.row_lower_ = self.rlo
        lp.row_upper_ = self.rhi
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = self.A.indptr
        lp.a_matrix_.index_ = self.A.indices
        lp.a_matrix_.value_ = self.A.data
        t0 = time.perf_counter()
        h.passModel(lp)
        h.run()
        dt = time.perf_counter() - t0
        st = h.getModelStatus()
        if st != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"HiGHS status {h.modelStatusToString(st)} (mode={self.mode}, link={self.link})")
        sol = h.getSolution()
        out = dict(obj=h.getInfo().objective_function_value, y=np.array(sol.col_value),
                   rd=np.array(sol.row_dual), cd=np.array(sol.col_dual), time=dt)
        if basis:
            b = h.getBasis()
            out["col_basic"] = np.array([s == highspy.HighsBasisStatus.kBasic for s in b.col_status])
        return out

    # ------------------------------------------------------------------ results
    def mu_ub(self, sol) -> np.ndarray:
        """dV/d(upper bound) of every column (zero unless the upper bound is active)."""
        return np.minimum(sol["cd"], 0.0)

    def capacity_subgradient(self, sol) -> np.ndarray:
        """dV/dx for the operational LP (x fixed through bounds and right-hand sides)."""
        assert self.mode == "oper"
        mu = self.mu_ub(sol)
        rd = sol["rd"]
        par, inp = self.par, self.inp
        g = np.zeros(NX)
        g[ID["P_pv"]] = np.sum(inp["cf_pv"].ravel() * mu[self.C("ppv")])
        g[ID["P_wd"]] = np.sum(inp["cf_wd"].ravel() * mu[self.C("pwd")])
        g[ID["E_bat"]] = mu[self.C("sb")].sum()
        g[ID["P_bat"]] = mu[self.C("bc")].sum() + mu[self.C("bd")].sum()
        g[ID["P_ely"]] = mu[self.C("e")].sum()
        g[ID["Q_L"]] = mu[self.C("qL")].sum()
        g[ID["Q_H"]] = mu[self.C("qH")].sum()
        g[ID["Q_eb"]] = mu[self.C("qeb")].sum()
        g[ID["Q_gb"]] = mu[self.C("qgb")].sum()
        g_tes_p = (mu[self.C("zc")].sum() + mu[self.C("zd")].sum()) / par.tes_ratio
        if self.link == "superposition":
            g[ID["E_h2"]] = rd[self.rows("up_h2")].sum() + rd[self.rows("up0_h2")].sum()
            g[ID["E_tes"]] = rd[self.rows("up_tes")].sum() + rd[self.rows("up0_tes")].sum() + g_tes_p
        else:
            g[ID["E_h2"]] = mu[self.C("sh")].sum()
            g[ID["E_tes"]] = mu[self.C("sz")].sum() + g_tes_p
        return g

    def input_gradients(self, sol) -> dict:
        """Sensitivities of the optimal value w.r.t. every time-varying input and w."""
        assert self.mode == "oper"
        NP, H = self.NP, self.H
        y, rd = sol["y"], sol["rd"]
        mu = self.mu_ub(sol)
        x = self.x
        g = dict(
            d_el=rd[self.rows("el")].reshape(NP, H),
            d_L=rd[self.rows("lt")].reshape(NP, H),
            d_H=rd[self.rows("ht")].reshape(NP, H),
            cf_pv=(x[ID["P_pv"]] * mu[self.C("ppv")]).reshape(NP, H),
            cf_wd=(x[ID["P_wd"]] * mu[self.C("pwd")]).reshape(NP, H),
            price=self.w[:, None] * (y[self.C("gp")] - y[self.C("gm")]).reshape(NP, H),
        )
        g["cop_L"] = -g["d_el"] * y[self.C("qL")].reshape(NP, H) / self.inp["cop_L"] ** 2
        g["w"] = self.period_costs(sol)
        return g

    def period_costs(self, sol) -> np.ndarray:
        """Unweighted operational cost of each period at the optimum."""
        y = sol["y"][: len(self.c0)]
        cy = (self.c0 * y).reshape(len(OPV), self.NP, self.H)
        return cy.sum(axis=(0, 2))

    def hourly_cost(self, sol) -> np.ndarray:
        y = sol["y"][: len(self.c0)]
        return (self.c0 * y).reshape(len(OPV), self.NP, self.H).sum(axis=0)

    def get(self, sol, v: str) -> np.ndarray:
        return sol["y"][self.C(v)].reshape(self.NP, self.H)

    def summary(self, sol) -> dict:
        """Energy and cost indicators (weighted for aggregated models)."""
        par = self.par
        wt = self.w[:, None]
        u = {k: float((wt * self.get(sol, k)).sum()) for k in ("uel", "uL", "uH", "uh2")}
        dem = (wt * (self.inp["d_el"] + self.inp["d_L"] + self.inp["d_H"])).sum() + par.d_h2 * wt.sum() * self.H
        gas = float((wt * self.get(sol, "qgb")).sum() / par.eta_gb)
        out = dict(unserved=sum(u.values()), unserved_by=u, demand=float(dem), co2=gas * par.eps_gas,
                   imp=float((wt * self.get(sol, "gp")).sum()), exp=float((wt * self.get(sol, "gm")).sum()))
        if self.mode == "design":
            xs = sol["y"][self.xcol0:self.xcol0 + NX]
            out["x"] = xs
            out["capex"] = float(par.kappa @ xs)
            out["opex"] = float(sol["obj"] - out["capex"])
        else:
            out["opex"] = float(sol["obj"])
        out["tac"] = float(sol["obj"])
        return out


def degeneracy(h, lp, tol=1e-9):
    """Primal degeneracy (a basic variable or row activity at a bound) and dual degeneracy
    (a nonbasic, non-fixed column with zero reduced cost) of the optimal basis held by ``h``."""
    b = h.getBasis()
    sol = h.getSolution()
    cs = np.array([int(v) for v in b.col_status])
    rs = np.array([int(v) for v in b.row_status])
    basic = int(highspy.HighsBasisStatus.kBasic)
    x = np.array(sol.col_value)
    r = np.array(sol.row_value)
    d = np.array(sol.col_dual)
    lo, hi = lp.clo, lp.chi
    cb = cs == basic
    at_b = (np.abs(x - lo) <= tol * (1 + np.abs(lo))) | (np.isfinite(hi) & (np.abs(x - hi) <= tol * (1 + np.abs(hi))))
    rb = rs == basic
    rat = (np.isfinite(lp.rlo) & (np.abs(r - lp.rlo) <= tol * (1 + np.abs(lp.rlo)))) | \
          (np.isfinite(lp.rhi) & (np.abs(r - lp.rhi) <= tol * (1 + np.abs(lp.rhi))))
    primal = bool(np.any(cb & at_b) or np.any(rb & rat))
    free_nb = (~cb) & (hi > lo + tol)
    dual = bool(np.any(free_nb & (np.abs(d) <= tol)))
    return primal, dual


# ---------------------------------------------------------------------- convenience
def full_inputs(X: np.ndarray, case: dict, par: HubParams, tier: str) -> dict:
    P = X.reshape(1, -1, X.shape[-1])
    return inputs_from_features(P, case, par, tier)


def solve_full_design(X, case, par, tier, method="ipm"):
    lp = HubLP(full_inputs(X, case, par, tier), [1.0], par, mode="design", link="cyclic")
    sol = lp.solve(method=method)
    s = lp.summary(sol)
    s["time"] = sol["time"]
    s["soc_h2"] = lp.get(sol, "sh").ravel()
    s["soc_tes"] = lp.get(sol, "sz").ravel()
    return s


def solve_full_oper(X, x, case, par, tier, method="ipm", daily=False):
    lp = HubLP(full_inputs(X, case, par, tier), [1.0], par, mode="oper", x=x, link="cyclic")
    sol = lp.solve(method=method)
    s = lp.summary(sol)
    s["V"] = float(sol["obj"])
    s["tac"] = float(par.kappa @ x + sol["obj"])
    s["g"] = lp.capacity_subgradient(sol)
    s["time"] = sol["time"]
    if daily:
        hc = lp.hourly_cost(sol).reshape(-1, 24)
        s["daily_cost"] = hc.sum(1)
        un = sum(lp.get(sol, k).ravel() for k in ("uel", "uL", "uH", "uh2"))
        s["daily_unserved"] = un.reshape(-1, 24).sum(1)
        s["soc_h2"] = lp.get(sol, "sh").ravel()
        s["soc_tes"] = lp.get(sol, "sz").ravel()
    return s
