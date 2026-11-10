"""VITA: value-informed neural temporal aggregation.

The aggregation map embeds each day with a temporal convolutional encoder, softly assigns it to
K learnable prototypes and returns representative days (soft cluster means of real days, raw
units), weights (soft cluster masses) and a hard day sequence used for inter-period storage
linking.  Training matches the value function of the aggregated operational LP to the
full-resolution one on anchor designs and on Rademacher perturbations (dual-informed Taylor
targets).  Gradients through the LP are exact LP sensitivities computed from HiGHS primal and
dual solutions (custom ``torch.autograd.Function``); persistent HiGHS models are warm-started
across training steps.
"""
from __future__ import annotations

import copy
import time

import highspy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import lp_models as L
from .baselines import Aggregation

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------- networks
class ConvEncoder(nn.Module):
    def __init__(self, F, m=16, ch=32):
        super().__init__()
        self.c1 = nn.Conv1d(F, ch, 3, padding=1)
        self.c2 = nn.Conv1d(ch, ch, 3, dilation=2, padding=2)
        self.lin = nn.Linear(2 * ch, m)
        self.ln = nn.LayerNorm(m)

    def forward(self, x):  # x: (D, F, 24)
        h = Fn.gelu(self.c1(x))
        h = Fn.gelu(self.c2(h))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=-1)
        return self.ln(self.lin(h))


class TransformerEncoder(nn.Module):
    """Two-layer Transformer variant (d = 32, 4 heads) for the development loop."""

    def __init__(self, F, m=16, d=32):
        super().__init__()
        self.inp = nn.Linear(F, d)
        self.pos = nn.Parameter(torch.zeros(24, d))
        layer = nn.TransformerEncoderLayer(d, 4, dim_feedforward=64, dropout=0.0, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, 2)
        self.lin = nn.Linear(2 * d, m)
        self.ln = nn.LayerNorm(m)

    def forward(self, x):  # (D, F, 24)
        h = self.inp(x.permute(0, 2, 1)) + self.pos
        h = self.tr(h)
        return self.ln(self.lin(torch.cat([h.mean(1), h.amax(1)], dim=-1)))


class Decoder(nn.Module):
    def __init__(self, F, m=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(m, 64), nn.GELU(), nn.Linear(64, 24 * F))
        self.F = F

    def forward(self, e):
        return self.net(e).reshape(-1, 24, self.F)


class VITAMap(nn.Module):
    def __init__(self, F, K, m=16, ch=32, encoder="conv"):
        super().__init__()
        self.K, self.F = K, F
        if encoder == "conv":
            self.encoder = ConvEncoder(F, m, ch)
            dim = m
        elif encoder == "transformer":
            self.encoder = TransformerEncoder(F, m)
            dim = m
        else:  # ablation A4: prototypes in standardized input space
            self.encoder = None
            dim = 24 * F
        self.mu = nn.Parameter(torch.zeros(K, dim))

    def embed(self, Z):  # Z: (D, 24, F) standardized
        if self.encoder is None:
            return Z.reshape(Z.shape[0], -1)
        return self.encoder(Z.permute(0, 2, 1))

    def soft_assign(self, Z, tau, straight_through=False):
        E = self.embed(Z)
        d2 = ((E[:, None, :] - self.mu[None]) ** 2).sum(-1)
        A = torch.softmax(-d2 / tau, dim=1)
        if straight_through:
            Ah = Fn.one_hot(A.argmax(1), self.K).to(A.dtype)
            A = Ah + A - A.detach()
        return A, E

    @torch.no_grad()
    def hard_assign(self, Z):
        """Nearest prototype; an empty cluster receives the day nearest to its prototype
        (taken from a cluster with more than one day) so that K stays fixed."""
        E = self.embed(Z)
        d2 = ((E[:, None, :] - self.mu[None]) ** 2).sum(-1).numpy()
        lab = d2.argmin(1)
        for _ in range(self.K):
            counts = np.bincount(lab, minlength=self.K)
            empty = np.where(counts == 0)[0]
            if len(empty) == 0:
                break
            k = empty[0]
            cand = np.where(counts[lab] > 1)[0]
            d = cand[np.argmin(d2[cand, k])]
            lab[d] = k
        return lab


def aggregate_soft(A, X):
    w = A.sum(0)
    P = torch.einsum("dk,dhf->khf", A, X) / w.clamp_min(1e-9)[:, None, None]
    return P, w


def aggregate_hard(lab, X, K, snap=False, Z=None):
    P = np.zeros((K,) + X.shape[1:])
    w = np.bincount(lab, minlength=K).astype(float)
    for k in range(K):
        idx = np.where(lab == k)[0]
        if len(idx) == 0:
            continue
        if snap:  # medoid snapping: the member day nearest to the cluster mean (standardized space)
            Zm = Z[idx].reshape(len(idx), -1)
            c = Zm.mean(0)
            P[k] = X[idx[np.argmin(((Zm - c) ** 2).sum(1))]]
        else:
            P[k] = X[idx].mean(0)
    return P, w


# ----------------------------------------------------------------------------- LP layer
class PersistentAggLP:
    """Aggregated operational LP for one fixed design; inputs updated in place, warm-started."""

    options: dict = {}

    def __init__(self, P, w, sigma, x, case, par, tier, link="superposition"):
        self.case, self.par, self.tier, self.link = case, par, tier, link
        inp = L.inputs_from_features(P, case, par, tier)
        self.lp = L.HubLP(inp, w, par, mode="oper", x=x, link=link, sigma=sigma if link == "superposition" else None)
        self.sigma = None if sigma is None else np.array(sigma)
        self._load()

    def _load(self):
        lp = self.lp
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("random_seed", 0)
        h.setOptionValue("presolve", "off")
        h.setOptionValue("simplex_strategy", 1)
        for k_, v_ in self.options.items():
            h.setOptionValue(k_, v_)
        m = highspy.HighsLp()
        m.num_col_, m.num_row_ = lp.ncol, lp.nrow
        m.col_cost_, m.col_lower_, m.col_upper_ = lp.c, lp.clo, lp.chi
        m.row_lower_, m.row_upper_ = lp.rlo, lp.rhi
        m.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        m.a_matrix_.start_, m.a_matrix_.index_, m.a_matrix_.value_ = lp.A.indptr, lp.A.indices, lp.A.data
        h.passModel(m)
        self.h = h
        self.cidx = np.arange(lp.ncol, dtype=np.int32)
        self.ridx = np.arange(lp.nrow, dtype=np.int32)

    def _set_sigma(self, sigma):
        lp, H = self.lp, self.lp.H
        changed = np.where(sigma != self.sigma)[0]
        for st, sv in (("h2", "sh"), ("tes", "sz")):
            S = lp.S[st]
            intra = lp.C(sv)
            r_inter, r_max, r_min = lp.rows(f"inter_{st}"), lp.rows(f"smax_{st}"), lp.rows(f"smin_{st}")
            for d in changed:
                ko, kn = self.sigma[d], sigma[d]
                self.h.changeCoeff(int(r_inter[d]), int(intra[ko * H + H - 1]), 0.0)
                self.h.changeCoeff(int(r_inter[d]), int(intra[kn * H + H - 1]), -1.0)
                self.h.changeCoeff(int(r_max[d]), int(S["Smax"][ko]), 0.0)
                self.h.changeCoeff(int(r_max[d]), int(S["Smax"][kn]), 1.0)
                self.h.changeCoeff(int(r_min[d]), int(S["Smin"][ko]), 0.0)
                self.h.changeCoeff(int(r_min[d]), int(S["Smin"][kn]), -1.0)
        self.sigma = np.array(sigma)
        lp.sigma = self.sigma

    def solve(self, P, w, sigma, x):
        lp, par = self.lp, self.par
        inp = L.inputs_from_features(P, self.case, par, self.tier)
        lp.inp, lp.w, lp.x = inp, np.asarray(w, float), np.asarray(x, float)
        N, H = lp.N, lp.H
        wt = np.repeat(lp.w, H)
        c0 = lp.c0
        c0[lp.C("gp")] = inp["price"].ravel() + par.imp_add
        c0[lp.C("gm")] = -(inp["price"].ravel() - par.exp_sub)
        c = np.zeros(lp.ncol)
        c[: len(c0)] = c0 * np.tile(wt, len(L.OPV))
        chi = lp.chi
        chi[lp.C("ppv")] = inp["cf_pv"].ravel() * x[L.ID["P_pv"]]
        chi[lp.C("pwd")] = inp["cf_wd"].ravel() * x[L.ID["P_wd"]]
        for v in ("bc", "bd"):
            chi[lp.C(v)] = x[L.ID["P_bat"]]
        chi[lp.C("sb")] = x[L.ID["E_bat"]]
        chi[lp.C("e")] = x[L.ID["P_ely"]]
        for v, n in (("qL", "Q_L"), ("qH", "Q_H"), ("qeb", "Q_eb"), ("qgb", "Q_gb")):
            chi[lp.C(v)] = x[L.ID[n]]
        for v in ("zc", "zd"):
            chi[lp.C(v)] = x[L.ID["E_tes"]] / par.tes_ratio
        rlo, rhi = lp.rlo, lp.rhi
        for nm, k in (("el", "d_el"), ("lt", "d_L"), ("ht", "d_H")):
            r = lp.rows(nm)
            rlo[r] = inp[k].ravel()
            rhi[r] = inp[k].ravel()
        if self.link == "superposition":
            for st, n in (("h2", "E_h2"), ("tes", "E_tes")):
                rhi[lp.rows(f"up_{st}")] = x[L.ID[n]]
                rhi[lp.rows(f"up0_{st}")] = x[L.ID[n]]
            if not np.array_equal(sigma, self.sigma):
                self._set_sigma(np.asarray(sigma))
        else:
            chi[lp.C("sh")] = x[L.ID["E_h2"]]
            chi[lp.C("sz")] = x[L.ID["E_tes"]]
        lp.c = c
        h = self.h
        h.changeColsCost(lp.ncol, self.cidx, c)
        h.changeColsBounds(lp.ncol, self.cidx, lp.clo, chi)
        h.changeRowsBounds(lp.nrow, self.ridx, rlo, rhi)
        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            # rare warm-start failure: rebuild from scratch and solve cold
            self.lp = L.HubLP(inp, w, par, mode="oper", x=x, link=self.link,
                              sigma=np.asarray(sigma) if self.link == "superposition" else None)
            self.sigma = None if self.link != "superposition" else np.array(sigma)
            self._load()
            self.h.run()
            if self.h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
                raise RuntimeError("aggregated LP not optimal")
            lp = self.lp
            h = self.h
        sol = h.getSolution()
        out = dict(obj=h.getInfo().objective_function_value, y=np.array(sol.col_value),
                   rd=np.array(sol.row_dual), cd=np.array(sol.col_dual))
        self.n_solves = getattr(self, "n_solves", 0) + 1
        if self.n_solves % 50 == 1:  # sampled degeneracy check of the optimal basis
            self.degen = getattr(self, "degen", []) + [L.degeneracy(h, lp)]
        g = lp.input_gradients(out)
        gP = L.feature_grad(g, P, self.case, self.tier)
        return out["obj"], gP, g["w"]


class YearSolver:
    """Persistent warm-started aggregated LPs of one training year, one per anchor."""

    def __init__(self, case, par, tier, link="superposition"):
        self.case, self.par, self.tier, self.link = case, par, tier, link
        self.models = {}
        self.n_solves = 0
        self.t_solve = 0.0

    def solve_batch(self, P, w, sigma, items):
        vals, gPs, gws = [], [], []
        t0 = time.perf_counter()
        for key, x in items:
            mdl = self.models.get(key)
            if mdl is None:
                mdl = PersistentAggLP(P, w, sigma, x, self.case, self.par, self.tier, self.link)
                self.models[key] = mdl
            v, gP, gw = mdl.solve(P, w, sigma, x)
            vals.append(v)
            gPs.append(gP)
            gws.append(gw)
        self.n_solves += len(items)
        self.t_solve += time.perf_counter() - t0
        return np.array(vals), np.stack(gPs), np.stack(gws)


class AggValue(torch.autograd.Function):
    """V-hat for a batch of designs; backward returns exact LP sensitivities w.r.t. P and w."""

    @staticmethod
    def forward(ctx, P, w, solver, sigma, items):
        vals, gP, gw = solver.solve_batch(P.detach().numpy(), w.detach().numpy(), sigma, items)
        ctx.save_for_backward(torch.from_numpy(gP), torch.from_numpy(gw))
        return torch.from_numpy(vals)

    @staticmethod
    def backward(ctx, gout):
        gP, gw = ctx.saved_tensors
        return (gout[:, None, None, None] * gP).sum(0), (gout[:, None] * gw).sum(0), None, None, None


# ----------------------------------------------------------------------------- helpers
def pretrain_autoencoder(model: VITAMap, Zs: list, epochs=200, seed=0, lr=1e-3, batch=73):
    """Reconstruction pretraining of the encoder (decoder discarded afterwards)."""
    if model.encoder is None:
        return []
    g = torch.Generator().manual_seed(seed)
    F = model.F
    dec = Decoder(F, model.mu.shape[1])
    Z = torch.cat([torch.as_tensor(z) for z in Zs], 0)
    opt = torch.optim.Adam(list(model.encoder.parameters()) + list(dec.parameters()), lr=lr)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(Z), generator=g)
        tot = 0.0
        for i in range(0, len(Z), batch):
            zb = Z[perm[i:i + batch]]
            loss = ((dec(model.embed(zb)) - zb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(zb)
        hist.append(tot / len(Z))
    return hist


def kmeanspp_init(model: VITAMap, Zs: list, seed=0, full=True):
    """Prototype initialization on the embeddings of the training days: k-means with k-means++
    seeding (``full=True``, 10 restarts) or the k-means++ seeds only (``full=False``)."""
    from sklearn.cluster import KMeans, kmeans_plusplus

    with torch.no_grad():
        E = torch.cat([model.embed(torch.as_tensor(z)) for z in Zs], 0).numpy()
    if full:
        centers = KMeans(model.K, init="k-means++", n_init=10, random_state=seed).fit(E).cluster_centers_
    else:
        centers, _ = kmeans_plusplus(E, model.K, random_state=seed)
    with torch.no_grad():
        model.mu.copy_(torch.from_numpy(centers))


def entropy(A):
    return -(A * torch.log(A.clamp_min(1e-12))).sum(1)


def aggregated_design(P, w, sigma, case, par, tier, link="superposition"):
    inp = L.inputs_from_features(P, case, par, tier)
    lp = L.HubLP(inp, w, par, mode="design", link=link, sigma=sigma if link == "superposition" else None)
    t0 = time.perf_counter()
    sol = lp.solve()
    s = lp.summary(sol)
    s["lp_time"] = time.perf_counter() - t0
    if link == "superposition":
        s["S_h2"] = sol["y"][lp.S["h2"]["S"]]
        s["S_tes"] = sol["y"][lp.S["tes"]["S"]]
    return s


def deploy(model: VITAMap, X, Z, K, snap=False):
    t0 = time.perf_counter()
    lab = model.hard_assign(torch.as_tensor(Z))
    P, w = aggregate_hard(lab, X, K, snap=snap, Z=Z)
    return Aggregation(P, w, lab, time.perf_counter() - t0)


# ----------------------------------------------------------------------------- training
def train_vita(task: dict) -> dict:
    """Train one VITA model (Algorithm 1).  ``task`` carries the fold data and settings:
    keys: hp (hyperparameters), K, seed, case, par, tier, X (dict year -> raw days),
    Z (dict year -> standardized days), train_years, val_year, anchors (dict year -> list),
    oracle_val (TAC of the full-resolution optimum of the validation year),
    full_eval (callable(year, x) -> dict with V, g, tac), link, log."""
    from .utils import seed_everything

    hp, K, seed = task["hp"], task["K"], task["seed"]
    case, par, tier = task["case"], task["par"], task["tier"]
    link = task.get("link", "superposition")
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    years = task["train_years"]
    X = {y: torch.as_tensor(task["X"][y]) for y in years}
    Z = {y: torch.as_tensor(task["Z"][y]) for y in years}
    F = task["X"][years[0]].shape[-1]
    mean, std = task["std_mean"], task["std_std"]
    model = VITAMap(F, K, hp["m"], hp["channels"], hp.get("encoder", "conv"))
    t_start = time.perf_counter()
    ae_hist = pretrain_autoencoder(model, [task["Z"][y] for y in years], hp["pretrain_epochs"], seed)
    kmeanspp_init(model, [task["Z"][y] for y in years], seed, full=hp.get("proto_init", "kmeans") == "kmeans")
    params = [{"params": [model.mu], "lr": hp["lr_mu"]}]
    if model.encoder is not None:
        params.insert(0, {"params": list(model.encoder.parameters()), "lr": hp["lr_theta"]})
    opt = torch.optim.Adam(params, weight_decay=hp["weight_decay"])
    anchors = {y: [dict(a) for a in task["anchors"][y]] for y in years}
    M = hp["M"]
    kappa = task["par"].kappa
    norm_tac = hp.get("loss_norm", "tac") == "tac"
    for y in years:
        for a in anchors[y]:
            a["delta"] = _rademacher(a["x"], M, hp["delta_frac"], rng)
            a["norm"] = float(kappa @ a["x"] + a["V"]) if norm_tac else a["V"]
    solvers = {y: YearSolver(case, par, tier, link) for y in years}
    R, S = hp["R"], hp["S"]
    total = R * S
    use_value = hp.get("value_loss", True)
    use_onpolicy = hp.get("on_policy", True)
    alpha = hp["alpha"]
    history = []
    best = None

    def validate(r, vm_err):
        val = task["val_year"]
        agg = deploy(model, task["X"][val], task["Z"][val], K)
        des = aggregated_design(agg.P, agg.w, agg.sigma, case, par, tier, link)
        ev = task["full_eval"](val, des["x"])
        reg = (ev["tac"] - task["oracle_val"]) / task["oracle_val"] * 100.0
        history.append(dict(iter=r, val_regret=reg, value_err=vm_err, t=time.perf_counter() - t_start))
        return reg

    reg0 = validate(0, np.nan)
    best = (reg0, copy.deepcopy(model.state_dict()), 0)
    sched = {}
    for r in range(1, R + 1):
        vm_errs = []
        sched.clear()
        for s in range(S):
            step = (r - 1) * S + s
            frac = step / max(total - 1, 1)
            tau = hp["tau_start"] * (hp["tau_end"] / hp["tau_start"]) ** frac
            lam_ent = hp["lambda_ent"] * (tau - hp["tau_end"]) / (hp["tau_start"] - hp["tau_end"])
            st_flag = step >= total - hp["st_steps"]
            loss = 0.0
            for y in years:
                A, _ = model.soft_assign(Z[y], tau, st_flag)
                P, w = aggregate_soft(A, X[y])
                Pt = (P - torch.as_tensor(mean)) / torch.as_tensor(std)
                dist = ((Z[y][:, None] - Pt[None]) ** 2).mean(dim=(2, 3))
                rec = (A * dist).sum(1).mean()
                ent = entropy(A).mean()
                ly = hp["lambda_rec"] * rec + lam_ent * ent
                if use_value:
                    # day sequence for storage linking (no gradient), refreshed at the start of each
                    # block of consecutive steps so that warm starts are not invalidated every step
                    skey = (y, "sigma", s // hp.get("block", 1))
                    if skey not in sched:
                        sched[skey] = A.detach().argmax(1).numpy()
                    sigma = sched[skey]
                    pick = _minibatch(sched, y, s, anchors[y], hp, rng)
                    items = []
                    for j in pick:
                        a = anchors[y][j]
                        items.append(((a["id"], 0), a["x"]))
                        if alpha > 0:
                            for mm in range(M):
                                items.append(((a["id"], mm + 1), a["x"] + a["delta"][mm]))
                    vals = AggValue.apply(P, w, solvers[y], sigma, items)
                    per = 1 + (M if alpha > 0 else 0)
                    # residuals (absolute), their weights and the term each belongs to
                    res, wts = [], []
                    for b, j in enumerate(pick):
                        a = anchors[y][j]
                        v0 = vals[b * per]
                        vm_errs.append(abs(v0.item() - a["V"]) / abs(kappa @ a["x"] + a["V"]))
                        res.append(v0 - a["V"])
                        wts.append(a["rho"] / a["norm"] ** 2)
                        for mm in range(M if alpha > 0 else 0):
                            if hp.get("taylor", "level") == "slope":
                                # first-order target on differences of aggregated values only
                                res.append(vals[b * per + mm + 1] - v0 - float(a["g"] @ a["delta"][mm]))
                            else:
                                tgt = a["V"] + float(a["g"] @ a["delta"][mm])
                                res.append(vals[b * per + mm + 1] - tgt)
                            wts.append(a["rho"] / a["norm"] ** 2 * alpha / M)
                    res = torch.stack(res)
                    wts = torch.tensor(wts)
                    if hp.get("value_offset", False):
                        # value matching up to a year-specific constant: a constant offset of V-hat
                        # does not change the minimizer of kappa^T x + V-hat(x)
                        off = (wts * res).sum() / wts.sum()
                        res = res - off
                    ly = ly + (wts * res ** 2).sum() / len(pick)
                loss = loss + ly
            opt.zero_grad()
            loss.backward()
            opt.step()
            _reseed(model, Z, X, mean, std, tau)
        vm = float(np.mean(vm_errs)) if vm_errs else np.nan
        if use_value and use_onpolicy:
            for y in years:
                agg = deploy(model, task["X"][y], task["Z"][y], K)
                des = aggregated_design(agg.P, agg.w, agg.sigma, case, par, tier, link)
                ev = task["full_eval"](y, des["x"])
                a = dict(id=f"op{r}_{y}", x=des["x"], V=ev["V"], g=ev["g"], rho=hp["rho_onpolicy"])
                a["norm"] = float(kappa @ a["x"] + a["V"]) if norm_tac else a["V"]
                a["delta"] = _rademacher(a["x"], M, hp["delta_frac"], rng)
                anchors[y].append(a)
        reg = validate(r, vm)
        _release_memory()
        if task.get("log"):
            task["log"](f"K={K} seed={seed} iter={r} val_regret={reg:.4f} value_err={vm:.4f}")
        if reg < best[0] - 1e-12:
            best = (reg, copy.deepcopy(model.state_dict()), r)
    model.load_state_dict(best[1])
    n_solves = sum(s.n_solves for s in solvers.values())
    t_lp = sum(s.t_solve for s in solvers.values())
    degen = [dg for sv in solvers.values() for mdl in sv.models.values() for dg in getattr(mdl, "degen", [])]
    for sv in solvers.values():
        sv.models.clear()
    solvers.clear()
    _release_memory()
    return dict(state=best[1], best_iter=best[2], best_val_regret=best[0], history=history,
                degeneracy=dict(n=len(degen), primal=int(sum(p for p, _ in degen)), dual=int(sum(d for _, d in degen))),
                train_time=time.perf_counter() - t_start, n_lp=n_solves, t_lp=t_lp, ae_loss=ae_hist[-1] if ae_hist else None,
                model_kwargs=dict(F=F, K=K, m=hp["m"], ch=hp["channels"], encoder=hp.get("encoder", "conv")))


def _minibatch(sched, y, s, anch, hp, rng):
    """Minibatches of ``batch_per_year`` anchors.  Within an outer iteration a random permutation
    of the anchor set is cut into minibatches; each minibatch is kept for ``block`` consecutive
    Adam steps so that the persistent LP models are warm-started from nearby inputs."""
    nb = min(hp["batch_per_year"], len(anch))
    block = hp.get("block", 1)
    b = s // block
    if (y, b) not in sched:
        perm = sched.get((y, "perm"))
        if perm is None or len(perm) < nb:
            perm = list(rng.permutation(len(anch)))
        sched[(y, b)] = np.array(perm[:nb])
        sched[(y, "perm")] = perm[nb:]
    return sched[(y, b)]


def _release_memory():
    """Free solver instances and return freed heap memory to the operating system."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _rademacher(x, M, frac, rng):
    s = rng.choice([-1.0, 1.0], size=(M, len(x)))
    return s * frac * np.where(np.abs(x) > 1e-9, x, 0.0)


@torch.no_grad()
def _reseed(model, Z, X, mean, std, tau):
    """Move a prototype with less than one day of soft mass (summed over training years) to the
    embedding of the training day with the largest standardized reconstruction error."""
    mass = 0.0
    worst = (-1.0, None)
    for y in Z:
        A, E = model.soft_assign(Z[y], tau)
        mass = mass + A.sum(0)
        lab = A.argmax(1)
        P, w = aggregate_soft(A, X[y])
        Pt = (P - torch.as_tensor(mean)) / torch.as_tensor(std)
        err = ((Z[y] - Pt[lab]) ** 2).mean(dim=(1, 2))
        i = int(err.argmax())
        if float(err[i]) > worst[0]:
            worst = (float(err[i]), E[i].clone())
    low = torch.where(mass < 1.0)[0]
    for k in low[:1]:
        model.mu[k] = worst[1]


def load_model(state, kw):
    m = VITAMap(kw["F"], kw["K"], kw["m"], kw["ch"], kw["encoder"])
    m.load_state_dict(state)
    return m
