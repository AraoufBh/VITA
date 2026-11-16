"""Regenerate every CSV file, table and macro of the paper from cached results.

Once the pipeline has run, every result is read from results/ and the script takes minutes.
usage: python scripts/make_assets.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from vita import pipeline as PL  # noqa: E402
from vita.costs import DESIGN_LABELS  # noqa: E402
from vita.evaluate import fossil_reference_co2  # noqa: E402
from vita.export import GEN, Numbers, write_csv, write_params_table  # noqa: E402
from vita.stats import holm, paired_wilcoxon  # noqa: E402
from vita.utils import RESULTS  # noqa: E402

METRICS = ["regret", "cost_err", "cap_dev", "unserved", "co2_err", "runtime"]
ND = {"regret": 3, "cost_err": 2, "cap_dev": 1, "unserved": 4, "co2_err": 2, "runtime": 2}
DISPLAY = {"VITA": "VITA", "KM": "KM", "KMed": "KMed", "HC": "HC", "HCX": "HC+X", "DP": "DP", "AE": "AE",
           "CO": "CO", "IX": "IX"}
STYLE = {"VITA": "m-VITA", "KM": "m-KM", "KMed": "m-KMed", "HC": "m-HC", "HCX": "m-HCX", "DP": "m-DP", "AE": "m-AE",
         "CO": "m-CO", "IX": "m-IX"}
CENTRAL = 150.0


# ----------------------------------------------------------------------------- collection
def collect(cfg):
    st = PL.experiment_settings(cfg)
    rows = []
    for f in PL.folds(cfg):
        for c in st["prices"]:
            for K in st["K_grid"]:
                for m in PL.BASELINES:
                    r = PL.baseline_task(m, f, f["test"], c, K)
                    rows.append(dict(method=m, year=f["test"], price=c, K=K, seed=-1, **r["metrics"]))
                for s in st["seeds"]:
                    r = PL.vita_eval_task(f, K, s, "full", f["test"], c)
                    rows.append(dict(method="VITA", year=f["test"], price=c, K=K, seed=s, **r["metrics"]))
    out = pd.DataFrame(rows)
    out["unserved"] = out["unserved"].clip(lower=0.0)  # solver noise below zero reported as zero
    return out


def per_instance(df, K, metric="regret", price=None):
    d = df[df.K == K]
    if price is not None:
        d = d[d.price == price]
    return d.groupby(["method", "year", "price"])[metric].mean().unstack("method")


def wilcoxon_vs(piv, methods):
    out, pv = {}, []
    for m in methods:
        diff = (piv["VITA"] - piv[m]).values
        p = paired_wilcoxon(piv["VITA"], piv[m])
        pv.append(p)
        out[m] = dict(p=p, better=float(np.median(diff)) < 0, wins=int((diff < 0).sum()), n=len(diff))
    for m, a in zip(methods, holm(pv)):
        out[m]["p_holm"] = float(a)
    return out


def mean_sd(df, method, K, metric, price=None):
    d = df[(df.method == method) & (df.K == K)]
    if price is not None:
        d = d[d.price == price]
    return float(d[metric].mean()), float(d[metric].std(ddof=1))


def fmt_pm(mu, sd, nd):
    return f"{mu:.{nd}f} $\\pm$ {sd:.{nd}f}"


def mark_best(cells_by_method, means_by_method, nd):
    vals = sorted(set(round(v, nd) for v in means_by_method.values()))
    out = {}
    for m, s in cells_by_method.items():
        r = round(means_by_method[m], nd)
        if r == vals[0]:
            s = "\\textbf{" + s + "}"
        elif len(vals) > 1 and r == vals[1]:
            s = "\\underline{" + s + "}"
        out[m] = s
    return out


# ----------------------------------------------------------------------------- Table III
def table_main(df, Ks, n_inst):
    lines = ["\\begin{table*}[!t]", "\\centering",
             "\\caption{Results on the Temporal Test Set (Four Test Years $\\times$ Three Carbon Prices; "
             "Mean $\\pm$ Standard Deviation)\\label{tab:main}}",
             "\\footnotesize", "\\setlength{\\tabcolsep}{4.5pt}",
             "\\begin{tabular}{@{}cl" + "c" * len(METRICS) + "@{}}", "\\toprule",
             "$K$ & Method & Regret [\\%] $\\downarrow$ & Cost error [\\%] $\\downarrow$ & Capacity dev. [\\%] $\\downarrow$ & "
             "Unserved [\\%] $\\downarrow$ & CO$_2$ error [\\%] $\\downarrow$ & Runtime [s] $\\downarrow$ \\\\", "\\midrule"]
    tests = {}
    for K in Ks:
        tests[K] = wilcoxon_vs(per_instance(df, K), PL.BASELINES)
        methods = ["VITA"] + PL.BASELINES
        cols = {}
        for k in METRICS:
            means = {m: mean_sd(df, m, K, k)[0] for m in methods}
            cells = {m: fmt_pm(*mean_sd(df, m, K, k), ND[k]) for m in methods}
            cols[k] = mark_best(cells, means, ND[k])
        for i, m in enumerate(methods):
            name = DISPLAY[m]
            if m != "VITA" and tests[K][m]["p_holm"] < 0.05:
                name += "$^{\\ast}$" if tests[K][m]["better"] else "$^{\\dagger}$"
            lead = f"\\multirow{{{len(methods)}}}{{*}}{{{K}}}" if i == 0 else ""
            lines.append(f"{lead} & {name} & " + " & ".join(cols[k][m] for k in METRICS) + " \\\\")
        lines.append("\\midrule" if K != Ks[-1] else "\\bottomrule")
    lines += ["\\end{tabular}",
              "\\par\\vspace{1mm}\\parbox{0.97\\textwidth}{\\footnotesize "
              f"$n={n_inst}$ test instances per $K$; VITA statistics pool three seeds. "
              "$^{\\ast}$ ($^{\\dagger}$): VITA has lower (higher) regret than the baseline under a two-sided Wilcoxon "
              "signed-rank test on paired instance regrets (VITA averaged over seeds), Holm-corrected over the eight "
              f"baselines, $p<0.05$, $n={n_inst}$. Bold: best mean; underlined: second best. Runtime: deployment "
              "(aggregation and aggregated design LP, plus full-resolution solves for IX).}",
              "\\end{table*}"]
    (GEN / "tab_main.tex").write_text("\n".join(lines) + "\n")
    return tests


def export_dat():
    from vita import export as _ex

    return _ex.DAT


# ----------------------------------------------------------------------------- figures
def strongest(df, K=None, n=4, exclude=()):
    d = df[df.method != "VITA"]
    if K is not None:
        d = d[d.K == K]
    order = d.groupby("method")["regret"].mean().sort_values()
    return [m for m in order.index if m not in exclude][:n]


def fig_regret_vs_k(df, Ks, best4):
    rows = []
    for K in Ks:
        row = [K]
        for m in ["VITA"] + best4:
            mu, sd = mean_sd(df, m, K, "regret")
            row += [mu, sd]
        rows.append(row)
    hdr = ["K"] + [f"{c}{m}" for m in ["VITA"] + best4 for c in ("mean", "sd")]
    write_csv("fig3_regret_vs_k.csv", hdr, rows)
    means = [r[i] for r in rows for i in range(1, len(r), 2)]
    tops = [r[i] + r[i + 1] for r in rows for i in range(1, len(r), 2)]
    return min(means), max(tops)


def fig_regret_vs_price(df, prices, best3):
    rows = []
    for c in prices:
        row = [c]
        for m in ["VITA"] + best3:
            mu, sd = mean_sd(df, m, 8, "regret", c)
            row += [mu, sd]
        rows.append(row)
    write_csv("fig4a_regret_vs_price.csv", ["price"] + [f"{c}{m}" for m in ["VITA"] + best3 for c in ("mean", "sd")],
              rows)
    return rows


def fig_capacity_dev(cfg, ev, best, seeds):
    """Signed annualized-cost-weighted capacity deviation per technology (K=8, central price, mean over test years)."""
    par = ev.par(CENTRAL)
    k = par.kappa
    acc = {"VITA": [], best: []}
    for f in PL.folds(cfg):
        o = ev.oracle(f["test"], CENTRAL)
        den = float(np.sum(k * o["x"]))
        xv = np.mean([PL.vita_eval_task(f, 8, s, "full", f["test"], CENTRAL)["x"] for s in seeds], axis=0)
        xb = PL.baseline_task(best, f, f["test"], CENTRAL, 8)["x"]
        acc["VITA"].append(k * (xv - o["x"]) / den * 100)
        acc[best].append(k * (xb - o["x"]) / den * 100)
    rows = []
    for i, lab in enumerate(DESIGN_LABELS):
        rows.append([str(i), lab.replace(" ", "~"), float(np.mean([a[i] for a in acc["VITA"]])),
                     float(np.mean([a[i] for a in acc[best]]))])
    write_csv("fig4b_capacity_dev.csv", ["idx", "tech", "VITA", "base"], rows)
    return rows


def fig_storage(cfg, ev, best, year):
    """Daily state of charge of the long-duration storage with the largest inter-day swing in FULL."""
    f = [g for g in PL.folds(cfg) if g["test"] == year][0]
    o = ev.oracle(year, CENTRAL)
    daily = {st: np.asarray(o[f"soc_{st}"])[23::24] for st in ("h2", "tes")}
    swing = {st: float(daily[st].max() - daily[st].min()) for st in daily}
    st = max(swing, key=swing.get)
    v = PL.vita_eval_task(f, 8, 0, "full", year, CENTRAL)
    b = PL.baseline_task(best, f, year, CENTRAL, 8)
    full_start = np.roll(daily[st], 1)  # state at the start of each day (cyclic year)
    rows = [[d + 1, float(full_start[d]), float(v[f"S_{st}"][d]), float(b[f"S_{st}"][d])] for d in range(365)]
    write_csv("fig5a_soc.csv", ["day", "full", "vita", "base"], rows)
    return st, swing, max(r[1] for r in rows), max(max(r[2], r[3]) for r in rows)


def fig_weights(cfg, year):
    f = [g for g in PL.folds(cfg) if g["test"] == year][0]
    v = PL.vita_eval_task(f, 8, 0, "full", year, CENTRAL)
    km = PL.baseline_task("KM", f, year, CENTRAL, 8)
    rows = []
    out = {}
    for name, r in (("vita", v), ("km", km)):
        cf = r["P"][..., 0].mean(1) + r["P"][..., 1].mean(1)
        order = np.argsort(cf)
        out[name] = [(float(r["w"][k]), float(cf[k]), float(r["P"][k, 0, 3])) for k in order]
    for i in range(len(out["vita"])):
        rows.append([i + 1, out["vita"][i][0], out["vita"][i][1], out["km"][i][0], out["km"][i][1]])
    write_csv("fig5b_weights.csv", ["rank", "wvita", "cfvita", "wkm", "cfkm"], rows)
    return out


def fig_training(models):
    """Validation regret and value-matching error per outer iteration (development fold, K=8)."""
    H = [m["history"] for m in models]
    R = min(len(h) for h in H)
    rows = []
    for r in range(R):
        reg = [h[r]["val_regret"] for h in H]
        err = [h[r]["value_err"] * 100 for h in H]
        e = [x for x in err if np.isfinite(x)]
        rows.append([r, float(np.mean(reg)), float(np.min(reg)), float(np.max(reg)),
                     float(np.mean(e)) if e else float("nan"), float(np.min(e)) if e else float("nan"),
                     float(np.max(e)) if e else float("nan")])
    write_csv("fig6_training.csv", ["iter", "reg", "regmin", "regmax", "err", "errmin", "errmax"], rows)
    return rows


# ----------------------------------------------------------------------------- Tables IV, V
def table_transfer(df, prices, best3, folds):
    lines = ["\\begin{table}[!t]", "\\centering",
             "\\caption{Transfer Results at $K=8$: Regret [\\%] by Carbon Price (VITA Trained at 150\\,EUR/t) "
             "and by Held-Out Year (Central Price)\\label{tab:transfer}}",
             "\\footnotesize", "\\setlength{\\tabcolsep}{2.2pt}",
             "\\begin{tabular}{@{}l" + "c" * (1 + len(best3)) + "@{}}", "\\toprule",
             "Instance & VITA & " + " & ".join(DISPLAY[m] for m in best3) + " \\\\", "\\midrule",
             "\\multicolumn{%d}{@{}l}{Scenario transfer (mean $\\pm$ s.d. over four test years)} \\\\" % (2 + len(best3))]
    methods = ["VITA"] + best3
    for c in prices:
        means = {m: mean_sd(df, m, 8, "regret", c)[0] for m in methods}
        cells = {m: fmt_pm(*mean_sd(df, m, 8, "regret", c), 3) for m in methods}
        cells = mark_best(cells, means, 3)
        lines.append(f"{int(c)} EUR/t & " + " & ".join(cells[m] for m in methods) + " \\\\")
    lines += ["\\midrule", "\\multicolumn{%d}{@{}l}{Temporal transfer (leave one year out, 150 EUR/t)} \\\\" % (2 + len(best3))]
    for f in folds:
        d = df[(df.K == 8) & (df.price == CENTRAL) & (df.year == f["test"])]
        means = {m: float(d[d.method == m]["regret"].mean()) for m in methods}
        cells = {}
        for m in methods:
            if m == "VITA":
                sd = float(d[d.method == m]["regret"].std(ddof=1))
                cells[m] = fmt_pm(means[m], sd, 3)
            else:
                cells[m] = f"{means[m]:.3f}"
        cells = mark_best(cells, means, 3)
        lines.append(f"Test {f['test']} & " + " & ".join(cells[m] for m in methods) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\vspace{1mm}\\parbox{0.98\\columnwidth}{\\footnotesize Temporal rows: VITA mean $\\pm$ s.d. over three "
              "seeds. Spatial transfer is not possible with the single-region data tier.}",
              "\\end{table}"]
    (GEN / "tab_transfer.tex").write_text("\n".join(lines) + "\n")


def collect_ablations(cfg, seeds, abl_base):
    rows = []
    for f in PL.folds(cfg):
        y = f["test"]
        for s in seeds:
            for var, kw in (("full", {}), ("A1", {}), ("A2", {}), ("A3", {}), ("A4", {}),
                            ("A5", dict(link="cyclic")), ("A6", dict(snap=True))):
                v = "full" if var in ("A5", "A6") else var
                r = PL.vita_eval_task(f, 8, s, v, y, CENTRAL, **kw)
                rows.append(dict(variant=var, year=y, seed=s, **r["metrics"]))
        b = PL.baseline_task(abl_base, f, y, CENTRAL, 8)
        rows.append(dict(variant="base", year=y, seed=-1, **b["metrics"]))
        b5 = PL.baseline_task(abl_base, f, y, CENTRAL, 8, link="cyclic")
        rows.append(dict(variant="base_A5", year=y, seed=-1, **b5["metrics"]))
    return pd.DataFrame(rows)


def table_ablations(da, abl_base, seeds):
    names = [("full", "VITA (full method)"), ("A1", "A1 no value loss"), ("A2", "A2 no Taylor targets"),
             ("A3", "A3 no on-policy anchors"), ("A4", "A4 no encoder"), ("A5", "A5 no storage linking"),
             ("A6", "A6 medoid snapping"), ("base", f"{DISPLAY[abl_base]} (best on validation)"),
             ("base_A5", f"{DISPLAY[abl_base]}, no storage linking")]
    means = {k: float(da[da.variant == k]["regret"].mean()) for k, _ in names}
    lines = ["\\begin{table}[!t]", "\\centering",
             "\\caption{Ablations at $K=8$ and 150\\,EUR/t on the Four Test Years\\label{tab:ablation}}",
             "\\footnotesize", "\\setlength{\\tabcolsep}{2pt}", "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
             "Variant & Regret [\\%] $\\downarrow$ & Cost err. [\\%] $\\downarrow$ & Cap. dev. [\\%] $\\downarrow$ \\\\",
             "\\midrule"]
    cells = {}
    for k, _ in names:
        d = da[da.variant == k]
        cells[k] = [fmt_pm(float(d["regret"].mean()), float(d["regret"].std(ddof=1)), 3),
                    f"{float(d['cost_err'].mean()):.2f}", f"{float(d['cap_dev'].mean()):.1f}"]
    best = mark_best({k: cells[k][0] for k, _ in names}, means, 3)
    for k, lab in names:
        lines.append(f"{lab} & {best[k]} & {cells[k][1]} & {cells[k][2]} \\\\")
        if k == "A6":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\vspace{1mm}\\parbox{0.98\\columnwidth}{\\footnotesize Regret: mean $\\pm$ s.d. over the four test "
              f"years with {len(seeds)} seed; cost error and capacity deviation: means. A5 and A6 change only the deployment "
              "of the trained map.}", "\\end{table}"]
    (GEN / "tab_ablation.tex").write_text("\n".join(lines) + "\n")
    return means


def gradient_shares(cfg, ev, fold, K=8, seed=0, tau=1.0):
    """Share of the value-loss gradient carried by each feature channel, in standardized units:
    ||dL/dP[..., f]|| * std_f, for the trained map on the first training year (anchors of that year)."""
    from vita.vita import AggValue, YearSolver, aggregate_soft, load_model

    tv = PL.vita_task(fold, K, seed)
    model = load_model(tv["state"], tv["model_kwargs"])
    st = PL.standardizer(fold)
    y = fold["train"][0]
    X = PL.ctx()[1][y].X
    par = ev.par(CENTRAL)
    anchors = PL.anchor_task(fold, y, K)
    Xt = torch.as_tensor(X)
    Z = torch.as_tensor(st(X))
    A, _ = model.soft_assign(Z, tau)
    P, w = aggregate_soft(A, Xt)
    P.retain_grad()
    sigma = A.detach().argmax(1).numpy()
    solver = YearSolver(cfg["case"], par, cfg["tier"])
    vals = AggValue.apply(P, w, solver, sigma, [((a["id"], 0), a["x"]) for a in anchors])
    tac = torch.tensor([float(par.kappa @ a["x"] + a["V"]) for a in anchors])
    V = torch.tensor([a["V"] for a in anchors])
    loss = (((vals - V) / tac) ** 2).mean()
    loss.backward()
    g = P.grad.numpy()
    norms = np.array([np.linalg.norm(g[..., f]) * st.std[f] for f in range(g.shape[-1])])
    return norms / norms.sum() * 100


def saturation(cfg, fold, K=8, seed=0):
    """Percentage of training days whose largest soft assignment exceeds 0.99 at several temperatures."""
    from vita.vita import load_model

    tv = PL.vita_task(fold, K, seed)
    model = load_model(tv["state"], tv["model_kwargs"])
    st = PL.standardizer(fold)
    Z = torch.as_tensor(st(PL.ctx()[1][fold["train"][0]].X))
    out = {}
    for tau, name in ((1.0, "One"), (0.37, "Late"), (0.05, "End")):
        with torch.no_grad():
            A, _ = model.soft_assign(Z, tau)
        out[name] = float((A.max(1).values > 0.99).double().mean() * 100)
    return out


# ----------------------------------------------------------------------------- main
def word(K):
    return {4: "Kfour", 8: "Keight", 16: "Ksixteen"}[K]


def main():
    cfg, data, ev = PL.ctx()
    st = PL.experiment_settings(cfg)
    nums = Numbers()
    write_params_table(cfg, ev.par(CENTRAL))
    info = PL._STATE["info"]
    fh = info["filled_hours"]
    nums.add("DataRows", f"{info['rows']:,}".replace(",", "{,}"))
    nums.add("FilledPVTotal", sum(v for k, v in fh.items() if k.startswith("pv_")))
    nums.add("FilledWindTotal", sum(v for k, v in fh.items() if k.startswith("wind_")))
    orc = {(y, c): ev.oracle(y, c) for y in cfg["data"]["years"] for c in st["prices"]}
    # median: one solve was paused (SIGSTOP) during the run, which inflates its wall time
    nums.add("OracleTimeMedian", float(np.median([o["time"] for o in orc.values()])), 0)
    full_t = [ev.full_eval(y, c, o["x"])["time"] for (y, c), o in orc.items()]
    nums.add("FullOperTimeMean", float(np.mean(full_t)), 1)
    df = collect(cfg)
    df.to_csv(RESULTS / "metrics_test.csv", index=False)
    Ks = cfg["experiment"]["K_main"]
    n_inst = len(per_instance(df, Ks[0]))
    nums.add("NTest", n_inst)
    tests = table_main(df, Ks, n_inst)
    # headline numbers
    for K in st["K_grid"]:
        w = word(K)
        mu, sd = mean_sd(df, "VITA", K, "regret")
        nums.add(f"RegVITA{w}", mu, 3)
        nums.add(f"RegSdVITA{w}", sd, 3)
        b = strongest(df, K, 1)[0]
        nums.add(f"BestBaseName{w}", DISPLAY[b])
        nums.add(f"RegBestBase{w}", mean_sd(df, b, K, "regret")[0], 3)
        for m in PL.BASELINES:
            nums.add(f"Reg{m}{w}", mean_sd(df, m, K, "regret")[0], 3)
        for k in ("cost_err", "cap_dev", "unserved", "co2_err", "runtime"):
            tag = {"cost_err": "CostErr", "cap_dev": "CapDev", "unserved": "Unserved", "co2_err": "COErr",
                   "runtime": "Runtime"}[k]
            nums.add(f"{tag}VITA{w}", mean_sd(df, "VITA", K, k)[0], ND[k])
            nums.add(f"{tag}BestBase{w}", mean_sd(df, b, K, k)[0], ND[k])
    for K in Ks:
        w = word(K)
        t = tests[K]
        nums.add(f"WinsVsBest{w}", t[strongest(df, K, 1)[0]]["wins"])
        sig_better = sum(1 for m in PL.BASELINES if t[m]["p_holm"] < 0.05 and t[m]["better"])
        sig_worse = sum(1 for m in PL.BASELINES if t[m]["p_holm"] < 0.05 and not t[m]["better"])
        nums.add(f"NSigBetter{w}", sig_better)
        nums.add(f"NSigWorse{w}", sig_worse)
        nums.add(f"MinPHolm{w}", min(t[m]["p_holm"] for m in PL.BASELINES), 3)
        nums.add(f"PHolmBest{w}", t[strongest(df, K, 1)[0]]["p_holm"], 3)
    # runtimes and speed-up
    vit_rt = mean_sd(df, "VITA", 8, "runtime")[0]
    nums.add("SpeedupKeight", float(np.median([o["time"] for o in orc.values()])) / vit_rt, 0)
    nums.add("RuntimeIXKeight", mean_sd(df, "IX", 8, "runtime")[0], 1)
    # figures
    best4 = strongest(df, None, 4)
    lo, hi = fig_regret_vs_k(df, st["K_grid"], best4)
    (GEN / "fig3_series.tex").write_text("\n".join(
        [f"\\gdef\\FigThreeName{chr(65 + i)}{{{DISPLAY[m]}}}\\gdef\\FigThreeStyle{chr(65 + i)}{{{STYLE[m]}}}"
         f"\\gdef\\FigThreeKey{chr(65 + i)}{{{m}}}" for i, m in enumerate(best4)]) + "\n")
    grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    nums.add("FigThreeYmin", max(g for g in grid if g <= 0.8 * lo), 2)
    nums.add("FigThreeYmax", min(g for g in grid if g >= 1.1 * hi), 0)
    best3 = strongest(df, 8, 3)
    fig_regret_vs_price(df, st["prices"], best3)
    bestK8 = strongest(df, 8, 1)[0]
    (GEN / "fig4_series.tex").write_text("\n".join(
        [f"\\gdef\\FigFourName{chr(65 + i)}{{{DISPLAY[m]}}}\\gdef\\FigFourStyle{chr(65 + i)}{{{STYLE[m]}}}"
         f"\\gdef\\FigFourKey{chr(65 + i)}{{{m}}}" for i, m in enumerate(best3)]
        + [f"\\gdef\\FigFourCapColor{{c{bestK8.replace('HCX', 'HCX')}}}"]) + "\n")
    capd = fig_capacity_dev(cfg, ev, bestK8, st["seeds"])
    nums.add("CapDevBaseName", DISPLAY[bestK8])
    for i, nm in ((0, "PV"), (1, "Wind"), (2, "BatE"), (7, "QH")):
        nums.add(f"CapDev{nm}VITA", abs(capd[i][2]), 2)  # magnitudes; directions are stated in the text
        nums.add(f"CapDev{nm}Base", abs(capd[i][3]), 2)
    year5 = max(cfg["data"]["years"])
    stg, swing, fmax, amax = fig_storage(cfg, ev, bestK8, year5)
    soc = pd.read_csv(export_dat() / "fig5a_soc.csv")
    soc["dow"] = pd.date_range(f"{year5}-01-01", periods=len(soc), freq="D").dayofweek
    g = soc.groupby("dow")[["full", "vita", "base"]].mean()
    for col, nm in (("full", "Full"), ("vita", "VITA"), ("base", "Base")):
        nums.add(f"SOC{nm}Mon", float(g.loc[0, col]), 0)
        nums.add(f"SOC{nm}Sat", float(g.loc[5, col]), 0)
    nums.add("CorrSOCVITA", float(np.corrcoef(soc.full, soc.vita)[0, 1]), 2)
    nums.add("CorrSOCBase", float(np.corrcoef(soc.full, soc.base)[0, 1]), 2)
    nums.add("FigFiveStorage", "hydrogen tank" if stg == "h2" else "thermal tank")
    nums.add("FigFiveYear", year5)
    nums.add("FigFiveSwing", swing[stg], 0)
    wts = fig_weights(cfg, year5)
    for nm, key in (("VITA", "vita"), ("KM", "km")):
        nums.add(f"NMixed{nm}", int(sum(1 for _, _, wk_ in wts[key] if 1e-6 < wk_ < 1 - 1e-6)))
        nums.add(f"NDaysMixed{nm}", int(sum(w_ for w_, _, wk_ in wts[key] if 1e-6 < wk_ < 1 - 1e-6)))
    wv = sorted([w_ for w_, _, _ in wts["vita"]], reverse=True)
    wk = [w_ for w_, _, _ in wts["km"]]
    nums.add("WVitaTopThree", int(sum(wv[:3])))
    nums.add("NVitaSmall", int(sum(1 for w_ in wv if w_ <= 2)))
    small_cf = [cf for w_, cf, _ in wts["vita"] if w_ <= 2]
    nums.add("CFVitaSmallMin", float(min(small_cf)) if small_cf else float("nan"), 2)
    nums.add("CFVitaSmallMax", float(max(small_cf)) if small_cf else float("nan"), 2)
    nums.add("WKMMin", int(min(wk)))
    nums.add("WKMMax", int(max(wk)))
    # training dynamics on the development fold (frozen configuration)
    fdev = PL.dev_fold(cfg)
    models = [PL.vita_task(fdev, 8, s) for s in st["seeds"]]
    tr = fig_training(models)

    nums.add("BestIterMean", float(np.mean([m["best_iter"] for m in models])), 1)
    deg = [m.get("degeneracy", {"n": 0, "primal": 0, "dual": 0}) for f in PL.folds(cfg) for K in st["K_grid"]
           for m in [PL.vita_task(f, K, s) for s in st["seeds"]]]
    n = sum(d["n"] for d in deg)
    nums.add("DegenSampled", n)
    nums.add("DegenPrimalPct", 100.0 * sum(d["primal"] for d in deg) / max(n, 1), 1)
    nums.add("DegenDualPct", 100.0 * sum(d["dual"] for d in deg) / max(n, 1), 1)
    # development fold: validation regrets of the frozen configuration and of the baselines
    for K in Ks:
        w = word(K)
        dv = [PL.vita_task(fdev, K, s_)["best_val_regret"] for s_ in st["seeds"]]
        base = {m: PL.baseline_task(m, fdev, fdev["val"], CENTRAL, K)["metrics"]["regret"]
                for m in ["KM", "KMed", "HC", "HCX", "DP", "AE", "CO"]}
        bm = min(base, key=base.get)
        nums.add(f"DevVITA{w}", float(np.mean(dv)), 3)
        nums.add(f"DevVITAMin{w}", float(np.min(dv)), 3)
        nums.add(f"DevVITAMax{w}", float(np.max(dv)), 3)
        nums.add(f"DevBestBase{w}", base[bm], 3)
        nums.add(f"DevBestBaseName{w}", DISPLAY[bm])
        nums.add(f"DevTarget{w}", (0.85 if K == 8 else 1.0) * base[bm], 3)
    allm = [PL.vita_task(f, K, s_) for f in PL.folds(cfg) for K in st["K_grid"] for s_ in st["seeds"]]
    nums.add("NRuns", len(allm))
    nums.add("NInitSelected", sum(1 for m in allm if m["best_iter"] == 0))
    nums.add("TrainTimeKsixteen", float(np.mean([m["train_time"] for m in allm if m["K"] == 16])) / 60, 1)
    nums.add("TrainTimeKfour", float(np.mean([m["train_time"] for m in allm if m["K"] == 4])) / 60, 1)
    nums.add("TrainTimeKeight", float(np.mean([m["train_time"] for m in allm if m["K"] == 8])) / 60, 1)
    v = df[df.method == "VITA"]
    b = df[df.method != "VITA"]
    nums.add("NCatVITA", int((v.regret > 5).sum()))
    nums.add("NVITAEvals", len(v))
    nums.add("NCatBase", int((b.regret > 5).sum()))
    nums.add("NBaseEvals", len(b))
    nums.add("MaxRegVITA", float(v.regret.max()), 1)
    nums.add("MedianRegVITAKeight", float(per_instance(df, 8)["VITA"].median()), 3)
    nums.add("MedianRegBestKeight", float(per_instance(df, 8)[strongest(df, 8, 1)[0]].median()), 3)
    # largest VITA regret and unserved energy
    imax = v.regret.idxmax()
    nums.add("MaxRegVITAYear", int(v.loc[imax, "year"]))
    nums.add("MaxRegVITAPrice", int(v.loc[imax, "price"]))
    nums.add("MaxRegVITAK", int(v.loc[imax, "K"]))
    nums.add("MaxRegVITAUnserved", float(v.loc[imax, "unserved"]), 2)
    nums.add("NUnservedEvals", int((df.unserved > 1e-6).sum()))
    nums.add("NAllEvals", len(df))
    nums.add("MaxRegBase", float(b.regret.max()), 1)
    nums.add("MaxRegBaseName", DISPLAY[b.loc[b.regret.idxmax(), "method"]])
    for K in st["K_grid"]:
        nums.add(f"MedianRegVITA{word(K)}Inst", float(per_instance(df, K)["VITA"].median()), 3)
    pw = {80.0: "Eighty", 150.0: "OneFifty", 250.0: "TwoFifty"}
    for c in st["prices"]:
        for m in ["VITA", "KM", "KMed", "HC"]:
            nums.add(f"Reg{m}Keight{pw[c]}", mean_sd(df, m, 8, "regret", c)[0], 3)
        xs = np.array([ev.oracle(y, c)["x"] for y in cfg["data"]["years"]])
        for i, nm in ((7, "QH"), (8, "Qeb"), (9, "Qgb")):
            nums.add(f"Oracle{nm}{pw[c]}", float(xs[:, i].mean()), 1)
        nums.add(f"OracleCO{pw[c]}", float(np.mean([ev.oracle(y, c)["co2"] for y in cfg["data"]["years"]])) / 1000, 1)
        nums.add(f"OracleNoCO{pw[c]}", sum(1 for y in cfg["data"]["years"] if ev.oracle(y, c)["co2"] < 1.0))
    ps = df[(df.K == 8) & (df.price == CENTRAL) & (df.method == "VITA")].groupby("seed").regret.mean()
    nums.add("VITASeedMinKeight", float(ps.min()), 3)
    nums.add("VITASeedMaxKeight", float(ps.max()), 3)
    d815 = df[(df.K == 8) & (df.price == CENTRAL) & (df.method == "VITA")].groupby("year").regret.mean()
    nums.add("VITAYearMinKeight", float(d815.min()), 3)
    nums.add("VITAYearMaxKeight", float(d815.max()), 3)
    sat = saturation(cfg, fdev)
    for t, v_ in sat.items():
        nums.add(f"SatTau{t}", v_, 1)
    sh = gradient_shares(cfg, ev, fdev)
    for f, nm in enumerate(["PV", "Wind", "Price", "Weekday"]):
        nums.add(f"GradShare{nm}", float(sh[f]), 1)
    # transfer and ablations
    folds = PL.folds(cfg)
    table_transfer(df, st["prices"], best3, folds)
    abl_seeds = cfg["experiment"].get("ablation_seeds", st["seeds"])
    abl_base = cfg["experiment"].get("ablation_baseline", "KMed")
    da = collect_ablations(cfg, abl_seeds, abl_base)
    da.to_csv(RESULTS / "metrics_ablation.csv", index=False)
    am = table_ablations(da, abl_base, abl_seeds)
    for k, v in am.items():
        nums.add("Abl" + {"full": "Full", "A1": "One", "A2": "Two", "A3": "Three", "A4": "Four", "A5": "Five",
                          "A6": "Six", "base": "Base", "base_A5": "BaseFive"}[k], v, 3)
    nums.write()
    # results summary for the README (mean test regret in %, twelve instances per K)
    lines = ["| Method | " + " | ".join(f"K = {K}" for K in st["K_grid"]) + " |",
             "|---|" + "---|" * len(st["K_grid"])]
    for m in ["VITA"] + PL.BASELINES:
        lines.append(f"| {DISPLAY[m]} | " + " | ".join(f"{mean_sd(df, m, K, 'regret')[0]:.3f}" for K in st["K_grid"]) + " |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"assets written: {len(nums.m)} macros")
    return nums


if __name__ == "__main__":
    main()
