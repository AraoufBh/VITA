"""Data loading, gap filling and daily feature tensors.

Tier B (used when the Open Power System Data platform is unreachable): ENTSO-E
Transparency Platform generation and day-ahead price data for Spain, 2015-2018, in the
compilation distributed with the Darts time-series library.  Timestamps are naive UTC;
they are shifted to fixed UTC+1, 29 February is dropped and every year has 8760 hours.

Tier A (OPSD time series 2020-10-06 and When2Heat 2023-07-27, Germany) is implemented in
``load_tier_a``; it was not exercised in the published run because the host was unreachable.
"""
from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import DATA_PROC, DATA_RAW, dump_json

HOURS = 24
DAYS = 365


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())


def write_checksums() -> None:
    lines = []
    for p in sorted(DATA_RAW.glob("*.csv")):
        lines.append(f"{sha256(p)}  {p.name}  {p.stat().st_size} bytes")
    (DATA_RAW / "CHECKSUMS.txt").write_text("\n".join(lines) + "\n")


def fill_gaps(s: pd.Series, max_gap: int = 3) -> tuple[pd.Series, int]:
    """Linear interpolation for gaps of at most ``max_gap`` hours; longer gaps take the
    value of the same hour one week earlier.  Returns the filled series and the count."""
    s = s.copy()
    na = s.isna()
    n_missing = int(na.sum())
    if n_missing == 0:
        return s, 0
    grp = (na != na.shift()).cumsum()
    run_len = na.groupby(grp).transform("sum")
    short = na & (run_len <= max_gap)
    interp = s.interpolate(method="linear", limit_direction="both")
    s[short] = interp[short]
    long_idx = np.where(s.isna().values)[0]
    for i in long_idx:
        j = i - 168
        while j >= 0 and np.isnan(s.iloc[j]):
            j -= 168
        s.iloc[i] = s.iloc[j] if j >= 0 else s.iloc[i + 168]
    return s, n_missing


@dataclass
class YearData:
    year: int
    X: np.ndarray          # (365, 24, F) raw features
    dates: pd.DatetimeIndex
    feature_names: list


def _weekday_tensor(dates: pd.DatetimeIndex) -> np.ndarray:
    wk = (dates.dayofweek < 5).astype(float)
    return np.repeat(wk[:, None], HOURS, axis=1)


def load_tier_b(cfg: dict) -> tuple[dict, dict]:
    path = DATA_RAW / "energy_dataset.csv"
    if not path.exists():
        download(cfg["data_file"]["url"], path)
    df = pd.read_csv(path)
    info = {"rows": int(len(df)), "sha256": sha256(path), "expected_sha256": cfg["data_file"]["sha256"]}
    t = pd.to_datetime(df["time"]) + pd.Timedelta(hours=1)  # naive UTC -> fixed UTC+1
    df.index = t
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    cols = {"pv": "generation solar", "wind": "generation wind onshore", "price": "price day ahead"}
    filled = {}
    out = {}
    q = cfg["data"]["cf_quantile"]
    for y in cfg["data"]["years"]:
        d = df[df.index.year == y]
        assert len(d) == 8760, (y, len(d))
        series = {}
        for k, c in cols.items():
            s, n = fill_gaps(d[c].astype(float), cfg["data"]["max_interp_gap"])
            filled[f"{k}_{y}"] = n
            series[k] = s.values
        cf = {}
        for k in ("pv", "wind"):
            qq = np.quantile(series[k], q)
            cf[k] = np.clip(series[k] / qq, 0.0, 1.0)
        dates = pd.DatetimeIndex(d.index[::24].normalize())
        X = np.stack([cf["pv"].reshape(DAYS, HOURS), cf["wind"].reshape(DAYS, HOURS),
                      series["price"].reshape(DAYS, HOURS), _weekday_tensor(dates)], axis=-1)
        out[y] = YearData(y, X, dates, ["pv", "wind", "price", "weekday"])
    info["filled_hours"] = filled
    return out, info


def load_tier_a(cfg: dict) -> tuple[dict, dict]:  # pragma: no cover - host unreachable in this run
    """OPSD + When2Heat processing for Germany (Section 4.1 of the study protocol)."""
    ts = pd.read_csv(DATA_RAW / "time_series_60min_singleindex.csv", parse_dates=["utc_timestamp"])
    wh = pd.read_csv(DATA_RAW / "when2heat.csv", sep=";", decimal=",", parse_dates=["utc_timestamp"])
    for d in (ts, wh):
        d.index = d["utc_timestamp"].dt.tz_localize(None) + pd.Timedelta(hours=1)
    price = ts["DE_AT_LU_price_day_ahead"].where(ts.index < "2018-10-01", ts["DE_LU_price_day_ahead"])
    frame = pd.DataFrame({"pv": ts["DE_solar_profile"], "wind": ts["DE_wind_onshore_profile"], "price": price,
                          "heat": wh["DE_heat_demand_total"], "cop": wh["DE_COP_ASHP_radiator"]})
    frame = frame[~((frame.index.month == 2) & (frame.index.day == 29))]
    out, filled = {}, {}
    for y in cfg["data"]["years"]:
        d = frame[frame.index.year == y]
        series = {}
        for k in frame.columns:
            s, n = fill_gaps(d[k].astype(float), cfg["data"]["max_interp_gap"])
            series[k] = s.values
            filled[f"{k}_{y}"] = n
        heat = series["heat"] / series["heat"].sum() * cfg["case"]["lt_annual_gwh"] * 1000.0
        cop = 3.2 * series["cop"] / series["cop"].mean()
        dates = pd.DatetimeIndex(d.index[::24].normalize())
        X = np.stack([series["pv"].reshape(DAYS, HOURS), series["wind"].reshape(DAYS, HOURS),
                      series["price"].reshape(DAYS, HOURS), heat.reshape(DAYS, HOURS),
                      cop.reshape(DAYS, HOURS), _weekday_tensor(dates)], axis=-1)
        out[y] = YearData(y, X, dates, ["pv", "wind", "price", "heat", "cop", "weekday"])
    return out, {"filled_hours": filled}


def load_data(cfg: dict) -> tuple[dict, dict]:
    if cfg["tier"] == "B":
        data, info = load_tier_b(cfg)
    else:
        data, info = load_tier_a(cfg)
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_PROC / f"features_tier{cfg['tier']}.npz",
                        **{str(y): d.X for y, d in data.items()})
    dump_json(info, DATA_PROC / f"data_info_tier{cfg['tier']}.json")
    return data, info


class Standardizer:
    """Per-feature z-scores with statistics of the training years only."""

    def __init__(self, Xs: list[np.ndarray]):
        cat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in Xs], axis=0)
        self.mean = cat.mean(0)
        self.std = cat.std(0)
        self.std[self.std < 1e-9] = 1.0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std + self.mean
