"""Shared helpers: paths, seeding, hashing, on-disk caching, timing and logging."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
PAPER = ROOT / "paper"


def set_single_thread_env() -> None:
    """Pin BLAS/OpenMP to one thread inside worker processes (determinism)."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
    except ImportError:  # torch is optional for LP-only utilities
        pass


def stable_hash(obj) -> str:
    """SHA-1 of a canonical byte representation (numpy arrays hashed by content)."""
    h = hashlib.sha1()

    def upd(o):
        if isinstance(o, np.ndarray):
            a = np.ascontiguousarray(o)
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        elif isinstance(o, dict):
            for k in sorted(o):
                h.update(str(k).encode())
                upd(o[k])
        elif isinstance(o, (list, tuple)):
            h.update(b"[")
            for e in o:
                upd(e)
            h.update(b"]")
        elif isinstance(o, float):
            h.update(repr(round(o, 12)).encode())
        else:
            h.update(repr(o).encode())

    upd(obj)
    return h.hexdigest()


def cache_get(key: str, sub: str = "lp"):
    p = CACHE / sub / key[:2] / f"{key}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def cache_put(key: str, value, sub: str = "lp") -> None:
    p = CACHE / sub / key[:2] / f"{key}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)


@contextmanager
def timer():
    t = {"start": time.perf_counter(), "elapsed": None}
    yield t
    t["elapsed"] = time.perf_counter() - t["start"]


def log(msg: str, path: Path | None = None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(line + "\n")


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def deep_update(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(tier: str = "B") -> dict:
    import yaml

    with open(ROOT / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    if tier == "B":
        with open(ROOT / "configs" / "tier_b.yaml") as f:
            cfg = deep_update(cfg, yaml.safe_load(f))
    return cfg
