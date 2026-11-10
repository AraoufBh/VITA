import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vita.costs import build_params  # noqa: E402
from vita.utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config("B")


@pytest.fixture(scope="session")
def par(cfg):
    return build_params(cfg)


def random_features(rng, n_days, F=4):
    """Random raw feature days (PV CF, wind CF, price, weekday) with daily structure."""
    h = np.arange(24)
    pv = np.clip(np.sin((h - 6) / 12 * np.pi), 0, None)[None, :] * rng.uniform(0.3, 1.0, (n_days, 1))
    pv = np.clip(pv + 0.03 * rng.random((n_days, 24)), 0, 1)
    wind = np.clip(rng.uniform(0.05, 0.7, (n_days, 1)) + 0.15 * rng.standard_normal((n_days, 24)), 0.0, 1.0)
    price = rng.uniform(20, 90, (n_days, 1)) + 15 * rng.standard_normal((n_days, 24))
    wk = np.repeat((rng.random((n_days, 1)) < 0.7).astype(float), 24, axis=1)
    return np.stack([pv, wind, price, wk], axis=-1)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
