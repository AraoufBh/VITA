"""Two VITA trainings with the same seed give identical histories and parameters."""
import numpy as np
import torch
from conftest import random_features

from vita import lp_models as L
from vita.data import Standardizer
from vita.vita import train_vita


def _setup(cfg, par):
    rng = np.random.default_rng(3)
    X = {y: random_features(rng, 14) for y in (1, 2, 3)}
    st = Standardizer([X[1], X[2]])
    case = cfg["case"]

    def full_eval(y, x):
        return L.solve_full_oper(X[y], x, case, par, "B", method="simplex")

    anchors = {}
    for y in (1, 2):
        anchors[y] = []
        for j, x in enumerate([np.array([30.0, 20, 20, 5, 13, 2, 6, 25, 5, 5, 100]),
                               np.array([60.0, 10, 40, 8, 13, 1, 7, 30, 1, 1, 150])]):
            e = full_eval(y, x)
            anchors[y].append(dict(id=f"a{j}", x=x, V=e["V"], g=e["g"], rho=1.0))
    oracle = L.solve_full_design(X[3], case, par, "B", method="simplex")["tac"]
    hp = dict(cfg["vita"])
    hp.update(R=1, S=3, pretrain_epochs=3, batch_per_year=2, block=1, st_steps=1)
    return dict(hp=hp, K=3, seed=0, case=case, par=par, tier="B", X=X, Z={y: st(X[y]) for y in X},
                train_years=[1, 2], val_year=3, anchors=anchors, oracle_val=oracle, full_eval=full_eval,
                std_mean=st.mean, std_std=st.std)


def test_training_is_deterministic(cfg, par):
    a = train_vita(_setup(cfg, par))
    b = train_vita(_setup(cfg, par))
    for ha, hb in zip(a["history"], b["history"]):
        assert abs(ha["val_regret"] - hb["val_regret"]) <= 1e-9
    for k in a["state"]:
        assert torch.equal(a["state"][k], b["state"][k])
