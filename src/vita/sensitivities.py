"""LP sensitivities used by VITA (Section IV-C of the paper, Proposition 2).

With marginals lambda = dV/d(rhs) (HiGHS row duals) and mu = dV/d(upper bound) (negative part of the reduced costs):
  dV/db = lambda;  dV/dc_j = y*_j;  dV/dA_ij = -lambda_i y*_j;
  dV/dCF^pv_{k,h} = P^pv mu(p^pv_{k,h});  dV/dD_{k,h} = lambda(balance_{k,h});
  dV/dpi_{k,h} = w_k (g+*_{k,h} - g-*_{k,h});  dV/dw_k = unweighted operational cost of representative day k;
  dV/dx_i = sum over all bounds (or right-hand sides) containing x_i of coefficient x dual.
The formulas are implemented as methods of :class:`vita.lp_models.HubLP`; this module exposes them as functions and
is the reference checked by ``tests/test_sensitivities_fd.py``.
"""
from __future__ import annotations

import numpy as np

from .lp_models import HubLP, feature_grad


def upper_bound_duals(lp: HubLP, sol: dict) -> np.ndarray:
    """dV/d(upper bound) of every column."""
    return lp.mu_ub(sol)


def capacity_subgradient(lp: HubLP, sol: dict) -> np.ndarray:
    """Subgradient of the operational value with respect to the 11 design capacities."""
    return lp.capacity_subgradient(sol)


def input_gradients(lp: HubLP, sol: dict) -> dict:
    """Derivatives with respect to demands, capacity factors, prices, COP and weights."""
    return lp.input_gradients(sol)


def feature_gradient(lp: HubLP, sol: dict, P: np.ndarray, case: dict, tier: str) -> np.ndarray:
    """Chain rule to the representative-day feature tensor P (K, 24, F)."""
    return feature_grad(lp.input_gradients(sol), P, case, tier)
