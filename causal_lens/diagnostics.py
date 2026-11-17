"""Checks that tell you whether a causal estimate deserves trust."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CausalData
from .estimators import fit_propensity


def ipw_weights(data: CausalData, stabilized: bool = True):
    """Return (propensity, weights) for the ATE target."""
    e = fit_propensity(data.X.to_numpy(float), data.t)
    w = np.where(data.t == 1, 1 / e, 1 / (1 - e))
    if stabilized:
        pt = data.t.mean()
        w = w * np.where(data.t == 1, pt, 1 - pt)
    return e, w


def _wmean_var(x, w):
    m = np.average(x, weights=w)
    return m, np.average((x - m) ** 2, weights=w)


def standardized_mean_differences(data: CausalData, weights: np.ndarray | None = None) -> pd.Series:
    """|SMD| per covariate between treated and control. Values below 0.1 are considered balanced."""
    w = np.ones(data.n) if weights is None else weights
    out = {}
    for col in data.X.columns:
        x = data.X[col].to_numpy(float)
        m1, v1 = _wmean_var(x[data.t == 1], w[data.t == 1])
        m0, v0 = _wmean_var(x[data.t == 0], w[data.t == 0])
        pooled = np.sqrt((v1 + v0) / 2)
        out[col] = abs(m1 - m0) / pooled if pooled > 0 else 0.0
    return pd.Series(out)


def effective_sample_size(w: np.ndarray) -> float:
    """Kish effective sample size: how many unweighted units the weighted sample is 'worth'."""
    return float(w.sum() ** 2 / (w ** 2).sum())


def e_value(rr: float) -> float:
    """VanderWeele-Ding E-value: the minimum risk-ratio strength an unmeasured
    confounder would need with both treatment and outcome to explain away `rr`."""
    rr = max(rr, 1 / rr) if rr > 0 else 1.0
    return float(rr + np.sqrt(rr * (rr - 1)))


def e_value_for_effect(ate: float, low: float, high: float, outcome_sd: float):
    """E-values for a continuous outcome, using the standard approximation
    RR ~ exp(0.91 * d) with d = effect / sd(outcome). Returns (point, ci_bound)."""
    rr = lambda eff: float(np.exp(0.91 * abs(eff) / outcome_sd))  # noqa: E731
    point = e_value(rr(ate))
    if low <= 0 <= high:
        return point, 1.0
    bound = low if ate > 0 else high
    return point, e_value(rr(bound))
