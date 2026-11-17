"""Average-treatment-effect estimators and CATE learners (scikit-learn only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import CausalData

CLIP = 0.02  # propensity scores are clipped to [CLIP, 1 - CLIP] to tame extreme weights


@dataclass
class Estimate:
    name: str
    ate: float
    low: float
    high: float
    note: str = ""

    @property
    def significant(self) -> bool:
        return self.low > 0 or self.high < 0


# --------------------------------------------------------------- building blocks
def fit_propensity(X, t, seed: int = 0) -> np.ndarray:
    """P(treated | X) from a regularised logistic regression, clipped."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=1000, random_state=seed))
    model.fit(X, t)
    return np.clip(model.predict_proba(X)[:, 1], CLIP, 1 - CLIP)


def _gbm(seed: int = 0):
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=8,
                                         min_samples_leaf=20, random_state=seed)


def _bootstrap(fn, X, t, y, B: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if t[idx].min() == t[idx].max():
            continue
        out.append(fn(X[idx], t[idx], y[idx]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------- point estimators
def _naive(X, t, y) -> float:
    return float(y[t == 1].mean() - y[t == 0].mean())


def _ols(X, t, y) -> float:
    Z = np.column_stack([np.ones(len(y)), t, (X - X.mean(0)) / (X.std(0) + 1e-9)])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return float(beta[1])


def _ipw(X, t, y) -> float:
    e = fit_propensity(X, t)
    w1, w0 = t / e, (1 - t) / (1 - e)
    return float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())


def aipw_scores(X, t, y, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Per-unit doubly-robust scores with cross-fitting; their mean is the ATE."""
    n = len(y)
    mu0, mu1 = np.zeros(n), np.zeros(n)
    e = np.zeros(n)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        for arm, mu in ((0, mu0), (1, mu1)):
            rows = tr[t[tr] == arm]
            mu[te] = _gbm(seed).fit(X[rows], y[rows]).predict(X[te])
        model = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=1000, random_state=seed))
        e[te] = model.fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
    e = np.clip(e, CLIP, 1 - CLIP)
    return mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)


def estimate_all(data: CausalData, B: int = 200, seed: int = 0) -> list[Estimate]:
    """Naive, regression adjustment, IPW (bootstrap CI) and AIPW (influence-function CI)."""
    X, t, y = data.X.to_numpy(float), data.t, data.y
    out = [
        Estimate("Naive difference", _naive(X, t, y), *_bootstrap(_naive, X, t, y, B, seed),
                 "Ignores confounding"),
        Estimate("Regression adjustment", _ols(X, t, y), *_bootstrap(_ols, X, t, y, B, seed),
                 "Linear outcome model"),
        Estimate("IPW", _ipw(X, t, y), *_bootstrap(_ipw, X, t, y, B, seed),
                 "Reweights by propensity"),
    ]
    psi = aipw_scores(X, t, y, seed=seed)
    ate, se = float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))
    out.append(Estimate("AIPW (doubly robust)", ate, ate - 1.96 * se, ate + 1.96 * se,
                        "Outcome model + propensity, cross-fitted"))
    return out


# --------------------------------------------------------------- CATE learners
def t_learner(data: CausalData, seed: int = 0) -> np.ndarray:
    """Fit one outcome model per arm; CATE = difference of predictions."""
    X, t, y = data.X.to_numpy(float), data.t, data.y
    m0 = _gbm(seed).fit(X[t == 0], y[t == 0])
    m1 = _gbm(seed).fit(X[t == 1], y[t == 1])
    return m1.predict(X) - m0.predict(X)


def x_learner(data: CausalData, seed: int = 0) -> np.ndarray:
    """Kuenzel et al. X-learner: impute individual effects, then blend by propensity."""
    X, t, y = data.X.to_numpy(float), data.t, data.y
    m0 = _gbm(seed).fit(X[t == 0], y[t == 0])
    m1 = _gbm(seed).fit(X[t == 1], y[t == 1])
    d1 = y[t == 1] - m0.predict(X[t == 1])     # effect imputed for treated
    d0 = m1.predict(X[t == 0]) - y[t == 0]     # effect imputed for controls
    tau1 = _gbm(seed).fit(X[t == 1], d1).predict(X)
    tau0 = _gbm(seed).fit(X[t == 0], d0).predict(X)
    g = fit_propensity(X, t, seed)
    return g * tau0 + (1 - g) * tau1
