"""Data containers, a simulator with known ground truth, and a CSV loader."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CausalData:
    """Observed data for a binary-treatment causal question.

    `true_ite` / `true_ate` are only available for simulated data, where they
    let us check how close each estimator gets to the truth.
    """
    X: pd.DataFrame                 # observed covariates
    t: np.ndarray                   # 0/1 treatment
    y: np.ndarray                   # outcome
    treatment_name: str = "treatment"
    outcome_name: str = "outcome"
    true_ite: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def true_ate(self) -> float | None:
        return None if self.true_ite is None else float(np.mean(self.true_ite))

    def frame(self) -> pd.DataFrame:
        df = self.X.copy()
        df[self.treatment_name] = self.t
        df[self.outcome_name] = self.y
        return df


def simulate(n: int = 3000, effect: float = 5.0, heterogeneity: float = 1.0,
             confounding: float = 1.0, hidden_confounding: float = 0.0,
             seed: int = 0) -> CausalData:
    """Simulate an observational job-training study.

    A training programme (treatment) is more likely to be taken by people who
    are older and healthier, and those same traits also raise earnings, so a
    naive comparison is biased (confounding). The true effect is `effect`
    on average and varies with education and health (heterogeneity).

    `hidden_confounding` adds an *unobserved* "motivation" variable that
    drives both treatment and outcome. No estimator can adjust for it, which
    is exactly what the sensitivity analysis is about.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 10, n).clip(20, 65)
    health = rng.normal(0, 1, n)
    income = rng.lognormal(3.0, 0.4, n)          # prior income, in thousands
    education = rng.binomial(1, 0.45, n)
    motivation = rng.normal(0, 1, n)

    z_age = (age - 40) / 10
    z_inc = (np.log(income) - 3.0) / 0.4
    logit = (-0.3 + confounding * (0.8 * z_age + 0.7 * health + 0.4 * z_inc)
             + hidden_confounding * 1.0 * motivation)
    p = 1.0 / (1.0 + np.exp(-logit))
    t = rng.binomial(1, p)

    ite = effect + heterogeneity * (2.0 * (education - 0.45) - 1.5 * health)
    base = (30 + 4.0 * z_age + 3.5 * health + 5.0 * z_inc + 2.0 * education
            + hidden_confounding * 4.0 * motivation)
    y = base + ite * t + rng.normal(0, 4.0, n)

    X = pd.DataFrame({"age": age, "health_score": health, "prior_income_k": income,
                      "education": education})
    return CausalData(X=X, t=t.astype(int), y=y, treatment_name="training",
                      outcome_name="earnings_k", true_ite=ite,
                      meta={"source": "simulated", "effect": effect, "heterogeneity": heterogeneity,
                            "confounding": confounding, "hidden_confounding": hidden_confounding, "seed": seed})


def load_csv(df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str]) -> CausalData:
    """Validate a user-supplied table and turn it into `CausalData`."""
    if not covariates:
        raise ValueError("Pick at least one covariate to adjust for.")
    if treatment in covariates or outcome in covariates:
        raise ValueError("Treatment and outcome cannot also be covariates.")
    cols = [treatment, outcome, *covariates]
    d = df[cols].dropna().reset_index(drop=True)
    if len(d) < 50:
        raise ValueError(f"Only {len(d)} complete rows - need at least 50.")
    t = d[treatment]
    if set(pd.unique(t)) - {0, 1, True, False}:
        raise ValueError(f"Treatment '{treatment}' must be binary (0/1).")
    if t.nunique() < 2:
        raise ValueError("Both treated and control units are needed.")
    y = pd.to_numeric(d[outcome], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Outcome '{outcome}' must be numeric.")
    X = pd.get_dummies(d[covariates], drop_first=True, dtype=float)
    return CausalData(X=X, t=t.astype(int).to_numpy(), y=y.to_numpy(float),
                      treatment_name=treatment, outcome_name=outcome, meta={"source": "csv"})
