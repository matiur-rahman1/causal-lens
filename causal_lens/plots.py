"""Matplotlib figures used by the app (also usable on their own)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

TEAL, ROSE, SLATE, AMBER = "#0d9488", "#e11d48", "#64748b", "#d97706"


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def forest(estimates, truth: float | None = None):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for i, e in enumerate(reversed(estimates)):
        colour = SLATE if e.name.startswith("Naive") else TEAL
        ax.errorbar(e.ate, i, xerr=[[e.ate - e.low], [e.high - e.ate]], fmt="o", color=colour,
                    capsize=4, lw=2, ms=7)
        ax.text(e.high, i + 0.22, f"  {e.ate:.2f}", fontsize=9, color=colour, va="bottom")
    ax.set_yticks(range(len(estimates)), [e.name for e in reversed(estimates)])
    ax.axvline(0, color="#cbd5e1", lw=1)
    if truth is not None:
        ax.axvline(truth, color=ROSE, ls="--", lw=1.5, label=f"true ATE = {truth:.2f}")
        ax.legend(frameon=False, loc="lower right")
    ax.set_xlabel("Estimated average treatment effect (95% CI)")
    return _clean(ax).figure


def overlap(e, t):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.linspace(0, 1, 41)
    ax.hist(e[t == 0], bins, alpha=.65, color=SLATE, label="control", density=True)
    ax.hist(e[t == 1], bins, alpha=.65, color=TEAL, label="treated", density=True)
    ax.set_xlabel("Propensity score P(treated | X)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    return _clean(ax).figure


def love_plot(before, after):
    order = before.sort_values().index
    fig, ax = plt.subplots(figsize=(6.5, 0.6 + 0.55 * len(order)))
    y = np.arange(len(order))
    ax.scatter(before[order], y, color=ROSE, label="unweighted", zorder=3)
    ax.scatter(after[order], y, color=TEAL, label="IPW-weighted", zorder=3)
    for yi, c in zip(y, order):
        ax.plot([after[c], before[c]], [yi, yi], color="#cbd5e1", zorder=1)
    ax.axvline(0.1, color=AMBER, ls="--", lw=1)
    ax.text(0.1, len(order) - 0.4, " 0.1 threshold", color=AMBER, fontsize=8)
    ax.set_yticks(y, order)
    ax.set_xlabel("|Standardised mean difference|")
    ax.legend(frameon=False, loc="lower right")
    return _clean(ax).figure


def cate_scatter(true, pred, title: str):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(true, pred, s=8, alpha=.35, color=TEAL)
    lim = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lim, lim, color=ROSE, ls="--", lw=1.2)
    ax.set_xlabel("True individual effect")
    ax.set_ylabel("Estimated effect")
    ax.set_title(title, fontsize=10)
    return _clean(ax).figure


def cate_by_group(frame, group: str, cate_col: str = "cate"):
    g = frame.groupby(group, observed=True)[cate_col].agg(["mean", "std", "count"])
    se = g["std"] / np.sqrt(g["count"])
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    pos = np.arange(len(g))
    ax.barh(pos, g["mean"], xerr=1.96 * se, color=TEAL, capsize=4, alpha=.9)
    ax.set_yticks(pos, [str(i) for i in g.index])
    ax.invert_yaxis()
    ax.axvline(frame[cate_col].mean(), color=ROSE, ls="--", lw=1, label="overall mean")
    ax.set_xlabel("Mean estimated effect")
    ax.set_ylabel(group)
    ax.legend(frameon=False, loc="lower right")
    return _clean(ax).figure


def bias_curve(strengths, estimates: dict, truth: float):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for name, vals in estimates.items():
        ax.plot(strengths, vals, marker="o", lw=2, label=name,
                color=SLATE if name.startswith("Naive") else TEAL if name.startswith("AIPW") else AMBER)
    ax.axhline(truth, color=ROSE, ls="--", lw=1.5, label="true ATE")
    ax.set_xlabel("Strength of unobserved confounding")
    ax.set_ylabel("Estimated ATE")
    ax.legend(frameon=False, fontsize=8)
    return _clean(ax).figure
