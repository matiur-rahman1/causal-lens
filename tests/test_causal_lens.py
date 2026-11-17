import numpy as np
import pandas as pd
import pytest

from causal_lens import estimate_all, load_csv, simulate
from causal_lens.diagnostics import (e_value, e_value_for_effect, effective_sample_size, ipw_weights,
                                     standardized_mean_differences)
from causal_lens.estimators import t_learner, x_learner


@pytest.fixture(scope="module")
def data():
    return simulate(n=3000, effect=5.0, confounding=1.0, seed=7)


@pytest.fixture(scope="module")
def results(data):
    return {e.name: e for e in estimate_all(data, B=60, seed=0)}


def test_simulation_is_reproducible():
    a, b = simulate(500, seed=3), simulate(500, seed=3)
    assert np.allclose(a.y, b.y) and np.array_equal(a.t, b.t)


def test_naive_estimate_is_badly_biased(data, results):
    assert results["Naive difference"].ate - data.true_ate > 3.0


@pytest.mark.parametrize("name", ["Regression adjustment", "IPW", "AIPW (doubly robust)"])
def test_adjusted_estimators_recover_truth(name, data, results):
    assert abs(results[name].ate - data.true_ate) < 0.9


@pytest.mark.parametrize("name", ["Regression adjustment", "IPW", "AIPW (doubly robust)"])
def test_confidence_interval_is_ordered_and_covers_truth(name, data, results):
    e = results[name]
    assert e.low < e.ate < e.high
    assert e.low - 0.3 <= data.true_ate <= e.high + 0.3


def test_ipw_improves_covariate_balance(data):
    _, w = ipw_weights(data)
    before = standardized_mean_differences(data)
    after = standardized_mean_differences(data, w)
    assert (after < before).all()
    assert after.max() < 0.1


def test_effective_sample_size_bounds(data):
    _, w = ipw_weights(data)
    ess = effective_sample_size(w)
    assert 0 < ess <= data.n
    assert effective_sample_size(np.ones(10)) == pytest.approx(10)


@pytest.mark.parametrize("learner", [t_learner, x_learner])
def test_cate_learners_beat_a_constant_guess(learner, data):
    cate = learner(data)
    rmse = np.sqrt(np.mean((cate - data.true_ite) ** 2))
    baseline = np.sqrt(np.mean((data.true_ite - data.true_ite.mean()) ** 2))
    assert rmse < 1.5 * baseline + 0.5
    assert np.corrcoef(cate, data.true_ite)[0, 1] > 0.5


def test_hidden_confounding_biases_adjusted_estimators():
    clean = simulate(3000, hidden_confounding=0.0, seed=5)
    dirty = simulate(3000, hidden_confounding=1.5, seed=5)
    e_clean = {e.name: e for e in estimate_all(clean, B=20)}["IPW"]
    e_dirty = {e.name: e for e in estimate_all(dirty, B=20)}["IPW"]
    assert abs(e_dirty.ate - dirty.true_ate) > abs(e_clean.ate - clean.true_ate) + 0.5


def test_e_value_known_values():
    assert e_value(1.0) == pytest.approx(1.0)
    assert e_value(2.0) == pytest.approx(2 + np.sqrt(2))      # VanderWeele & Ding (2017)
    assert e_value(0.5) == pytest.approx(e_value(2.0))        # symmetric for protective effects
    point, bound = e_value_for_effect(2.0, -0.5, 4.5, outcome_sd=5)
    assert point > 1 and bound == 1.0                         # CI includes zero -> E-value 1


def test_load_csv_validates_input():
    df = pd.DataFrame({"t": np.r_[np.zeros(60), np.ones(60)], "y": np.arange(120.0),
                       "a": np.arange(120.0), "grp": ["u", "v"] * 60})
    ok = load_csv(df, "t", "y", ["a", "grp"])
    assert ok.n == 120 and "grp_v" in ok.X.columns
    with pytest.raises(ValueError, match="binary"):
        load_csv(df.assign(t=np.arange(120)), "t", "y", ["a"])
    with pytest.raises(ValueError, match="covariate"):
        load_csv(df, "t", "y", [])
    with pytest.raises(ValueError, match="at least 50"):
        load_csv(df.head(20), "t", "y", ["a"])
