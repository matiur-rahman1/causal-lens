"""Causal Lens - interactive causal-inference explorer (Streamlit)."""
import numpy as np
import pandas as pd
import streamlit as st

from causal_lens import estimate_all, load_csv, simulate
from causal_lens import plots
from causal_lens.diagnostics import (e_value_for_effect, effective_sample_size, ipw_weights,
                                     standardized_mean_differences)
from causal_lens.estimators import t_learner, x_learner

st.set_page_config(page_title="Causal Lens", page_icon="🧭", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1200px}
.card{border-radius:14px;padding:16px 20px;border:1px solid #e2e8f0;background:#fff}
.card h3{margin:0;font-size:2rem}
.card p{margin:2px 0 0;color:#64748b;font-size:.85rem}
.bad h3{color:#e11d48}.good h3{color:#0d9488}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Simulating...")
def get_sim(n, effect, het, conf, hidden, seed):
    return simulate(n, effect, het, conf, hidden, seed)


@st.cache_data(show_spinner="Estimating effects (bootstrap + cross-fitting)...")
def run_estimates(_data, key):
    return estimate_all(_data, B=200)


@st.cache_data(show_spinner=False)
def run_cates(_data, key):
    return t_learner(_data), x_learner(_data)


st.sidebar.title("🧭 Causal Lens")
source = st.sidebar.radio("Data", ["Simulated study", "Upload CSV"])

if source == "Simulated study":
    st.sidebar.caption("Job-training programme with a known true effect.")
    n = st.sidebar.slider("Sample size", 500, 6000, 3000, 250)
    effect = st.sidebar.slider("True average effect", 0.0, 10.0, 5.0, 0.5)
    het = st.sidebar.slider("Effect heterogeneity", 0.0, 3.0, 1.0, 0.25)
    conf = st.sidebar.slider("Confounding strength", 0.0, 2.0, 1.0, 0.25)
    hidden = st.sidebar.slider("Hidden confounding", 0.0, 2.0, 0.0, 0.25,
                               help="An unobserved 'motivation' variable. No method can adjust for it.")
    seed = st.sidebar.number_input("Random seed", 0, 9999, 0)
    data = get_sim(n, effect, het, conf, hidden, int(seed))
    key = ("sim", n, effect, het, conf, hidden, int(seed))
else:
    up = st.sidebar.file_uploader("CSV file", type="csv")
    if up is None:
        st.title("Causal Lens")
        st.info("Upload a CSV with a binary treatment column, a numeric outcome and some covariates. "
                "A sample file is in the repository: `data/sample_observational.csv`.")
        st.stop()
    raw = pd.read_csv(up)
    cols = list(raw.columns)
    tcol = st.sidebar.selectbox("Treatment (0/1)", cols)
    ycol = st.sidebar.selectbox("Outcome", [c for c in cols if c != tcol])
    xcols = st.sidebar.multiselect("Covariates", [c for c in cols if c not in (tcol, ycol)])
    try:
        data = load_csv(raw, tcol, ycol, xcols)
    except ValueError as err:
        st.title("Causal Lens")
        st.warning(str(err))
        st.stop()
    key = ("csv", up.name, up.size, tcol, ycol, tuple(xcols))

truth = data.true_ate
est = run_estimates(data, key)
by_name = {e.name: e for e in est}

# ------------------------------------------------------------------ header
st.title("Does the treatment really work?")
st.caption("Observational data hides confounding. Compare naive and adjusted estimators, check the assumptions, "
           "and see how fragile the conclusion is.")

tabs = st.tabs(["Estimates", "Diagnostics", "Who benefits?", "Sensitivity", "About"])

# ------------------------------------------------------------------ estimates
with tabs[0]:
    naive, best = by_name["Naive difference"], by_name["AIPW (doubly robust)"]
    c = st.columns(4 if truth is not None else 3)
    c[0].markdown(f'<div class="card bad"><h3>{naive.ate:+.2f}</h3><p>Naive difference in means</p></div>', unsafe_allow_html=True)
    c[1].markdown(f'<div class="card good"><h3>{best.ate:+.2f}</h3><p>Doubly-robust (AIPW) estimate<br>95% CI [{best.low:.2f}, {best.high:.2f}]</p></div>', unsafe_allow_html=True)
    c[2].markdown(f'<div class="card"><h3>{data.t.mean():.0%}</h3><p>treated · n = {data.n:,}</p></div>', unsafe_allow_html=True)
    if truth is not None:
        c[3].markdown(f'<div class="card"><h3>{truth:+.2f}</h3><p>True effect (known in simulation)</p></div>', unsafe_allow_html=True)
    st.write("")
    st.pyplot(plots.forest(est, truth), clear_figure=True)
    table = pd.DataFrame([{"Method": e.name, "ATE": round(e.ate, 3), "CI low": round(e.low, 3),
                           "CI high": round(e.high, 3),
                           **({"Error vs truth": round(e.ate - truth, 3)} if truth is not None else {}),
                           "Notes": e.note} for e in est])
    st.dataframe(table, hide_index=True, use_container_width=True)
    if truth is not None:
        st.success(f"The naive comparison is off by **{naive.ate - truth:+.2f}**; AIPW is off by "
                   f"**{best.ate - truth:+.2f}**. Raise *Hidden confounding* in the sidebar to see adjustment fail.")

# ------------------------------------------------------------------ diagnostics
with tabs[1]:
    e, w = ipw_weights(data)
    smd0, smd1 = standardized_mean_differences(data), standardized_mean_differences(data, w)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Overlap")
        st.caption("Treated and control units must both exist at every propensity level (positivity).")
        st.pyplot(plots.overlap(e, data.t), clear_figure=True)
    with c2:
        st.subheader("Covariate balance")
        st.caption("Weighting should push every dot below the 0.1 line.")
        st.pyplot(plots.love_plot(smd0, smd1), clear_figure=True)
    ess = effective_sample_size(w)
    m = st.columns(3)
    m[0].metric("Max |SMD| before", f"{smd0.max():.2f}")
    m[1].metric("Max |SMD| after weighting", f"{smd1.max():.2f}")
    m[2].metric("Effective sample size", f"{ess:,.0f}", f"{ess / data.n:.0%} of n", delta_color="off")
    if (e < .05).mean() + (e > .95).mean() > .05:
        st.warning("More than 5% of units have extreme propensity scores: weak overlap, estimates may be unstable.")

# ------------------------------------------------------------------ heterogeneity
with tabs[2]:
    t_cate, x_cate = run_cates(data, key)
    learner = st.radio("Learner", ["T-learner", "X-learner"], horizontal=True)
    cate = t_cate if learner == "T-learner" else x_cate
    frame = data.X.copy()
    frame["cate"] = cate
    c1, c2 = st.columns(2)
    with c1:
        if data.true_ite is not None:
            rmse = float(np.sqrt(np.mean((cate - data.true_ite) ** 2)))
            st.pyplot(plots.cate_scatter(data.true_ite, cate, f"{learner}: RMSE {rmse:.2f}"), clear_figure=True)
        else:
            st.metric("Mean estimated effect", f"{cate.mean():.2f}")
            st.metric("Spread (std)", f"{cate.std():.2f}")
    with c2:
        cols_ = list(data.X.columns)
        feat = st.selectbox("Break down by", cols_, index=cols_.index("health_score") if "health_score" in cols_ else 0)
        col = frame[feat]
        if col.nunique() > 6:
            bins = pd.qcut(col, 4, duplicates="drop")
            labels = [f"{iv.left:.3g} to {iv.right:.3g}" for iv in bins.cat.categories]
            frame["group"] = pd.Categorical.from_codes(bins.cat.codes, labels, ordered=True)
        else:
            order = sorted(col.unique())
            frame["group"] = pd.Categorical(col.astype(str), [str(o) for o in order], ordered=True)
        st.pyplot(plots.cate_by_group(frame, "group"), clear_figure=True)
    st.caption("Individual effects are estimates, not measurements. Treat subgroup differences as hypotheses.")

# ------------------------------------------------------------------ sensitivity
with tabs[3]:
    sd = float(data.y.std())
    point, bound = e_value_for_effect(best.ate, best.low, best.high, sd)
    c = st.columns(2)
    c[0].metric("E-value (point estimate)", f"{point:.2f}")
    c[1].metric("E-value (CI bound)", f"{bound:.2f}")
    st.write(f"An unmeasured confounder would need to be associated with **both** treatment and outcome by a risk ratio of "
             f"about **{point:.2f}** each to fully explain away the AIPW estimate"
             + (f", and **{bound:.2f}** to move the confidence interval to include zero." if bound > 1
                else ". The confidence interval already includes zero."))
    st.caption("E-value (VanderWeele & Ding, 2017) with the approximation RR ≈ exp(0.91·d), d = effect / SD(outcome).")
    if source == "Simulated study":
        st.subheader("What happens as hidden confounding grows?")
        if st.button("Run sweep"):
            grid = [0.0, 0.5, 1.0, 1.5, 2.0]
            rows = {"Naive difference": [], "Regression adjustment": [], "AIPW (doubly robust)": []}
            bar = st.progress(0.0)
            for i, h in enumerate(grid):
                d = simulate(2000, effect, het, conf, h, int(seed))
                r = {x.name: x.ate for x in estimate_all(d, B=10)}
                for k in rows:
                    rows[k].append(r[k])
                bar.progress((i + 1) / len(grid))
            st.pyplot(plots.bias_curve(grid, rows, truth), clear_figure=True)
            st.caption("Adjustment removes *observed* confounding only. The gap that opens up is the price of what you did not measure.")

# ------------------------------------------------------------------ about
with tabs[4]:
    st.markdown("""
**Causal Lens** estimates the average treatment effect (ATE) of a binary treatment from observational data.

| Method | Idea | Weakness |
|---|---|---|
| Naive difference | Compare group means | Biased by confounding |
| Regression adjustment | Linear model with covariates | Wrong if outcome is non-linear |
| IPW | Reweight to mimic a randomised trial | Unstable with extreme weights |
| AIPW | Outcome model **and** propensity model; right if either is right | Needs overlap |

**Assumptions:** no unmeasured confounding, overlap (positivity), consistency. None of these can be proven from data, which is why the *Diagnostics* and *Sensitivity* tabs exist.

Educational tool. Not a substitute for a randomised experiment or domain expertise.
""")
