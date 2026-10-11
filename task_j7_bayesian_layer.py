"""
Task J7 (future-work #7 / locked-plan 8K-8L): Bayesian logistic layer.

Kula et al. used Bayesian modelling to get calibrated, uncertainty-aware
predictions and to test whether delay patterns add information. Two Bayesian
models are fitted with PyMC (NUTS), each WITH and WITHOUT the pattern feature:

  B1  hierarchical Bayesian logistic regression on interpretable features
      (activity, elapsed, stall, assignee history, priority/type/status) with
      partially-pooled PROJECT intercepts  -> coefficient credible intervals,
      per-project effects, pattern effect with HDI.
  B2  Bayesian calibration layer ON TOP OF XGBoost v4: logit(p_xgb) (+ pattern)
      fitted on VALIDATION rows using an XGBoost model trained on TRAIN only
      (honest out-of-sample scores), with project-level partial pooling
      -> calibrated probabilities + per-issue credible intervals.

Fitting uses a 30,000-row stratified subsample (NUTS, 2 chains x 800 draws);
evaluation uses ALL test rows (posterior mean over 200 draws).
Model comparison: PSIS-LOO on the fitting sample + test AUC / Brier / ECE.
"""

import json
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss

from h_common import LABEL, REPORTS_DIR, seeded
from task_i2_model_upgrades import load, BASE, ACT

import os
N_FIT = int(os.environ.get("J7_N_FIT", 30000))
DRAWS = TUNE = int(os.environ.get("J7_DRAWS", 800))
CHAINS = 2
OUT_TAG = os.environ.get("J7_TAG", "")
RNG = np.random.default_rng(0)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def design(df, stats=None, with_pattern=False, patterns=None):
    num = {
        "log_min_since_last_change": df["log_min_since_last_change"],
        "log_min_since_last_activity": df["log_min_since_last_activity"],
        "log_min_since_last_comment": df["log_min_since_last_comment"],
        "ever_in_review": df["ever_in_review"],
        "ever_in_progress": df["ever_in_progress"],
        "log_n_transitions": np.log1p(df["n_transitions"]),
        "log_comments_7d": np.log1p(df["comments_last_7d"]),
        "assignee_prior_delay_rate": df["assignee_prior_delay_rate"].fillna(0.5),
        "log_story_point": np.log1p(df["Story_Point"].fillna(0)),
        "log_elapsed": np.log1p(df["elapsed_minutes"]),
        "log1p_stall": df["log1p_stall"].fillna(0),
        "checkpoint_index": df["checkpoint_index"],
        "is_unassigned": df["is_unassigned"],
        "log_n_links": np.log1p(df["n_links"]),
        "priority_num": pd.to_numeric(df["Priority_Normalized"].astype(str), errors="coerce").fillna(3),
        "priority_unknown": (df["Priority_Normalized"].astype(str) == "Unknown").astype(int),
    }
    X = pd.DataFrame(num)
    for t in ["Bug", "Improvement", "Task", "New Feature", "Story", "Sub-task"]:
        X[f"type_{t}"] = (df["Type_Normalized"].astype(str) == t).astype(int)
    for b in ["review", "in_progress", "testing", "waiting", "done", "reopened"]:
        X[f"status_{b}"] = (df["status_bucket"].astype(str) == b).astype(int)
    if with_pattern:
        for p in patterns:
            X[f"pattern_{p}"] = (df["pattern_label"].astype(str) == p).astype(int)
    X = X.astype(float)
    if stats is None:
        stats = (X.mean(), X.std().replace(0, 1))
    return ((X - stats[0]) / stats[1]).to_numpy(), list(X.columns), stats


def fit_b1(Xf, yf, proj_idx, n_proj, names):
    with pm.Model(coords={"feature": names}) as model:
        mu_a = pm.Normal("mu_a", 0, 1.5)
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        z = pm.Normal("z_proj", 0, 1, shape=n_proj)
        a = pm.Deterministic("a_proj", mu_a + sigma_a * z)
        beta = pm.Normal("beta", 0, 1, dims="feature")
        eta = a[proj_idx] + pm.math.dot(Xf, beta)
        pm.Bernoulli("y", logit_p=eta, observed=yf)
        idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, cores=2, target_accept=0.9, random_seed=0,
                          progressbar=False, idata_kwargs={"log_likelihood": True})
    return idata


def fit_b2(Lf, Pf, yf, proj_idx, n_proj, pat_names):
    with pm.Model(coords={"pattern": pat_names}) as model:
        mu_a = pm.Normal("mu_a", 0, 1.5)
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        z = pm.Normal("z_proj", 0, 1, shape=n_proj)
        a = pm.Deterministic("a_proj", mu_a + sigma_a * z)
        slope = pm.Normal("slope", 1, 1)
        eta = a[proj_idx] + slope * Lf
        if Pf is not None:
            gamma = pm.Normal("gamma", 0, 1, dims="pattern")
            eta = eta + pm.math.dot(Pf, gamma)
        pm.Bernoulli("y", logit_p=eta, observed=yf)
        idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, cores=2, target_accept=0.9, random_seed=0,
                          progressbar=False, idata_kwargs={"log_likelihood": True})
    return idata


def posterior_probs(idata, eta_fn, n_draws=200):
    post = idata.posterior
    if hasattr(post, "to_dataset"):
        post = post.to_dataset()
    post = post.stack(s=("chain", "draw"))
    sel = RNG.choice(post.sizes["s"], size=min(n_draws, post.sizes["s"]), replace=False)
    P = []
    for i in sel:
        P.append(1 / (1 + np.exp(-eta_fn(post.isel(s=i)))))
    P = np.asarray(P, dtype=np.float32)
    return P.mean(0), np.quantile(P, 0.05, axis=0), np.quantile(P, 0.95, axis=0)


def main():
    ck = load()
    proj_codes = {p: i for i, p in enumerate(sorted(ck["Project_ID"].astype(int).unique()))}
    ck["proj_idx"] = ck["Project_ID"].astype(int).map(proj_codes)
    patterns = sorted(ck["pattern_label"].astype(str).unique())
    patterns = [p for p in patterns if p != "insufficient_history"]
    train, val, tv = ck[ck.split == "train"], ck[ck.split == "val"], ck[ck.split.isin(["train", "val"])]
    test = ck[ck.split == "test"].reset_index(drop=True)
    y_te = test[LABEL].to_numpy()
    fit = tv.groupby("checkpoint_index").sample(n=N_FIT // 10, random_state=0).reset_index(drop=True)
    results, out = {}, {}

    # ---------------- B1: hierarchical interpretable model ----------------
    for wp in [False, True]:
        tag = "B1_with_pattern" if wp else "B1_no_pattern"
        Xf, names, stats = design(fit, with_pattern=wp, patterns=patterns)
        print(f"\n=== {tag}: {Xf.shape} ===", flush=True)
        idata = fit_b1(Xf, fit[LABEL].to_numpy(), fit["proj_idx"].to_numpy(), len(proj_codes), names)
        Xt, _, _ = design(test, stats=stats, with_pattern=wp, patterns=patterns)
        pi = test["proj_idx"].to_numpy()
        mean, lo, hi = posterior_probs(idata, lambda d: d["a_proj"].values[pi] + Xt @ d["beta"].values)
        summ = az.summary(idata, var_names=["beta", "mu_a", "sigma_a"], ci_prob=0.94)
        summ.to_csv(f"{REPORTS_DIR}/j7_{tag}_posterior.csv")
        results[tag] = {"test_auc": roc_auc_score(y_te, mean), "test_brier": brier_score_loss(y_te, mean),
                        "test_ece": ece(y_te, mean), "mean_90pct_interval_width": float(np.mean(hi - lo)),
                        "max_rhat": float(summ["r_hat"].max()), "min_ess_bulk": float(summ["ess_bulk"].min())}
        out[tag] = (idata, mean)
        print(json.dumps(results[tag], indent=2), flush=True)
        print(summ.round(3).to_string(), flush=True)

    # ---------------- B2: Bayesian calibration layer on XGBoost v4 ----------------
    cols = BASE + ACT
    cfg = dict(learning_rate=0.05, max_depth=8, n_estimators=32)
    pv, pt = [], []
    for s in [0, 1, 2]:
        m = xgb.XGBClassifier(**cfg, **seeded(s))
        m.fit(train[cols], train[LABEL])
        pv.append(m.predict_proba(val[cols])[:, 1])
        pt.append(m.predict_proba(test[cols])[:, 1])
    pv, pt = np.mean(pv, 0), np.mean(pt, 0)
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    vf = val.assign(xgb_logit=lg(pv)).groupby("checkpoint_index").sample(
        n=N_FIT // 10, random_state=0).reset_index(drop=True)
    pat_v = np.column_stack([(vf["pattern_label"].astype(str) == p).astype(float) for p in patterns])
    pat_t = np.column_stack([(test["pattern_label"].astype(str) == p).astype(float) for p in patterns])
    results["XGB_v4_train_only_raw"] = {"test_auc": roc_auc_score(y_te, pt), "test_brier": brier_score_loss(y_te, pt),
                                        "test_ece": ece(y_te, pt)}
    for wp in [False, True]:
        tag = "B2_xgb_calib_with_pattern" if wp else "B2_xgb_calib_no_pattern"
        print(f"\n=== {tag} ===", flush=True)
        idata = fit_b2(vf["xgb_logit"].to_numpy(), pat_v if wp else None, vf[LABEL].to_numpy(),
                       vf["proj_idx"].to_numpy(), len(proj_codes), patterns)
        Lt, pi = lg(pt), test["proj_idx"].to_numpy()
        if wp:
            fn = lambda d: d["a_proj"].values[pi] + d["slope"].values * Lt + pat_t @ d["gamma"].values
        else:
            fn = lambda d: d["a_proj"].values[pi] + d["slope"].values * Lt
        mean, lo, hi = posterior_probs(idata, fn)
        vn = ["slope", "mu_a", "sigma_a"] + (["gamma"] if wp else [])
        summ = az.summary(idata, var_names=vn, ci_prob=0.94)
        summ.to_csv(f"{REPORTS_DIR}/j7_{tag}_posterior.csv")
        results[tag] = {"test_auc": roc_auc_score(y_te, mean), "test_brier": brier_score_loss(y_te, mean),
                        "test_ece": ece(y_te, mean), "mean_90pct_interval_width": float(np.mean(hi - lo)),
                        "max_rhat": float(summ["r_hat"].max())}
        out[tag] = (idata, mean)
        print(json.dumps(results[tag], indent=2), flush=True)
        print(summ.round(3).to_string(), flush=True)
        if not wp:
            np.savez_compressed(f"{REPORTS_DIR}/j7_b2_test_intervals.npz", mean=mean, lo=lo, hi=hi, y=y_te,
                                checkpoint=test["checkpoint_index"].to_numpy())

    # ---------------- LOO: does the pattern feature add information? ----------------
    for a, b in [("B1_no_pattern", "B1_with_pattern"), ("B2_xgb_calib_no_pattern", "B2_xgb_calib_with_pattern")]:
        try:
            cmp = az.compare({a: out[a][0], b: out[b][0]}, ic="loo")
        except TypeError:
            cmp = az.compare({a: out[a][0], b: out[b][0]})
        cmp.to_csv(f"{REPORTS_DIR}/j7_loo_{a.split('_')[0]}.csv")
        print(f"\nPSIS-LOO comparison {a} vs {b}:\n{cmp.round(2).to_string()}", flush=True)
        results[f"loo_{a.split('_')[0]}"] = cmp.round(3).astype(str).to_dict()

    json.dump(results, open(f"{REPORTS_DIR}/j7{OUT_TAG}_summary.json", "w"), indent=2, default=float)
    print("\nSUMMARY:\n" + pd.DataFrame({k: v for k, v in results.items() if not k.startswith("loo")}).T.round(4).to_string())


if __name__ == "__main__":
    main()
