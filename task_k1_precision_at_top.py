"""
Task K1: "of the issues we flag as highest-risk, how many really end up late?"

Precision@top-k% is the operational number for a project manager who can only
chase a handful of issues. Reported for:
  FLOW mode (v5b, delay > project median), test rows, seed-averaged ensemble:
    - top 5/10/20% of all rows, overall and per checkpoint
    - the same on UNDETERMINED rows only (not yet past the project median, so the
      label is still open -- the honest version)
    - per project x checkpoint: flag the top 10% within each group (what a PM
      of one project sees), precision averaged over groups
  SCRUM mode (spillover model), test rows:
    - top 10/20/30% overall and per sprint-progress milestone
    - sprint-level: expected spill count (sum of risks) vs actual, MAE and
      correlation; "will >= half of the open committed issues spill" AUC
Base rate (precision of flagging at random) is printed next to every figure.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPORTS = "reports"


def prec_at(y, p, frac):
    k = max(int(round(len(y) * frac)), 1)
    idx = np.argsort(-p, kind="mergesort")[:k]
    return y[idx].mean(), y[idx].sum() / max(y.sum(), 1), k


def flow():
    z = np.load(f"{REPORTS}/j4_test_predictions.npz")
    y, c, und, safe, p = z["y"], z["checkpoint"], z["undet"], z["safe"], z["v5b"]
    ck = pd.read_csv("data/features/phase8_checkpoints.csv", usecols=["Issue_ID", "checkpoint_index", "split", "Project_ID"])
    te = ck[ck.split == "test"].reset_index(drop=True)
    assert (te.checkpoint_index.to_numpy() == c).all()
    proj = te["Project_ID"].to_numpy()
    rows = []
    for subset, m0 in [("all rows", np.ones(len(y), bool)), ("undetermined rows", und), ("safe & undetermined", safe & und)]:
        for scope in ["all"] + [f"M{t}" for t in range(1, 11)]:
            m = m0 if scope == "all" else m0 & (c == int(scope[1:]))
            for frac in (0.05, 0.10, 0.20):
                pr, rc, k = prec_at(y[m], p[m], frac)
                rows.append({"mode": "flow", "subset": subset, "scope": scope, "top": f"{int(frac*100)}%", "k": k,
                             "precision": pr, "recall": rc, "base_rate": y[m].mean(), "lift": pr / y[m].mean()})
    # per project x checkpoint top-10%
    for subset, m0 in [("all rows", np.ones(len(y), bool)), ("undetermined rows", und)]:
        g = []
        for (pj, t), idx in pd.Series(np.arange(len(y))[m0]).groupby([proj[m0], c[m0]]):
            ii = idx.to_numpy()
            if len(ii) < 20:
                continue
            pr, _, _ = prec_at(y[ii], p[ii], 0.10)
            g.append((pr, y[ii].mean(), len(ii)))
        g = np.array(g)
        rows.append({"mode": "flow", "subset": subset, "scope": "per project x checkpoint (mean)", "top": "10%",
                     "k": int(len(g)), "precision": g[:, 0].mean(), "recall": np.nan, "base_rate": g[:, 1].mean(),
                     "lift": g[:, 0].mean() / g[:, 1].mean()})
    return pd.DataFrame(rows)


def scrum():
    meta = json.load(open("models/xgb_scrum_spillover_meta.json"))
    rows = pd.read_csv("data/features/scrum_rows.csv.gz")
    for c in ["Priority", "Type", "status_bucket"]:
        rows[c] = rows[c].astype(str).astype("category")
    rows["Project_cat"] = rows["Project_ID"].astype(int).astype("category")
    te = rows[rows.split == "test"].reset_index(drop=True)
    models = [joblib.load(f"models/xgb_scrum_spillover_seed{s}.joblib") for s in meta["seeds"]]
    p = np.mean([m.predict_proba(te[meta["features"]])[:, 1] for m in models], axis=0)
    y = te["spilled"].to_numpy()
    print(f"  scrum test AUC check: {roc_auc_score(y, p):.4f} (J5 reported 0.746)")
    out = []
    for scope in ["all", 0.0, 0.25, 0.5, 0.75]:
        m = np.ones(len(y), bool) if scope == "all" else te["progress"].to_numpy() == scope
        for frac in (0.10, 0.20, 0.30):
            pr, rc, k = prec_at(y[m], p[m], frac)
            out.append({"mode": "scrum", "subset": "committed open issues", "scope": f"progress {scope}",
                        "top": f"{int(frac*100)}%", "k": k, "precision": pr, "recall": rc,
                        "base_rate": y[m].mean(), "lift": pr / y[m].mean()})
    te = te.assign(risk=p)
    sl = te.groupby(["sprint_key", "progress"]).agg(expected=("risk", "sum"), actual=("spilled", "sum"),
                                                  n_open=("spilled", "size")).reset_index()
    sl = sl[sl.n_open >= 3]
    sprint_rows = []
    for prog, g in sl.groupby("progress"):
        big = (g["actual"] / g["n_open"] >= 0.5).astype(int)
        sprint_rows.append({"progress": prog, "n_sprints": len(g),
                            "spill_count_MAE": (g["expected"] - g["actual"]).abs().mean(),
                            "mean_actual_spills": g["actual"].mean(),
                            "corr_expected_vs_actual": np.corrcoef(g["expected"], g["actual"])[0, 1],
                            "AUC_half_or_more_spill": roc_auc_score(big, g["expected"] / g["n_open"])
                            if big.nunique() == 2 else np.nan})
    return pd.DataFrame(out), pd.DataFrame(sprint_rows)


def main():
    f = flow()
    s, sp = scrum()
    res = pd.concat([f, s], ignore_index=True)
    res.to_csv(f"{REPORTS}/k1_precision_at_top.csv", index=False)
    sp.to_csv(f"{REPORTS}/k1_scrum_sprint_level.csv", index=False)
    pd.set_option("display.width", 200)
    show = res[(res.scope.isin(["all", "M1", "M3", "M5", "M10", "per project x checkpoint (mean)"])) |
               (res["mode"] == "scrum")]
    print(show[["mode", "subset", "scope", "top", "precision", "recall", "base_rate", "lift"]].round(3).to_string(index=False))
    print("\nScrum sprint-level:\n" + sp.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
