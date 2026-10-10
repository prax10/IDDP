"""
Task H1 (future-work #1): individually tune ALL 10 per-checkpoint models.

B5 tuned only M1, M5 and M10 before declaring pooled and per-checkpoint
"equivalent" (max gap 0.0098). This closes the gap for M2-M4 and M6-M9,
and re-runs M1/M5/M10 under the same procedure.

Selection procedure (identical for every model so the race is fair):
  - 9-config grid (lr in {0.05, 0.1, 0.2} x depth in {4, 6, 8}),
    early stopping (30 rounds) on that model's own validation rows
  - each config scored by the MEAN val AUC over 3 training seeds
    (single-seed selection is noisy at this data size)
  - n_estimators = max(median best round across seeds, 10)
  - refit on train+val, 5 seeds, score on test
The pooled model is re-selected with the same multi-seed procedure, so
neither side gets a better-tuned treatment than the other.

Comparisons per checkpoint (paired bootstrap, 1,000 test resamples, of
seed-mean AUC difference):
  A = individually-tuned per-checkpoint   vs   B = pooled (tuned)
Equivalence = 95% CI fully inside +/-0.01 AUC.
Also reports the tuned static model on the same rows (crossover check).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from h_common import (REPORTS_DIR, LABEL, GRID, SEEDS, TUNED_STATIC, dyn_cols, static_cols, seeded,
                      load_dynamic, load_static, paired_bootstrap_seeds, seed_metric_panel, verdict)

COLS = dyn_cols(30)
SEL_SEEDS = [0, 1, 2]


def select(train, val, cols, tag):
    rows = []
    for cfg in GRID:
        aucs, iters = [], []
        for s in SEL_SEEDS:
            m = xgb.XGBClassifier(n_estimators=1000, early_stopping_rounds=30, **cfg, **seeded(s))
            m.fit(train[cols], train[LABEL], eval_set=[(val[cols], val[LABEL])], verbose=False)
            aucs.append(roc_auc_score(val[LABEL], m.predict_proba(val[cols])[:, 1]))
            iters.append(m.best_iteration + 1)
        rows.append({**cfg, "val_auc_mean": np.mean(aucs), "val_auc_sd": np.std(aucs, ddof=1),
                     "trees_median": int(np.median(iters))})
    g = pd.DataFrame(rows).sort_values("val_auc_mean", ascending=False).reset_index(drop=True)
    b = g.iloc[0]
    cfg = {"learning_rate": float(b["learning_rate"]), "max_depth": int(b["max_depth"]),
           "n_estimators": max(int(b["trees_median"]), 10)}
    print(f"  [{tag}] selected {cfg}  val_AUC={b['val_auc_mean']:.4f} (+/-{b['val_auc_sd']:.4f})", flush=True)
    g.insert(0, "model", tag)
    return cfg, g


def fit_seeds(cfg, data, cols, X_test):
    out = []
    for s in SEEDS:
        m = xgb.XGBClassifier(**cfg, **seeded(s))
        m.fit(data[cols], data[LABEL])
        out.append(m.predict_proba(X_test)[:, 1])
    return out


def main():
    ck = load_dynamic()
    train = ck[ck["split"] == "train"]
    val = ck[ck["split"] == "val"]
    trainval = ck[ck["split"].isin(["train", "val"])]
    test = ck[ck["split"] == "test"].reset_index(drop=True)
    y = test[LABEL].to_numpy()
    ck_idx = test["checkpoint_index"].to_numpy()

    grids = []
    print("=== pooled: multi-seed selection ===")
    pooled_cfg, g = select(train, val, COLS, "pooled")
    grids.append(g)
    P_pooled = fit_seeds(pooled_cfg, trainval, COLS, test[COLS])

    print("=== static (tuned config from B4) on the same checkpoint rows ===")
    st = load_static()
    st_tv = st[st["split"].isin(["train", "val"])]
    P_static = fit_seeds(TUNED_STATIC, st_tv, static_cols(30), test[static_cols(30)])

    print("=== per-checkpoint: individual multi-seed selection ===")
    cfgs, P_perck = {}, {}
    for t in range(1, 11):
        tr_t = train[train["checkpoint_index"] == t]
        va_t = val[val["checkpoint_index"] == t]
        tv_t = trainval[trainval["checkpoint_index"] == t]
        cfgs[t], g = select(tr_t, va_t, COLS, f"M{t}")
        grids.append(g)
        m = ck_idx == t
        P_perck[t] = fit_seeds(cfgs[t], tv_t, COLS, test.loc[m, COLS])
    pd.concat(grids).to_csv(f"{REPORTS_DIR}/h1_tuning_grids.csv", index=False)

    # ---- per-checkpoint table
    rows, panel = [], []
    for t in range(1, 11):
        m = ck_idx == t
        yt = y[m]
        pp = [p[m] for p in P_pooled]
        ps = [p[m] for p in P_static]
        pc = P_perck[t]
        bs = paired_bootstrap_seeds(yt, pc, pp, n_boot=1000, seed=t)
        bs_static = paired_bootstrap_seeds(yt, pp, ps, n_boot=1000, seed=100 + t)
        a_pc = [roc_auc_score(yt, p) for p in pc]
        a_pp = [roc_auc_score(yt, p) for p in pp]
        a_st = [roc_auc_score(yt, p) for p in ps]
        rows.append({
            "checkpoint_index": t, "N": int(m.sum()), "pct_delayed": yt.mean(),
            "perck_lr": cfgs[t]["learning_rate"], "perck_depth": cfgs[t]["max_depth"],
            "perck_trees": cfgs[t]["n_estimators"],
            "static_auc": np.mean(a_st), "pooled_auc": np.mean(a_pp), "pooled_auc_sd": np.std(a_pp, ddof=1),
            "perck_auc": np.mean(a_pc), "perck_auc_sd": np.std(a_pc, ddof=1),
            "gap_perck_minus_pooled": bs["d_auc"], "gap_lo": bs["d_auc_lo"], "gap_hi": bs["d_auc_hi"],
            "verdict_vs_pooled": verdict(bs["d_auc_lo"], bs["d_auc_hi"]).replace("A ", "per-ck ").replace("B ", "pooled "),
            "acc_gap_perck_minus_pooled": bs["d_acc"],
            "pooled_minus_static": bs_static["d_auc"], "pooled_minus_static_lo": bs_static["d_auc_lo"],
            "pooled_minus_static_hi": bs_static["d_auc_hi"],
        })
        for name, P in [("static", ps), ("pooled", pp), ("per_checkpoint", pc)]:
            panel.append(seed_metric_panel(yt, P, model=name, checkpoint=f"M{t}"))
        r = rows[-1]
        print(f"  M{t}: static={r['static_auc']:.4f} pooled={r['pooled_auc']:.4f} perck={r['perck_auc']:.4f} "
              f"gap={r['gap_perck_minus_pooled']:+.4f} [{r['gap_lo']:+.4f},{r['gap_hi']:+.4f}] "
              f"{r['verdict_vs_pooled']}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{REPORTS_DIR}/h1_percheckpoint_all10_tuned.csv", index=False)
    pd.DataFrame(panel).to_csv(f"{REPORTS_DIR}/h1_metrics_panel.csv", index=False)

    gaps = res["gap_perck_minus_pooled"]
    n_equiv = res["verdict_vs_pooled"].str.startswith("equivalent").sum()
    print(f"\npooled config: {pooled_cfg}")
    print(f"per-checkpoint beats pooled at {(gaps > 0).sum()}/10 (point estimate); "
          f"mean gap {gaps.mean():+.4f}; max |gap| {gaps.abs().max():.4f}")
    print(f"equivalent (CI inside +/-0.01) at {n_equiv}/10")
    sig = res[~res["verdict_vs_pooled"].str.startswith(("equivalent", "inconclusive"))]
    print("significant differences:", sig[["checkpoint_index", "verdict_vs_pooled"]].to_dict("records") or "none")
    cross = res[res["pooled_minus_static_lo"] > 0]["checkpoint_index"]
    print(f"first checkpoint where pooled > static with CI excluding 0: M{cross.min() if len(cross) else 'none'}")
    plot(res)


def plot(res):
    SURFACE, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = res["checkpoint_index"]
    ax.axhspan(-0.01, 0.01, color="#54A24B", alpha=0.12, label="equivalence margin (+/-0.01)")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.errorbar(x, res["gap_perck_minus_pooled"],
                yerr=[res["gap_perck_minus_pooled"] - res["gap_lo"], res["gap_hi"] - res["gap_perck_minus_pooled"]],
                fmt="o", color="#eb6834", capsize=4, linewidth=1.8, label="per-checkpoint - pooled (95% CI)")
    ax.set_xticks(list(x))
    ax.set_xlabel("Checkpoint", color=INK2)
    ax.set_ylabel("AUC difference", color=INK2)
    ax.set_title("All 10 checkpoints individually tuned: per-checkpoint vs pooled", color=INK)
    ax.grid(color=GRIDC, linewidth=0.8)
    for s in ax.spines.values():
        s.set_color(GRIDC)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=GRIDC)
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/h1_percheckpoint_vs_pooled.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
