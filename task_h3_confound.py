"""
Task H3 (future-work #3): remove the tuned-vs-untuned confound.

Problem: in B4/B5 the untuned pooled model (lr=0.05, depth=6, early-stopped
on val) was fit on TRAIN only, while the tuned one (lr=0.1, depth=8,
11 rounds) was refit on TRAIN+VAL. The +0.13 AUC gap at M10 mixed two
causes: hyperparameters, and 13% more (and the most recent) training data.

Fix: a 2x2 factorial, every cell scored on the identical test rows.
    config  in {untuned, tuned}
    data    in {train, train+val}
Untuned keeps its own early-stopped tree count when refit on train+val,
so the only thing that changes along the data axis is the data.

Every cell is fit with 5 training seeds (XGBoost row/column subsampling).
Single-run AUC moves by up to ~0.02 at M10 between seeds, which is larger
than several of the effects we want to measure, so all point estimates
are seed means and all CIs are paired bootstraps (1,000 test resamples)
of the seed-mean AUC difference.

    hyperparameter effect = tuned - untuned    (data held fixed)
    data effect           = train+val - train  (config held fixed)
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from h_common import (REPORTS_DIR, LABEL, UNTUNED_DYN, TUNED_DYN, SEEDS, dyn_cols, seeded,
                      load_dynamic, paired_bootstrap_seeds, seed_metric_panel, verdict)

COLS = dyn_cols(30)
CELLS = ["untuned|train", "untuned|train+val", "tuned|train", "tuned|train+val"]


def main():
    ck = load_dynamic()
    train = ck[ck["split"] == "train"]
    val = ck[ck["split"] == "val"]
    trainval = ck[ck["split"].isin(["train", "val"])]
    test = ck[ck["split"] == "test"].reset_index(drop=True)
    y_test = test[LABEL].to_numpy()
    ck_idx = test["checkpoint_index"].to_numpy()
    print(f"train={len(train)} val={len(val)} trainval={len(trainval)} test={len(test)}")

    proba = {c: [] for c in CELLS}
    untuned_trees = []
    for s in SEEDS:
        u = xgb.XGBClassifier(n_estimators=400, early_stopping_rounds=20, **UNTUNED_DYN, **seeded(s))
        u.fit(train[COLS], train[LABEL], eval_set=[(val[COLS], val[LABEL])], verbose=False)
        n_u = u.best_iteration + 1
        untuned_trees.append(n_u)
        u_tv = xgb.XGBClassifier(n_estimators=n_u, **UNTUNED_DYN, **seeded(s))
        u_tv.fit(trainval[COLS], trainval[LABEL])
        t_tr = xgb.XGBClassifier(**TUNED_DYN, **seeded(s))
        t_tr.fit(train[COLS], train[LABEL])
        t_tv = xgb.XGBClassifier(**TUNED_DYN, **seeded(s))
        t_tv.fit(trainval[COLS], trainval[LABEL])
        for c, m in zip(CELLS, [u, u_tv, t_tr, t_tv]):
            proba[c].append(m.predict_proba(test[COLS])[:, 1])
        print(f"  seed {s}: untuned early-stopped at {n_u} trees", flush=True)

    # ---- metric panel (seed mean +/- sd)
    rows = []
    for c in CELLS:
        cfg, data = c.split("|")
        rows.append(seed_metric_panel(y_test, proba[c], config=cfg, train_data=data, checkpoint="all"))
        for t in range(1, 11):
            m = ck_idx == t
            rows.append(seed_metric_panel(y_test[m], [p[m] for p in proba[c]],
                                          config=cfg, train_data=data, checkpoint=f"M{t}"))
    panel = pd.DataFrame(rows)
    panel.to_csv(f"{REPORTS_DIR}/h3_confound_factorial_metrics.csv", index=False)
    order = ["all"] + [f"M{t}" for t in range(1, 11)]
    auc_wide = panel.pivot_table(index="checkpoint", columns=["config", "train_data"], values="auc").reindex(order)
    acc_wide = panel.pivot_table(index="checkpoint", columns=["config", "train_data"], values="accuracy").reindex(order)
    print("\nTest AUC (mean of 5 seeds):")
    print(auc_wide.round(4).to_string())
    print("\nTest accuracy @0.5 (mean of 5 seeds):")
    print(acc_wide.round(4).to_string())

    # ---- decomposition
    contrasts = [
        ("hyperparams | data=train", "tuned|train", "untuned|train"),
        ("hyperparams | data=train+val", "tuned|train+val", "untuned|train+val"),
        ("data | config=untuned", "untuned|train+val", "untuned|train"),
        ("data | config=tuned", "tuned|train+val", "tuned|train"),
        ("ORIGINAL confounded gap", "tuned|train+val", "untuned|train"),
    ]
    eff_rows = []
    for scope in order:
        m = np.ones(len(test), bool) if scope == "all" else ck_idx == int(scope[1:])
        nb = 1000
        for label, a, b in contrasts:
            bs = paired_bootstrap_seeds(y_test[m], [p[m] for p in proba[a]], [p[m] for p in proba[b]],
                                        n_boot=nb, seed=0)
            eff_rows.append({"scope": scope, "contrast": label, "A": a, "B": b, "n_boot": nb, **bs,
                             "verdict": verdict(bs["d_auc_lo"], bs["d_auc_hi"])})
        print(f"  bootstrap done: {scope}", flush=True)
    eff = pd.DataFrame(eff_rows)
    eff.to_csv(f"{REPORTS_DIR}/h3_confound_decomposition.csv", index=False)

    print("\nAUC decomposition (A - B, 95% CI) | accuracy diff:")
    for scope in ["all", "M1", "M3", "M5", "M10"]:
        print(f"  [{scope}]")
        for _, r in eff[eff["scope"] == scope].iterrows():
            print(f"    {r['contrast']:30s} {r['d_auc']:+.4f} [{r['d_auc_lo']:+.4f}, {r['d_auc_hi']:+.4f}]"
                  f"  acc {r['d_acc']:+.4f} [{r['d_acc_lo']:+.4f}, {r['d_acc_hi']:+.4f}]  {r['verdict']}")

    def d(scope, label):
        return eff[(eff["scope"] == scope) & (eff["contrast"] == label)]["d_auc"].iloc[0]

    summary = {"untuned_trees_per_seed": untuned_trees}
    for scope in ["all", "M1", "M5", "M10"]:
        total = d(scope, "ORIGINAL confounded gap")
        hp = 0.5 * (d(scope, "hyperparams | data=train") + d(scope, "hyperparams | data=train+val"))
        dt = 0.5 * (d(scope, "data | config=untuned") + d(scope, "data | config=tuned"))
        summary[scope] = {"total_gap": total, "hyperparam_main_effect": hp, "data_main_effect": dt,
                          "interaction": total - hp - dt}
    print("\nMain-effect split of the original gap:")
    print(json.dumps(summary, indent=2, default=float))
    with open(f"{REPORTS_DIR}/h3_confound_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    plot(auc_wide)


def plot(auc_wide):
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    colors = {("untuned", "train"): "#2a78d6", ("untuned", "train+val"): "#2a78d6",
              ("tuned", "train"): "#eb6834", ("tuned", "train+val"): "#eb6834"}
    styles = {"train": "--", "train+val": "-"}
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = list(range(1, 11))
    for (cfg, data), c in colors.items():
        ax.plot(x, auc_wide.loc[[f"M{t}" for t in x], (cfg, data)], color=c, linewidth=2,
                linestyle=styles[data], marker="o" if cfg == "tuned" else "s", markersize=4,
                label=f"{cfg} config, fit on {data}")
    ax.set_xticks(x)
    ax.set_xlabel("Checkpoint", color=INK2)
    ax.set_ylabel("Test AUC (mean of 5 seeds)", color=INK2)
    ax.set_title("Same colour = same hyperparameters; solid vs dashed = training data", color=INK)
    ax.grid(color=GRID, linewidth=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/h3_confound_factorial.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
