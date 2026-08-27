"""
Task B6: accuracy reporting for the final tuned models, plus bootstrap
CIs and significance tests (Wilcoxon signed-rank + Vargha-Delaney A12)
for the static-vs-dynamic and cross-vs-within-project comparisons, and
the pattern-label contribution.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, accuracy_score

FEATURES_DIR = "data/features"
REPORTS_DIR = "reports"
MODELS_DIR = "models"

STATIC_FEATURE_COLS = [
    "Project_ID", "Priority_Normalized", "Type_Normalized", "Story_Point",
    "assignee_prior_resolved_count", "assignee_prior_delay_rate",
    "assignee_concurrent_open", "is_unassigned", "n_links", "n_blocking_links",
]
TEXT_COLS = [f"text_pc_{i}" for i in range(30)]
STATIC_FEATURE_COLS_V2 = STATIC_FEATURE_COLS + TEXT_COLS
CATEGORICAL_COLS = ["Project_ID", "Priority_Normalized", "Type_Normalized"]
DYN_FEATURE_COLS = STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes", "pattern_label"]
DYN_FEATURE_COLS_NO_PATTERN = STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes"]

TUNED_DYN_CFG = dict(
    n_estimators=11, learning_rate=0.1, max_depth=8,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    enable_categorical=True, tree_method="hist", random_state=42,
)

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

SURFACE, INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRIDLINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
)


def load_dynamic():
    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv")
    pattern = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints_with_pattern.csv")
    ck = ck.merge(pattern, on=["Issue_ID", "checkpoint_index"], how="left")
    ck["log1p_stall"] = np.log1p(ck["stall_ratio"])
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv")
    ck = ck.drop(columns=["Project_ID", "Type_Normalized"]).merge(
        feat, left_on="Issue_ID", right_on="ID", how="left"
    )
    for col in CATEGORICAL_COLS:
        ck[col] = ck[col].astype("category")
    ck["pattern_label"] = ck["pattern_label"].astype("category")
    return ck


def vargha_delaney_a12_paired(x, y):
    """A12 via pairwise comparison for equal-length paired samples:
    P(x>y) + 0.5*P(x==y)."""
    x, y = np.asarray(x), np.asarray(y)
    wins = np.sum(x > y)
    ties = np.sum(x == y)
    return (wins + 0.5 * ties) / len(x)


def part1_accuracy(ck, static_model, pooled_model):
    print("=== Part 1: accuracy reporting ===")
    split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv")
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv").merge(
        split[["ID", "split"]], on="ID", how="left"
    )
    for col in CATEGORICAL_COLS:
        feat[col] = feat[col].astype("category")
    static_test = feat[feat["split"] == "test"]

    static_proba_full = static_model.predict_proba(static_test[STATIC_FEATURE_COLS_V2])[:, 1]
    y_full = static_test["delayed_v2"].to_numpy()
    static_acc_05 = accuracy_score(y_full, (static_proba_full >= 0.5).astype(int))
    static_acc_tuned = accuracy_score(y_full, (static_proba_full >= 0.405).astype(int))
    maj_class_full = int(round(1 - y_full.mean())) if y_full.mean() < 0.5 else 1
    maj_class_full = 0 if y_full.mean() < 0.5 else 1
    maj_acc_full = accuracy_score(y_full, np.full(len(y_full), maj_class_full))
    print(f"  Static, full test set: acc@0.5={static_acc_05:.4f} acc@tuned(0.405)={static_acc_tuned:.4f} "
          f"majority_acc={maj_acc_full:.4f} (majority_class={maj_class_full})")

    test_ck = ck[ck["split"] == "test"]
    rows = []
    for t in range(1, 11):
        sub = test_ck[test_ck["checkpoint_index"] == t]
        y_true = sub["delayed_v2"].to_numpy()
        n = len(y_true)
        majority_class = 1 if y_true.mean() >= 0.5 else 0
        maj_acc = accuracy_score(y_true, np.full(n, majority_class))

        pooled_proba = pooled_model.predict_proba(sub[DYN_FEATURE_COLS])[:, 1]
        pooled_acc = accuracy_score(y_true, (pooled_proba >= 0.5).astype(int))

        static_proba = static_model.predict_proba(sub[STATIC_FEATURE_COLS_V2])[:, 1]
        static_acc = accuracy_score(y_true, (static_proba >= 0.5).astype(int))

        rows.append({
            "checkpoint_index": t, "N": n, "majority_class": majority_class,
            "majority_class_accuracy": maj_acc,
            "static_accuracy": static_acc, "pooled_dynamic_accuracy": pooled_acc,
        })
        print(f"  M{t}: N={n} majority_acc={maj_acc:.4f} static_acc={static_acc:.4f} "
              f"pooled_acc={pooled_acc:.4f}")

    acc_df = pd.DataFrame(rows)
    final_table = pd.read_csv(f"{REPORTS_DIR}/phase8_final_tuned_comparison.csv")
    final_table = final_table.drop(
        columns=[c for c in ["majority_class", "majority_class_accuracy", "static_accuracy",
                              "pooled_dynamic_accuracy"] if c in final_table.columns]
    )
    final_table = final_table.merge(acc_df, on=["checkpoint_index", "N"], how="left")
    final_table.to_csv(f"{REPORTS_DIR}/phase8_final_tuned_comparison.csv", index=False)
    print(f"\n  updated -> {REPORTS_DIR}/phase8_final_tuned_comparison.csv")

    return {
        "static_acc_05": static_acc_05, "static_acc_tuned": static_acc_tuned,
        "static_majority_acc": maj_acc_full,
    }, acc_df


def part2a_bootstrap_ci(ck, static_model, pooled_model):
    print("\n=== Part 2A: bootstrap 95% CIs on AUC per checkpoint ===")
    test_ck = ck[ck["split"] == "test"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    rows = []
    first_significant = None
    for t in range(1, 11):
        sub = test_ck[test_ck["checkpoint_index"] == t].reset_index(drop=True)
        y_true = sub["delayed_v2"].to_numpy()
        static_proba = static_model.predict_proba(sub[STATIC_FEATURE_COLS_V2])[:, 1]
        pooled_proba = pooled_model.predict_proba(sub[DYN_FEATURE_COLS])[:, 1]

        n = len(sub)
        static_aucs, pooled_aucs, diffs = [], [], []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, n)
            yb = y_true[idx]
            if yb.min() == yb.max():
                continue
            sa = roc_auc_score(yb, static_proba[idx])
            pa = roc_auc_score(yb, pooled_proba[idx])
            static_aucs.append(sa)
            pooled_aucs.append(pa)
            diffs.append(pa - sa)

        static_aucs, pooled_aucs, diffs = map(np.array, (static_aucs, pooled_aucs, diffs))
        static_ci = np.percentile(static_aucs, [2.5, 97.5])
        pooled_ci = np.percentile(pooled_aucs, [2.5, 97.5])
        diff_ci = np.percentile(diffs, [2.5, 97.5])
        significant = diff_ci[0] > 0
        if significant and first_significant is None:
            first_significant = t

        rows.append({
            "checkpoint_index": t, "N": n,
            "static_auc_point": roc_auc_score(y_true, static_proba),
            "static_auc_ci_lo": static_ci[0], "static_auc_ci_hi": static_ci[1],
            "pooled_auc_point": roc_auc_score(y_true, pooled_proba),
            "pooled_auc_ci_lo": pooled_ci[0], "pooled_auc_ci_hi": pooled_ci[1],
            "diff_ci_lo": diff_ci[0], "diff_ci_hi": diff_ci[1],
            "significant_dynamic_better": significant,
        })
        print(f"  M{t}: static={rows[-1]['static_auc_point']:.4f} [{static_ci[0]:.4f},{static_ci[1]:.4f}] "
              f"pooled={rows[-1]['pooled_auc_point']:.4f} [{pooled_ci[0]:.4f},{pooled_ci[1]:.4f}] "
              f"diff_CI=[{diff_ci[0]:+.4f},{diff_ci[1]:+.4f}] "
              f"significant={'YES' if significant else 'no'}")

    ci_df = pd.DataFrame(rows)
    print(f"\n  bootstrap seed={BOOTSTRAP_SEED}, n_resamples={N_BOOTSTRAP}")
    print(f"  FIRST checkpoint where diff CI excludes zero (favoring dynamic): "
          f"{'M' + str(first_significant) if first_significant else 'NONE'}")

    plot_ci_band(ci_df)
    return ci_df, first_significant


def plot_ci_band(ci_df):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    diff_point = ci_df["pooled_auc_point"] - ci_df["static_auc_point"]
    ax.plot(ci_df["checkpoint_index"], diff_point, color="#eb6834", linewidth=2, marker="o", markersize=5,
            label="Pooled dynamic - Static (AUC)")
    ax.fill_between(ci_df["checkpoint_index"], ci_df["diff_ci_lo"], ci_df["diff_ci_hi"],
                     color="#eb6834", alpha=0.2, label="95% bootstrap CI")
    ax.axhline(0, color=INK_MUTED, linewidth=1.5, linestyle="--", label="No difference")

    ax.set_xlabel("Checkpoint index (1-10)", color=INK_SECONDARY)
    ax.set_ylabel("AUC difference (dynamic - static)", color=INK_SECONDARY)
    ax.set_title("Dynamic - static AUC difference, with 95% bootstrap CI", color=INK_PRIMARY)
    ax.set_xticks(range(1, 11))
    ax.tick_params(colors=INK_MUTED)
    ax.grid(color=GRIDLINE, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=GRIDLINE, labelcolor=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/auc_difference_ci.png", dpi=150, facecolor=SURFACE)
    print(f"  saved -> {REPORTS_DIR}/auc_difference_ci.png")


def part2b_wilcoxon_checkpoints(ci_df):
    print("\n=== Part 2B: Wilcoxon signed-rank (static vs dynamic across M1-M10) ===")
    static_aucs = ci_df["static_auc_point"].to_numpy()
    pooled_aucs = ci_df["pooled_auc_point"].to_numpy()
    stat, p = wilcoxon(pooled_aucs, static_aucs)
    a12 = vargha_delaney_a12_paired(pooled_aucs, static_aucs)
    print(f"  Wilcoxon signed-rank: stat={stat:.4f} p={p:.6f}")
    print(f"  Vargha-Delaney A12 (P(dynamic > static) + 0.5*ties): {a12:.4f}")
    return {"wilcoxon_stat": stat, "wilcoxon_p": p, "a12": a12}


def part2c_phase9():
    print("\n=== Part 2C: Wilcoxon signed-rank (Phase 9 cross vs within, 38 projects) ===")
    df = pd.read_csv(f"{REPORTS_DIR}/phase9_generalization_v3.csv")
    valid = df[~df["within_project_skipped"]]
    cross = valid["cross_project_auc"].to_numpy()
    within = valid["within_project_auc"].to_numpy()
    stat, p = wilcoxon(within, cross)
    a12 = vargha_delaney_a12_paired(within, cross)
    print(f"  n={len(valid)} projects")
    print(f"  Wilcoxon signed-rank (within vs cross): stat={stat:.4f} p={p:.6f}")
    print(f"  Vargha-Delaney A12 (P(within > cross) + 0.5*ties): {a12:.4f}")
    return {"n_projects": len(valid), "wilcoxon_stat": stat, "wilcoxon_p": p, "a12": a12}


def part2d_pattern_contribution(ck):
    print("\n=== Part 2D: pattern-label contribution (bootstrap CI per checkpoint) ===")
    trainval = ck[ck["split"].isin(["train", "val"])]
    test_ck = ck[ck["split"] == "test"]

    print("  training matched no-pattern model (same tuned config, no pattern_label feature) ...")
    no_pattern_model = xgb.XGBClassifier(**TUNED_DYN_CFG)
    no_pattern_model.fit(trainval[DYN_FEATURE_COLS_NO_PATTERN], trainval["delayed_v2"])
    joblib.dump(no_pattern_model, f"{MODELS_DIR}/xgb_dynamic_pooled_no_pattern_v3_tuned.joblib")

    with_pattern_model = joblib.load(f"{MODELS_DIR}/xgb_dynamic_pooled_v3_tuned.joblib")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for t in range(1, 11):
        sub = test_ck[test_ck["checkpoint_index"] == t].reset_index(drop=True)
        y_true = sub["delayed_v2"].to_numpy()
        with_proba = with_pattern_model.predict_proba(sub[DYN_FEATURE_COLS])[:, 1]
        without_proba = no_pattern_model.predict_proba(sub[DYN_FEATURE_COLS_NO_PATTERN])[:, 1]

        n = len(sub)
        diffs = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, n)
            yb = y_true[idx]
            if yb.min() == yb.max():
                continue
            wa = roc_auc_score(yb, with_proba[idx])
            wo = roc_auc_score(yb, without_proba[idx])
            diffs.append(wa - wo)
        diffs = np.array(diffs)
        ci = np.percentile(diffs, [2.5, 97.5])
        point_diff = roc_auc_score(y_true, with_proba) - roc_auc_score(y_true, without_proba)
        significant = not (ci[0] <= 0 <= ci[1])
        rows.append({
            "checkpoint_index": t, "N": n,
            "with_pattern_auc": roc_auc_score(y_true, with_proba),
            "without_pattern_auc": roc_auc_score(y_true, without_proba),
            "pattern_diff_point": point_diff,
            "pattern_diff_ci_lo": ci[0], "pattern_diff_ci_hi": ci[1],
            "pattern_significant": significant,
        })
        print(f"  M{t}: with={rows[-1]['with_pattern_auc']:.4f} without={rows[-1]['without_pattern_auc']:.4f} "
              f"diff={point_diff:+.4f} CI=[{ci[0]:+.4f},{ci[1]:+.4f}] "
              f"significant={'YES' if significant else 'no'}")

    df = pd.DataFrame(rows)
    n_significant = df["pattern_significant"].sum()
    print(f"\n  pattern contribution significant at {n_significant}/10 checkpoints")
    return df


def main():
    ck = load_dynamic()
    static_model = joblib.load(f"{MODELS_DIR}/xgb_static_v3_tuned.joblib")
    pooled_model = joblib.load(f"{MODELS_DIR}/xgb_dynamic_pooled_v3_tuned.joblib")

    acc_summary, acc_df = part1_accuracy(ck, static_model, pooled_model)
    ci_df, first_sig = part2a_bootstrap_ci(ck, static_model, pooled_model)
    wilcoxon_ck = part2b_wilcoxon_checkpoints(ci_df)
    wilcoxon_p9 = part2c_phase9()
    pattern_df = part2d_pattern_contribution(ck)

    stats_rows = [
        {"test": "checkpoint_static_vs_dynamic", "n": 10,
         "wilcoxon_stat": wilcoxon_ck["wilcoxon_stat"], "wilcoxon_p": wilcoxon_ck["wilcoxon_p"],
         "a12": wilcoxon_ck["a12"], "first_significant_checkpoint": first_sig},
        {"test": "phase9_cross_vs_within", "n": wilcoxon_p9["n_projects"],
         "wilcoxon_stat": wilcoxon_p9["wilcoxon_stat"], "wilcoxon_p": wilcoxon_p9["wilcoxon_p"],
         "a12": wilcoxon_p9["a12"], "first_significant_checkpoint": None},
    ]
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(f"{REPORTS_DIR}/statistical_tests.csv", index=False)
    pattern_df.to_csv(f"{REPORTS_DIR}/statistical_tests_pattern_contribution.csv", index=False)
    ci_df.to_csv(f"{REPORTS_DIR}/statistical_tests_checkpoint_ci.csv", index=False)
    print(f"\nSaved -> {REPORTS_DIR}/statistical_tests.csv, "
          f"{REPORTS_DIR}/statistical_tests_pattern_contribution.csv, "
          f"{REPORTS_DIR}/statistical_tests_checkpoint_ci.csv")


if __name__ == "__main__":
    main()
