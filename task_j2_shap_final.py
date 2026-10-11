"""
Task J2: SHAP explainability for the final dynamic model (and the Scrum model).

Uses XGBoost's exact TreeSHAP (pred_contribs=True) on the saved seed-0 model,
over a stratified sample of test rows (up to 8,000 per checkpoint).
Outputs
  j2_shap_global_<model>.csv     mean |SHAP| per feature (all test rows)
  j2_shap_groups_<model>.csv     summed by feature group
  j2_shap_shift_<model>.csv      early (M1-M3) vs late (M8-M10) importance shift
  j2_shap_<model>.png            top-20 features + groups + shift
Usage: python task_j2_shap_final.py v5b     (any model with models/xgb_dynamic_pooled_<name>_meta.json)
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from h_common import REPORTS_DIR, MODELS_DIR

GROUPS = {
    "text (30 SBERT PCs)": lambda c: c.startswith("text_pc_"),
    "workflow status history": lambda c: c in {"n_transitions", "n_reopens", "n_back_moves", "ever_in_progress",
                                               "ever_in_review", "status_bucket", "log_min_since_last_change"},
    "recency of activity": lambda c: c in {"log_min_since_last_activity", "log_min_since_last_comment",
                                           "comments_last_7d", "n_field_chg_last_7d"},
    "comments": lambda c: c in {"n_comments", "n_comment_authors"},
    "assignment / reassignment": lambda c: c in {"n_assignee_changes", "ever_reassigned", "assigned_at_T",
                                                 "log_min_since_assign_change", "log_min_to_first_assign",
                                                 "is_unassigned"},
    "people track record": lambda c: c in {"assignee_prior_delay_rate", "assignee_prior_resolved_count",
                                           "assignee_rate_known", "assignee_n_known", "reporter_rate_known"},
    "project context": lambda c: c in {"Project_ID", "proj_rate_known", "proj_rate_known_90d", "proj_n_resolved_90d",
                                       "project_open_at_T", "project_resolved_30d"},
    "elapsed time / threshold": lambda c: c in {"elapsed_minutes", "log_elapsed_over_median", "past_median"},
    "stall + pattern (Kula)": lambda c: c in {"log1p_stall", "pattern_label"},
    "sprint / scrum": lambda c: "sprint" in c,
    "estimates / story points": lambda c: c in {"Story_Point", "sp_at_T", "n_sp_changes", "n_estimate_chg"},
    "issue attributes": lambda c: c in {"Priority_Normalized", "Type_Normalized", "log_title_len", "log_desc_len",
                                        "desc_has_code"},
    "links / field churn": lambda c: c in {"n_links", "n_blocking_links", "n_field_chg_total", "n_priority_chg",
                                           "n_type_chg", "n_rank_chg", "n_link_chg", "n_fixver_chg", "n_epic_chg",
                                           "n_label_chg", "n_component_chg"},
    "workload": lambda c: c in {"assignee_concurrent_open", "assignee_open_at_T"},
}


def group_of(c):
    for g, f in GROUPS.items():
        if f(c):
            return g
    return "other"


def main(name):
    meta = json.load(open(f"{MODELS_DIR}/xgb_dynamic_pooled_{name}_meta.json"))
    cols = meta["features"]
    model = joblib.load(f"{MODELS_DIR}/xgb_dynamic_pooled_{name}_seed0.joblib")
    from task_j4_v5_model import load_all
    ck = load_all()
    te = ck[ck.split == "test"]
    sample = te.groupby("checkpoint_index").sample(n=8000, random_state=0)
    dm = xgb.DMatrix(sample[cols], enable_categorical=True)
    contrib = model.get_booster().predict(dm, pred_contribs=True)[:, :-1]
    S = pd.DataFrame(np.abs(contrib), columns=cols)
    S["checkpoint_index"] = sample["checkpoint_index"].to_numpy()

    glob = S[cols].mean().sort_values(ascending=False).rename("mean_abs_shap")
    glob.to_csv(f"{REPORTS_DIR}/j2_shap_global_{name}.csv")
    grp = glob.groupby(glob.index.map(group_of)).sum().sort_values(ascending=False).rename("summed_mean_abs_shap")
    grp.to_csv(f"{REPORTS_DIR}/j2_shap_groups_{name}.csv")
    early = S[S.checkpoint_index <= 3][cols].mean()
    late = S[S.checkpoint_index >= 8][cols].mean()
    shift = pd.DataFrame({"early_M1_M3": early, "late_M8_M10": late})
    shift["late_minus_early"] = shift["late_M8_M10"] - shift["early_M1_M3"]
    shift = shift.sort_values("late_minus_early")
    shift.to_csv(f"{REPORTS_DIR}/j2_shap_shift_{name}.csv")
    gshift = shift.groupby(shift.index.map(group_of))[["early_M1_M3", "late_M8_M10"]].sum()
    gshift["late_minus_early"] = gshift["late_M8_M10"] - gshift["early_M1_M3"]
    gshift.sort_values("late_minus_early").to_csv(f"{REPORTS_DIR}/j2_shap_group_shift_{name}.csv")

    print(f"Top 20 features (mean |SHAP|, {name}):\n" + glob.head(20).round(4).to_string())
    print("\nBy group:\n" + grp.round(4).to_string())
    print("\nGroup shift early -> late:\n" + gshift.sort_values("late_minus_early").round(4).to_string())

    SURFACE, INK, INK2, GRIDC, ACC, ACC2 = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    fig.patch.set_facecolor(SURFACE)
    t = glob.head(20)[::-1]
    axes[0].barh(t.index, t.values, color=ACC)
    axes[0].set_title("Top 20 features (mean |SHAP|)", color=INK)
    g = grp[::-1]
    axes[1].barh(g.index, g.values, color=ACC)
    axes[1].set_title("Feature groups (summed)", color=INK)
    gs = gshift.sort_values("late_minus_early")
    y = np.arange(len(gs))
    axes[2].barh(y - 0.2, gs["early_M1_M3"], height=0.4, color=ACC, label="early (M1-M3)")
    axes[2].barh(y + 0.2, gs["late_M8_M10"], height=0.4, color=ACC2, label="late (M8-M10)")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(gs.index)
    axes[2].legend(fontsize=9, facecolor=SURFACE, edgecolor=GRIDC)
    axes[2].set_title("What matters early vs late", color=INK)
    for a in axes:
        a.set_facecolor(SURFACE)
        a.grid(color=GRIDC, linewidth=0.8, axis="x")
        a.tick_params(colors=INK2, labelsize=8)
        for sp in a.spines.values():
            sp.set_color(GRIDC)
    fig.suptitle(f"SHAP explanations -- dynamic model {name}", color=INK)
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/j2_shap_{name}.png", dpi=140, facecolor=SURFACE)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v5b")
