"""
Phase 11: reassignment suggester. Importable module, no retraining --
reuses the tuned dynamic pooled model (lr=0.1, depth=8, n_estimators=11)
and swaps only assignee-related features between candidates.
"""

import numpy as np
import pandas as pd
import shap

import demo_data as dd

MIN_CANDIDATE_ISSUES = 20
LOOSENED_THRESHOLDS = [20, 10, 5]
MIN_CANDIDATES_WANTED = 3
RISK_REDUCTION_THRESHOLD = 0.05  # candidate must beat current assignee by >=0.05 absolute risk
TOP_K_DRIVERS = 3  # how many top |SHAP| features we inspect for the honesty check

_explainer_cache = {}


def _get_explainer(dynamic_model):
    key = id(dynamic_model)
    if key not in _explainer_cache:
        _explainer_cache[key] = shap.TreeExplainer(dynamic_model)
    return _explainer_cache[key]


def _candidate_pool(feat_full, project_id, exclude_assignee_id):
    """Developers with >= N prior resolved issues in this project. Loosens
    the threshold if too few candidates are found; reports which threshold
    was used."""
    proj_issues = feat_full[feat_full["Project_ID"] == project_id]
    counts = proj_issues["Assignee_ID"].value_counts()

    for threshold in LOOSENED_THRESHOLDS:
        eligible = counts[counts >= threshold].index
        eligible = [a for a in eligible if a != exclude_assignee_id and not pd.isna(a)]
        if len(eligible) >= MIN_CANDIDATES_WANTED or threshold == LOOSENED_THRESHOLDS[-1]:
            return eligible, threshold
    return [], LOOSENED_THRESHOLDS[-1]


def _candidate_snapshot_features(feat_full, project_id, assignee_id):
    """Most recent (by Creation_Date) issue for this assignee in this
    project, used as a proxy for their 'current' workload/track record."""
    rows = feat_full[
        (feat_full["Project_ID"] == project_id) & (feat_full["Assignee_ID"] == assignee_id)
    ]
    if "Creation_Date" in rows.columns:
        rows = rows.sort_values("Creation_Date")
    latest = rows.iloc[-1]
    return {col: latest[col] for col in dd.ASSIGNEE_FEATURE_COLS}


def suggest_reassignment(issue_id, portfolio, feat_full, dynamic_model, top_n=5):
    """
    Returns a dict:
      {
        "issue_id", "current_assignee_id", "current_risk",
        "verdict": "reassignment_may_help" | "reassignment_unlikely_to_help",
        "top_drivers": [(feature, shap_value), ...],
        "candidate_pool_threshold": int,
        "candidates": [ {assignee_id, predicted_risk, risk_delta}, ... ]  # empty if verdict says unlikely to help
      }
    """
    row = portfolio[portfolio["ID"] == issue_id]
    if row.empty:
        raise ValueError(f"issue_id {issue_id} not found in the current portfolio")
    row = row.iloc[0]

    project_id = row["Project_ID"]
    current_assignee = row["Assignee_ID"]
    current_features = row[dd.DYN_FEATURE_COLS].to_frame().T
    for col in dd.CATEGORICAL_COLS + ["pattern_label"]:
        current_features[col] = current_features[col].astype(portfolio[col].dtype)
    for col in dd.STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes"]:
        current_features[col] = current_features[col].astype(float) if col not in dd.CATEGORICAL_COLS else current_features[col]

    current_risk = float(dynamic_model.predict_proba(current_features[dd.DYN_FEATURE_COLS])[:, 1][0])

    # --- Honesty check: SHAP under the current assignee ---
    explainer = _get_explainer(dynamic_model)
    shap_values = explainer(current_features[dd.DYN_FEATURE_COLS])
    contrib = pd.Series(shap_values.values[0], index=dd.DYN_FEATURE_COLS)
    contrib_abs_sorted = contrib.abs().sort_values(ascending=False)
    top_features = contrib_abs_sorted.head(TOP_K_DRIVERS).index.tolist()
    top_drivers = [(f, float(contrib[f])) for f in top_features]

    assignee_features_in_top = [f for f in top_features if f in dd.ASSIGNEE_FEATURE_COLS]
    assignee_material = len(assignee_features_in_top) > 0

    result = {
        "issue_id": issue_id,
        "current_assignee_id": current_assignee,
        "current_risk": current_risk,
        "top_drivers": top_drivers,
        "candidate_pool_threshold": None,
        "candidates": [],
    }

    if not assignee_material:
        result["verdict"] = "reassignment_unlikely_to_help"
        result["explanation"] = (
            f"The top drivers of this issue's predicted risk are "
            f"{', '.join(f for f, _ in top_drivers)} -- none of these are assignee-related. "
            f"Reassigning this issue is unlikely to change the outcome; the risk is coming from "
            f"the issue itself (its type, links, elapsed time, or content), not from who it's assigned to."
        )
        return result

    # --- Candidate pool ---
    candidates, threshold_used = _candidate_pool(feat_full, project_id, current_assignee)
    result["candidate_pool_threshold"] = threshold_used

    if not candidates:
        result["verdict"] = "reassignment_unlikely_to_help"
        result["explanation"] = (
            f"Assignee-related features ({', '.join(assignee_features_in_top)}) are material to this "
            f"issue's risk, but no candidate developers with >= {threshold_used} prior resolved issues "
            f"in this project were found. No swap can be evaluated."
        )
        return result

    scored_candidates = []
    for cand_id in candidates:
        cand_features = current_features.copy()
        snapshot = _candidate_snapshot_features(feat_full, project_id, cand_id)
        for col, val in snapshot.items():
            cand_features[col] = float(val)
        cand_features["is_unassigned"] = 0.0
        cand_risk = float(dynamic_model.predict_proba(cand_features[dd.DYN_FEATURE_COLS])[:, 1][0])
        scored_candidates.append({
            "assignee_id": cand_id,
            "predicted_risk": cand_risk,
            "risk_delta": cand_risk - current_risk,  # negative = improvement
        })

    scored_candidates.sort(key=lambda c: c["predicted_risk"])
    best = scored_candidates[0]

    if best["risk_delta"] <= -RISK_REDUCTION_THRESHOLD:
        result["verdict"] = "reassignment_may_help"
        result["candidates"] = scored_candidates[:top_n]
        result["explanation"] = (
            f"Assignee-related features ({', '.join(assignee_features_in_top)}) materially drive this "
            f"issue's risk, and the best candidate reduces predicted risk by "
            f"{abs(best['risk_delta']):.3f} (>= {RISK_REDUCTION_THRESHOLD} threshold) -- reassignment "
            f"looks worthwhile."
        )
    else:
        result["verdict"] = "reassignment_unlikely_to_help"
        result["candidates"] = scored_candidates[:top_n]
        result["explanation"] = (
            f"Assignee-related features ({', '.join(assignee_features_in_top)}) are material, but the "
            f"best available candidate only changes predicted risk by {best['risk_delta']:+.3f}, below the "
            f"{RISK_REDUCTION_THRESHOLD} threshold needed to call it a meaningful improvement."
        )

    return result
