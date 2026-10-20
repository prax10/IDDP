"""
Task N2: v7 = v6 with final-state leaks removed + new known-at-T features (task_n1_v7_features.py).

Stage "val"  : every variant fitted on TRAIN, scored on VAL (1 seed). Used to choose v7. Test untouched.
Stage "test" : pre-declared variants (v6, v6_clean, v7) fitted on TRAIN+VAL, 5 seeds, scored once on TEST.
               Predictions saved to reports/n2_preds_<variant>.npz.
Stage "report": AUC tables + paired bootstrap vs v6 -> reports/n2_results.csv
"""
import gc, json, os, sys, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
from h_common import LABEL, REPORTS_DIR, MODELS_DIR, seeded, paired_bootstrap_seeds
from task_i2_model_upgrades import weights
from task_j4_v5_model import load_all

META = json.load(open(f"{MODELS_DIR}/xgb_dynamic_live_v6_meta.json"))
V6 = list(META["features"])
CFG = dict(META["config"])
CFG["n_estimators"] = int(CFG["n_estimators"])
HL = float(META["recency_half_life_years"])

# features computed from an issue's FINAL state (final assignee, final link list, final story points)
LEAK_ASSIGNEE = ["assignee_prior_resolved_count", "assignee_prior_delay_rate", "assignee_concurrent_open",
                 "is_unassigned", "assignee_open_at_T", "assignee_rate_known", "assignee_n_known"]
LEAK_LINKS = ["n_links", "n_blocking_links"]
LEAK_SP = ["Story_Point"]
T_ASSIGNEE = ["is_unassigned_T", "assignee_rate_known_T", "assignee_n_known_T"]
NEW_LINKS = ["n_links_at_T", "n_blockedby_at_T", "n_blocks_at_T", "n_dup_at_T", "log_min_since_link_change",
             "n_open_blockers_at_T", "n_blockers_resolved_7d", "any_open_blocker"]
NEW_FV = ["has_fixver_at_T", "n_fixver_at_T", "fixver_placeholder", "log_fixver_assigned", "fixver_done_share",
          "log_fixver_resolved_14d", "log_fixver_age_days", "log_fixver_days_left_est"]
NEW_FORM = ["a_rate_90d", "a_n_res_30d", "a_n_res_90d", "a_last5_late", "log_a_days_since_res"]


def minus(a, b):
    return [c for c in a if c not in b]


CLEAN_A = minus(V6, LEAK_ASSIGNEE) + T_ASSIGNEE
CLEAN = minus(CLEAN_A, LEAK_LINKS + LEAK_SP)
VARIANTS = {
    "v6": V6,
    "v6_clean_assignee": CLEAN_A,
    "v6_clean_assignee_links": minus(CLEAN_A, LEAK_LINKS),
    "v6_clean": CLEAN,
    "clean+links_at_T": CLEAN + NEW_LINKS,
    "clean+fixver": CLEAN + NEW_FV,
    "clean+assignee_form": CLEAN + NEW_FORM,
    "v7": CLEAN + NEW_LINKS + NEW_FV + NEW_FORM,
    "v6+new (not clean, info only)": V6 + NEW_LINKS + NEW_FV + NEW_FORM,
}


def data():
    ck = load_all()
    nf = pd.read_csv("data/features/phase8_v7_features.csv")
    for c in nf.columns:
        if nf[c].dtype == "float64":
            nf[c] = nf[c].astype("float32")
    ck = ck.merge(nf, on=["Issue_ID", "checkpoint_index"], how="left")
    assert ck["n_links_at_T"].notna().all()
    return ck


def fit(df, cols, seed, cfg=None, hl=None):
    cfg = cfg or CFG
    hl = HL if hl is None else hl
    m = xgb.XGBClassifier(**{**seeded(seed), **cfg, "n_jobs": 2})
    m.fit(df[cols], df[LABEL], sample_weight=weights(df, hl, df["age_years"].min()))
    return m


def per_ck(y, p, c):
    return {f"M{k}": roc_auc_score(y[c == k], p[c == k]) for k in (1, 3, 5, 8, 10)}


def val_stage(names):
    ck = data()
    tr, va = ck[ck.split == "train"], ck[ck.split == "val"]
    yv, cv = va[LABEL].to_numpy(), va["checkpoint_index"].to_numpy()
    out = f"{REPORTS_DIR}/n2_val_ablation.csv"
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    done = {r["variant"] for r in rows}
    for name in names:
        if name in done:
            continue
        cols = VARIANTS[name]
        t0 = time.time()
        p = fit(tr, cols, 0).predict_proba(va[cols])[:, 1]
        r = {"variant": name, "n_features": len(cols), "val_auc": roc_auc_score(yv, p), **per_ck(yv, p, cv)}
        rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{name:34s} val AUC {r['val_auc']:.4f}  M1 {r['M1']:.4f}  M10 {r['M10']:.4f}  ({len(cols)} feats, {time.time()-t0:.0f}s)", flush=True)


def test_stage(names, cfg=None, hl=None, tag=""):
    ck = data()
    tv, te = ck[ck.split.isin(["train", "val"])], ck[ck.split == "test"].reset_index(drop=True)
    yt = te[LABEL].to_numpy()
    for name in names:
        out = f"{REPORTS_DIR}/n2_preds_{name}{tag}.npz"
        if os.path.exists(out):
            print(f"{name}: cached", flush=True)
            continue
        cols = VARIANTS[name]
        vcfg, vhl = cfg, hl
        if name == "v7" and cfg is None and os.path.exists(f"{MODELS_DIR}/n4_v7_selected.json"):
            sel = json.load(open(f"{MODELS_DIR}/n4_v7_selected.json"))   # chosen by rolling-origin validation (N4)
            vcfg, vhl = sel["config"], sel["half_life"]
            print(f"  v7 config from N4: {vcfg}", flush=True)
        P = []
        for s in range(5):
            t0 = time.time()
            m = fit(tv, cols, s, vcfg, vhl)
            P.append(m.predict_proba(te[cols])[:, 1])
            if name == "v7":
                m.save_model(f"{MODELS_DIR}/xgb_v7{tag}_seed{s}.json")
            print(f"  {name}{tag} seed {s}: {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(out, test=np.array(P), yt=yt, ck_t=te["checkpoint_index"].to_numpy(),
                            past=te["past_median"].to_numpy(), safe=te["safe"].to_numpy().astype(bool),
                            issue=te["Issue_ID"].to_numpy())
        print(f"{name}{tag}: test AUC (seed mean) {np.mean([roc_auc_score(yt, p) for p in P]):.4f}", flush=True)
        gc.collect()


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "val":
        val_stage(sys.argv[2:] or list(VARIANTS))
    elif st == "test":
        test_stage(sys.argv[2:])
