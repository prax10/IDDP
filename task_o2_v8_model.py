"""
Task O2: v8 = v7 + task_o1 features, with priority/type rebuilt as of T.
Stage "val": TRAIN -> VAL ablation (1 seed, v7 config from N4). Stage "test": TRAIN+VAL -> TEST, 5 seeds, scored once.
"""
import gc, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from h_common import LABEL, REPORTS_DIR, MODELS_DIR
import task_n2_v7_model as n2

SEL = json.load(open(f"{MODELS_DIR}/n4_v7_selected.json"))
CFG, HL = SEL["config"], SEL["half_life"]
V7 = n2.VARIANTS["v7"]
PT = [c for c in V7 if c not in ("Priority_Normalized", "Type_Normalized")] + ["priority_T", "type_T", "n_priority_up"]
KNN = ["knn_late_rate", "knn_log_ratio", "knn_top_sim", "knn_n"]
EPIC = ["has_epic_T", "log_epic_children", "epic_done_share", "log_epic_resolved_14d", "log_epic_age_days", "log_epic_days_left_est"]
COMP = ["has_component_T", "comp_rate_known", "log_comp_n_known"]
REP = ["rep_rate_90d", "log_rep_n_res_90d"]
VARIANTS = {
    "v7": V7,
    "v7_priority_type_at_T": PT,
    "pt+knn": PT + KNN,
    "pt+epic": PT + EPIC,
    "pt+component": PT + COMP,
    "pt+reporter": PT + REP,
    "v8_all": PT + KNN + EPIC + COMP + REP,
    "pt+epic+reporter": PT + EPIC + REP,
    "pt+epic+reporter+knn": PT + EPIC + REP + KNN,
    "v8": PT + EPIC,          # chosen by 3-seed validation (task_o3_select.py, reports/o3_v8_selection.csv)
}


def data():
    ck = n2.data()
    nf = pd.read_csv("data/features/phase8_v8_features.csv")
    nf["type_T"] = nf["type_T"].astype("category")
    ck = ck.merge(nf, on=["Issue_ID", "checkpoint_index"], how="left")
    assert ck["n_priority_up"].notna().all()
    return ck


def val_stage(names):
    ck = data()
    tr, va = ck[ck.split == "train"], ck[ck.split == "val"]
    yv, cv = va[LABEL].to_numpy(), va["checkpoint_index"].to_numpy()
    out = f"{REPORTS_DIR}/o2_val_ablation.csv"
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    done = {r["variant"] for r in rows}
    for name in names:
        if name in done:
            continue
        cols = VARIANTS[name]
        t0 = time.time()
        p = n2.fit(tr, cols, 0, CFG, HL).predict_proba(va[cols])[:, 1]
        r = {"variant": name, "n_features": len(cols), "val_auc": roc_auc_score(yv, p), **n2.per_ck(yv, p, cv)}
        rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{name:26s} val AUC {r['val_auc']:.4f}  M1 {r['M1']:.4f}  M10 {r['M10']:.4f}  ({len(cols)} feats, {time.time()-t0:.0f}s)", flush=True)


def test_stage(names, cfg=None, tag=""):
    cfg = cfg or CFG
    ck = data()
    tv, te = ck[ck.split.isin(["train", "val"])], ck[ck.split == "test"].reset_index(drop=True)
    yt = te[LABEL].to_numpy()
    for name in names:
        out = f"{REPORTS_DIR}/n2_preds_{name}{tag}.npz"
        if os.path.exists(out):
            continue
        cols = VARIANTS[name]
        P = []
        for s in range(5):
            m = n2.fit(tv, cols, s, cfg, HL)
            P.append(m.predict_proba(te[cols])[:, 1])
            if name.startswith("v8"):
                m.save_model(f"{MODELS_DIR}/xgb_v8{tag}_seed{s}.json")
            print(f"  {name} seed {s}", flush=True)
        np.savez_compressed(out, test=np.array(P), yt=yt, ck_t=te["checkpoint_index"].to_numpy(),
                            past=te["past_median"].to_numpy(), safe=te["safe"].to_numpy().astype(bool),
                            issue=te["Issue_ID"].to_numpy())
        print(f"{name}{tag}: test AUC (seed mean) {np.mean([roc_auc_score(yt, p) for p in P]):.4f}", flush=True)
        gc.collect()


if __name__ == "__main__":
    if sys.argv[1] == "val":
        val_stage(sys.argv[2:] or list(VARIANTS))
    else:
        test_stage(sys.argv[2:])
