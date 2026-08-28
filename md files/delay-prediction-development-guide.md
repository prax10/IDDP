# Intelligent Dynamic Delay Prediction — Full Development Guide

*From EDA to a working, explainable, dynamic delay-prediction model on TAWOS.*

This is the build manual for the whole project. Work top to bottom. Each
phase says **what to do**, **the code**, **what to look for**, and **the
decision it drives**. Nothing here assumes anything not already set up.

---

## 0. Where you are (baseline)

- MySQL 8.4 running; full TAWOS loaded into `TAWOS_DB` (458,232 issues, 39
  projects, 12 Jira repos — the v1.1 figures; use these, not 508k/44).
- Python connects via SQLAlchemy + PyMySQL. Password has an `@`, so it is
  URL-encoded as `%40`.
- Work in Python (pandas), not Workbench SQL, because the model is Python and
  the same code carries into the build.

**Standard notebook header (top of every notebook):**
```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
pd.set_option('display.max_columns', None)
engine = create_engine("mysql+pymysql://root:root%40123@localhost/TAWOS_DB")
```

**Suggested file layout:**
```
capstone/
  01_EDA.ipynb
  02_labeling_and_features.ipynb
  03_static_model.ipynb
  04_dynamic_model.ipynb
  05_generalization.ipynb
  06_explainability.ipynb
  data/            # cached CSVs so you don't re-query MySQL constantly
  models/          # saved trained models
  app/             # Streamlit prototype
```

**Disk note:** the MacBook Air is tight on space. Cache intermediate
DataFrames to `data/*.parquet` so you query MySQL once, not repeatedly, and
keep an eye that free space doesn't hit zero (MySQL errors out if it does).

---

## PHASE 1 — Exploratory Data Analysis

**Goal of EDA:** answer the specific questions that decide how the model is
built — not to make charts. Every block below ties to a decision.

### 1.1 Load resolved issues
```python
df = pd.read_sql("""SELECT ID, Project_ID, Priority, Type, Status, Resolution,
  Resolution_Time_Minutes, In_Progress_Minutes, Story_Point, Total_Effort_Minutes,
  Assignee_ID, Creator_ID, Reporter_ID, Sprint_ID,
  Creation_Date, Resolution_Date, Estimation_Date
FROM Issue""", engine)
print("Total:", len(df))
print("Resolved:", df['Resolution_Time_Minutes'].notna().sum())
```
**Decision:** the resolved count is your true trainable size. You can only
learn delay from finished issues.

### 1.2 Issues per project
```python
pp = df.groupby('Project_ID').size().sort_values(ascending=False)
print(pp); print("Projects:", len(pp))
```
**Decision:** which projects are big enough to model, and which to hold out
for the generalization test (need enough resolved issues per project).

### 1.3 Missing values
```python
print((df.isna().mean()*100).round(1).sort_values(ascending=False))
```
**Decision:** a column that's mostly empty (e.g. Story_Point, Estimation_Date)
can't be a core feature. This prunes your feature list before you build it.

---

## PHASE 2 — Define the target (`delayed`)

The single most important decision in the project. TAWOS has no "delayed"
column; you derive it.

### 2.1 Resolution-time distribution
```python
res = df[df['Resolution_Time_Minutes'].notna()].copy()
res['days'] = res['Resolution_Time_Minutes']/1440
print(res['days'].describe())
res['days'].clip(0,60).hist(bins=50); plt.title('Resolution days (clip 60)'); plt.show()
```
**Expect:** heavy right skew. Note median and 75th percentile.

### 2.2 The label — per-project median (primary definition)
```python
res['proj_median'] = res.groupby('Project_ID')['Resolution_Time_Minutes'].transform('median')
res['delayed'] = (res['Resolution_Time_Minutes'] > res['proj_median']).astype(int)
print(res['delayed'].value_counts(normalize=True))
```
**Why per-project median:** a "slow" task in a fast project differs from a slow
one in a slow project. Normalizing per project makes "delayed" meaningful
across 39 different teams — which is exactly what a *generalization* study needs.

**Alternative definitions to keep in your back pocket (mention in viva):**
- **Top-25% slowest** (`> 75th percentile`): fewer positives (~25%), more
  realistic "these are the genuinely late ones," but imbalanced.
- **Estimated vs actual**: if `Total_Effort_Minutes` / `Story_Point` estimates
  are reliable, delayed = actual >> estimate. Only viable if those fields are
  well-populated (check 1.3).

**Decision:** which definition. Default to per-project median unless the data
pushes you elsewhere. Whatever you pick, state it explicitly in the paper —
label definition is the first thing a reviewer checks.

### 2.3 Sanity check
```python
print(res.groupby('Priority')['delayed'].mean().sort_values(ascending=False))
```
**Look for:** delay rate varying by priority in a sensible way. Confirms the
label captures something real, not noise.

---

## PHASE 3 — Confirm the dynamic method is feasible

This is what makes the project *dynamic* (your novelty) rather than a static
baseline. It hinges on the `Change_Log` table.

### 3.1 Milestone check
```python
cl = pd.read_sql("""SELECT Issue_ID, From_String, To_String, Creation_Date
FROM Change_Log WHERE Change_Type='STATUS'
ORDER BY Issue_ID, Creation_Date LIMIT 40""", engine)
print(cl)
```
**Look for:** the same `Issue_ID` with multiple timestamped status rows
(Open → In Progress → Resolved). If present → dynamic prediction is feasible.

### 3.2 How many issues have a usable timeline
```python
q = """SELECT status_changes, COUNT(*) AS n FROM (
  SELECT Issue_ID, COUNT(*) AS status_changes
  FROM Change_Log WHERE Change_Type='STATUS' GROUP BY Issue_ID
) t GROUP BY status_changes ORDER BY status_changes LIMIT 15"""
print(pd.read_sql(q, engine))
```
**Decision:** issues with 2+ status changes are dynamically predictable.
- Many such issues → full dynamic model.
- Few → dynamic on the subset that has them + **snapshot fallback** for the
  rest (predict at creation and at a mid-point). This is your feasibility-slide
  risk mitigation, now data-driven.

---

## PHASE 4 — Validate engineered features

Confirm each feature from your research gaps actually exists and varies.

### 4.1 Workload (Gap 4)
```python
wl = pd.read_sql("""SELECT Assignee_ID, COUNT(*) issues
FROM Issue WHERE Assignee_ID IS NOT NULL
GROUP BY Assignee_ID ORDER BY issues DESC LIMIT 10""", engine)
print(wl)
```
**Decision:** assignees with many issues → per-developer history + concurrent
load are computable.

### 4.2 Links (Gap 9)
```python
print(pd.read_sql("SELECT COUNT(*) total_links FROM Issue_Link", engine))
print(pd.read_sql("""SELECT Description, COUNT(*) n FROM Issue_Link
GROUP BY Description ORDER BY n DESC LIMIT 10""", engine))
```
**Decision:** enough "blocks"/"depends" links → dependency features viable.

### 4.3 Text (Gap 6)
```python
print(pd.read_sql("""SELECT SUM(Description_Text IS NOT NULL AND Description_Text<>'') has_text,
  COUNT(*) total FROM Issue""", engine))
```
**Decision:** most issues have text → SBERT features viable.

---

## PHASE 5 — Feature–target relationships
```python
for col in ['Priority','Type']:
    print(f"\n--- delay rate by {col} ---")
    print(res.groupby(col)['delayed'].agg(['mean','count']).sort_values('mean', ascending=False))
```
**Decision:** variation across categories = usable signal. Flat = weak feature.

**End of EDA.** You now know: usable size, label definition + balance, whether
dynamic is feasible, and which features are real. Everything below builds on
these answers.

---

## PHASE 6 — Feature engineering (the static feature table)

Build one row per resolved issue with features known *at prediction time*.

**Critical rule — no leakage:** never use anything only known *after* the task
finished (e.g. `Resolution_Date`, `Resolution_Time_Minutes`, final status) as
a feature. Those define the label; using them as inputs is cheating and the
model would score ~100% and mean nothing.

### 6.1 Base features (safe, known at/near creation)
```python
feat = res[['ID','Project_ID','Priority','Type','Story_Point','delayed']].copy()
# encode categoricals
feat = pd.get_dummies(feat, columns=['Priority','Type'], dummy_na=True)
```

### 6.2 Developer-workload features (Gap 4)
For each issue, at its creation time, compute:
- assignee's number of previously-resolved issues (experience)
- assignee's historical delay rate (their past `delayed` mean, using only
  earlier issues — chronological, no leakage)
- assignee's concurrent open issues at creation (workload)
```python
# historical delay rate per assignee, computed chronologically
res_sorted = res.sort_values('Creation_Date')
res_sorted['assignee_prior_delay_rate'] = (
    res_sorted.groupby('Assignee_ID')['delayed']
    .apply(lambda s: s.shift().expanding().mean()).reset_index(level=0, drop=True)
)
```
(Concurrent load needs an interval count — issues where creation < T <
resolution for that assignee. Build it once, cache it.)

### 6.3 Dependency features (Gap 9)
```python
links = pd.read_sql("""SELECT Issue_ID, COUNT(*) n_links
FROM Issue_Link GROUP BY Issue_ID""", engine)
feat = feat.merge(links, left_on='ID', right_on='Issue_ID', how='left')
feat['n_links'] = feat['n_links'].fillna(0)
```
(Optionally: count of *blocking* links, and whether any blocker is itself
delayed.)

### 6.4 Text features (Gap 6, SBERT)
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')   # small, fast, good enough
txt = pd.read_sql("SELECT ID, Title, Description_Text FROM Issue WHERE ...", engine)
emb = model.encode((txt['Title'].fillna('') + '. ' +
                    txt['Description_Text'].fillna('')).tolist(),
                   show_progress_bar=True, batch_size=64)
# emb is (n, 384); reduce with PCA to ~30 dims before feeding the model
from sklearn.decomposition import PCA
emb30 = PCA(n_components=30, random_state=0).fit_transform(emb)
```
**Note on your disk/RAM:** encoding 458k texts is heavy. Encode only the
issues you're modeling (resolved subset), batch it, and cache `emb30` to disk.
If it's too much, encode a per-project sample.

---

## PHASE 7 — The static model (foundation, build first)

A one-shot predictor: given features at creation, predict `delayed`. This is
your baseline and must work before dynamic.

### 7.1 Chronological split (NOT random)
```python
res_sorted = res.sort_values('Creation_Date')
cut = int(len(res_sorted)*0.8)
train_idx = res_sorted.index[:cut]; test_idx = res_sorted.index[cut:]
```
**Why chronological:** predicting delay is predicting the *future*. A random
split lets the model peek at future issues to predict past ones — unrealistic
and inflates scores. Train on older issues, test on newer. This is the
methodologically correct choice and a point in your favour in the paper.

### 7.2 Train gradient boosting
```python
from xgboost import XGBClassifier
X_train, y_train = feat.loc[train_idx].drop(columns=['delayed','ID']), feat.loc[train_idx,'delayed']
X_test,  y_test  = feat.loc[test_idx ].drop(columns=['delayed','ID']), feat.loc[test_idx ,'delayed']

clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss')
clf.fit(X_train, y_train)
```
**Why gradient boosting:** tree ensembles dominate on tabular issue data (your
lit review), beat deep learning here, handle mixed features, need little
tuning. CatBoost is a fine alternative (handles categoricals natively).

### 7.3 Evaluate
```python
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
proba = clf.predict_proba(X_test)[:,1]
pred = (proba>0.5).astype(int)
print("AUC:", roc_auc_score(y_test, proba))
print(precision_recall_fscore_support(y_test, pred, average='binary'))
print(confusion_matrix(y_test, pred))
```
**Metrics that matter:** AUC (ranking quality), precision/recall/F1 (with
imbalance, don't trust accuracy alone). Compare against a dumb baseline
(predict majority class) to prove the model adds value.

---

## PHASE 8 — The dynamic model (your novelty)

Static predicts once. Dynamic predicts **repeatedly as the task progresses**,
re-forecasting as new information arrives. Built from `Change_Log` timestamps.

### 8.1 Reconstruct each task's timeline
For each issue, pull its ordered status changes:
```python
timeline = pd.read_sql("""SELECT Issue_ID, From_String, To_String, Creation_Date
FROM Change_Log WHERE Change_Type='STATUS' ORDER BY Issue_ID, Creation_Date""", engine)
```
Each issue now has a sequence of (status, timestamp) events.

### 8.2 Create snapshots (the core idea)
For each issue, create several rows — one per "checkpoint" in its life
(e.g. at creation, at first status change, at 50% of elapsed time). Each
snapshot's features use **only what was known up to that moment**:
- time elapsed so far
- current status
- number of status changes so far
- assignee's concurrent load *at that moment*
- number of comments so far (from `Comment` table, filtered by date)
- whether any blocking issue was delayed *as of that moment*

Each snapshot carries the same final label (`delayed`), but different
feature values reflecting how much was known. The model learns to predict the
outcome earlier and earlier, and its confidence updates over time — that's
"dynamic re-forecasting."

### 8.3 Train on snapshots
Same gradient-boosting model, trained on the snapshot table instead of the
one-row-per-issue table. Evaluate how accuracy improves as the task
progresses (accuracy at 10% elapsed vs 50% vs 80%) — this plot is a headline
result for your paper.

**Bayesian element (Kula's method):** if you implement the Bayesian layer,
it models the delay probability as a distribution that updates at each
snapshot. Realistic library: **PyMC** or **Stan (CmdStanPy)**. Keep this
softer in scope — a well-built snapshot + gradient-boosting model already
demonstrates dynamic prediction; the Bayesian layer is the "closest to Kula"
refinement. Confirm Kula's exact approach from their replication package
(figshare.com/s/4672f25236520a2b4428) before committing.

---

## PHASE 9 — The generalization experiment (your core contribution)

The research question: does this work across projects, not just one?

### 9.1 Cross-project hold-out (38-vs-39)
```python
projects = res['Project_ID'].unique()
results = []
for held_out in projects:
    train = feat[feat['Project_ID'] != held_out]
    test  = feat[feat['Project_ID'] == held_out]
    # train model on 38 projects, test on the 1 held-out
    # record AUC/F1 for held_out
```
Loop over projects (or a representative sample). Each iteration trains on 38,
tests on the 39th it has never seen.

**What it shows:** if performance holds on unseen projects, the method
*generalizes*. If it collapses, that's *also* a finding (delay prediction is
project-specific) — either way you have a publishable result.

### 9.2 Report
- Table: per-project held-out AUC/F1.
- Compare within-project performance vs cross-project (the drop = the
  "generalization gap").
- Compare against Kula's published single-company numbers.

---

## PHASE 10 — Explainability (SHAP)

Turns predictions into reasons — your "explainable" pillar.
```python
import shap
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)          # global: which features matter
shap.plots.waterfall(explainer(X_test.iloc[0])) # local: why THIS task
```
**Use it two ways:**
- **Global** summary plot → for the paper ("workload and dependencies are the
  top delay drivers").
- **Local** per-issue → for the prototype's "why is this at risk" panel, and
  it feeds the reassignment suggester's honesty check.

---

## PHASE 11 — Reassignment suggester (stretch, request-based)

On-demand only (not automatic). For a flagged issue:
1. Take its current features.
2. For each candidate developer, swap in *their* workload/history features.
3. Re-run the model → predicted risk per candidate.
4. Rank; suggest the lowest-risk assignee.
5. **Honesty check via SHAP:** if the delay is driven by a blocking
   dependency (not the assignee), report that reassignment won't help instead
   of suggesting a swap.

Build this last — it depends on everything above working.

---

## PHASE 12 — The Streamlit prototype

Wrap the trained model in the interface. New views of existing predictions;
no new machinery.
```python
# app/app.py
import streamlit as st, pandas as pd, joblib
model = joblib.load('models/xgb.joblib')
# tabs: Dashboard | Task Risk & Explanations | Team Workload | How It Works
```
Views: project risk dashboard (at-risk vs on-track + top-5), filterable task
list, click-to-expand SHAP, developer workload, reassignment suggester.
Run with `streamlit run app/app.py`.

**Scope discipline:** no live Jira sync, no task editing, no notifications —
those are future work. The prototype *shows and explains* predictions.

---

## Build order (recommended)

1. **EDA** (Phases 1–5) — answer the decision questions. *Do not skip.*
2. **Labeling + features** (Phase 6) — build the feature table, no leakage.
3. **Static model** (Phase 7) — baseline that works end to end.
4. **Dynamic model** (Phase 8) — snapshots; your novelty.
5. **Generalization** (Phase 9) — the core research result.
6. **SHAP** (Phase 10) — explanations for paper + prototype.
7. **Prototype** (Phase 12) — the demo.
8. **Reassignment** (Phase 11) — last, if time allows.

Each stage produces a complete, presentable result on its own. If time runs
short, you stop at any stage with something defensible.

---

## Pitfalls to avoid (learned the hard way)

- **Leakage** — never feed post-resolution info as a feature. #1 way capstones
  get invalidated.
- **Random split** — use chronological. A random split inflates scores and is
  wrong for a forecasting task.
- **Trusting accuracy** — with class imbalance, report AUC + F1, compare to a
  majority-class baseline.
- **Over-scoping** — the model + generalization finding is the contribution;
  don't drift into building a product.
- **Disk space** — cache to disk, watch free space, don't re-query 458k rows
  needlessly.
- **Label definition unstated** — always state exactly how "delayed" is
  defined; it's the first thing reviewers check.
- **Bayesian over-commitment** — confirm Kula's exact method from their
  package before promising a specific implementation.

---

## What to have ready for the viva

- Why per-project median for the label (fairness across teams).
- Why chronological split (forecasting realism).
- Why gradient boosting (tabular data; beats DL — cite Grinsztajn 2022 +
  your lit review).
- What "dynamic" means here (snapshots + re-forecasting from Change_Log).
- The generalization result (does it transfer across projects — the headline).
- Honest limitations (label is a proxy; cross-project drop; Bayesian layer
  scope).
