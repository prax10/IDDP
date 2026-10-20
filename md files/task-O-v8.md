# Task O: v8, more known-at-T signal

Branch `v7-honest`. Scripts `task_o1_v8_features.py`, `task_o2_v8_model.py`, `task_o3_select.py`,
`task_o4_blend.py`. Reports `o2_val_ablation.csv`, `o3_v8_selection.csv`, `o4_blend_val.csv`,
`o5_results_v6_v7_v8.csv`, `o5_by_checkpoint.csv`, `o5_v8_vs_v7.csv`, `n3_clearcut_v8.csv`.
All choices on train → validation; test scored once.

## Headline

**v8 test AUC 0.869 (5-model ensemble 0.870)**, up from v7's 0.867: +0.0018 [+0.0015, +0.0020], and
+0.0045 at M1. Every feature is known at the checkpoint.

| Test (scored once) | v7 | **v8** |
|---|---|---|
| AUC overall (seed mean / 5-model ensemble) | 0.867 / 0.868 | **0.869 / 0.870** |
| Early warning, M1–M3 | 0.821 | **0.824** |
| Late stage, M5–M10 | 0.891 | **0.892** |
| Final stretch, M8–M10 | 0.904 | **0.904** |
| M1 / M8 / M10 | 0.806 / 0.896 / 0.911 | **0.811 / 0.897 / 0.911** |
| Outcome still open | 0.837 | **0.839** |
| Censoring-safe + outcome open | 0.826 | **0.828** |
| Top-10% flag precision, outcome open / at M1 | 0.982 / 0.940 | **0.982 / 0.947** |
| Accuracy / balanced accuracy @0.5 | 0.808 / 0.735 | **0.808 / 0.736** |
| Clear-cut task (secondary): all rows / undecided rows | 0.924 / 0.881 | **0.926 / 0.885** |

## What was added

| Feature group | Val AUC (1 seed, on v7) | Kept |
|---|---|---|
| v7 | 0.7703 | |
| Priority and type **as of T** (rebuilt from the change log; differ from the final value on 15% / 14% of rows) + priority raises before T | 0.7734 (+0.003) | ✅ |
| + **epic progress at T** (epic size, share of its issues done, recent resolutions, age, velocity-based days left) | 0.7768 (+0.0034) | ✅ |
| + similar past issues (20 nearest SBERT neighbours resolved before creation: late rate, duration ratio) | 0.7740 (+0.0006) | ❌ |
| + reporter recent form (90 days) | 0.7740 (+0.0006) | ❌ |
| + component late rate at T | 0.7732 (−0.0002) | ❌ |

Final set chosen with 3 validation seeds (`o3_v8_selection.csv`): priority/type at T + epic progress
0.7769 ± 0.0002, ahead of every larger combination (0.7760–0.7765).

- **Priority and type at T help.** Using the value as of the checkpoint beats the final value, so the
  final values were noise here, not a leak.
- **Epic progress is the strongest new signal**, the same idea as fix-version progress in v7: an issue
  in an epic that is moving slowly is likely to be late.
- **Blending with an MLP hurts** (val 0.7773 → 0.7761), so v8 stays a single XGBoost with exact SHAP.

## Paper / viva lines

- "Every feature is computed from information available at the checkpoint, including the assignee,
  priority, type, links, fix version and epic as they stood at that moment."
- Headline: overall AUC **0.869** (0.870 for the 5-model ensemble); **0.904** from 80% of expected
  duration onward; top-10% flags 98% precise.
- Secondary: clearly late vs clearly on-time issues, borderline excluded, **0.926** (0.885 on rows whose
  outcome is not yet decided).

Models: `models/xgb_v8_seed{0..4}.json`, `xgb_v8_meta.json` (the app still loads v6).
