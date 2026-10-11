# Evaluation (held-out test split)

**Headline: attack-class F1 = 0.91 (95% bootstrap CI 0.88 to 0.93)**
Quote the lower bound (0.88) if you need one number.

> WARNING: result looks too good (F1>=0.99, zero FPR, or <30 test attacks). Check for duplicates/leakage before quoting it.

- Dataset: `Botnet-Friday-no-metadata.csv`; rows train/val/test = 105622/35208/35208
- Method: stratified 60/20/20, seed 42; scaler/selection/models fit on train, fusion on val, metrics on test
- Operating point: fused Isolation Forest + Random Forest, meta-learner, threshold 0.99 tuned on validation only

| | Pred normal | Pred attack |
|---|---|---|
| **Actual normal** | 34921 | 0 |
| **Actual attack** | 49 | 238 |

Not claimed: zero false positives, near-perfect accuracy, or performance on traffic from other networks.
Single dataset, single seed; benchmark flows are not live traffic.
