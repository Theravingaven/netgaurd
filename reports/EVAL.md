# Evaluation (random stratified held-out test split)

**Headline: attack-class F1 = 0.99 (95% bootstrap CI 0.99 to 1.00)**
Quote the lower bound (0.99) if you need one number.

> WARNING: result looks too good (F1>=0.99, zero FPR, or <30 test attacks). Check for duplicates/leakage before quoting it.

- Dataset: `Portscan-Friday-no-metadata.csv`; rows train/val/test = 71713/23904/23905
- Method: stratified 60/20/20, seed 42; scaler/selection/models fit on train, fusion and threshold on val, metrics on test
- Operating point: fused Isolation Forest + Random Forest, meta-learner, threshold 0.92 tuned on validation only

| | Pred normal | Pred attack |
|---|---|---|
| **Actual normal** | 23514 | 0 |
| **Actual attack** | 6 | 385 |

Not claimed: zero false positives, near-perfect accuracy, or performance on traffic from other networks.
Single dataset, single seed; benchmark flows are not live traffic.
