# Evaluation (chronological held-out test split)

**Headline: attack-class F1 = 0.88 (95% bootstrap CI 0.85 to 0.91)**
Quote the lower bound (0.85) if you need one number.

- Dataset: `Portscan-Friday-no-metadata.csv`; rows train/val/test = 71713/23904/23905
- Method: chronological 60/20/20 in file order; scaler/selection/models fit on train, fusion and threshold on val, metrics on test
- Operating point: fused Isolation Forest + Random Forest, meta-learner, threshold 0.98 tuned on validation only

| | Pred normal | Pred attack |
|---|---|---|
| **Actual normal** | 23565 | 2 |
| **Actual attack** | 69 | 269 |

Not claimed: zero false positives, near-perfect accuracy, or performance on traffic from other networks.
Single dataset, single seed; benchmark flows are not live traffic.
