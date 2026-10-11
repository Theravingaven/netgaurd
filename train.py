"""
train.py: honest, reproducible training for NetGuard IDS.

Usage:  python train.py path/to/data.csv [--top-n 15]

- Stratified 60/20/20 train/val/test split (before ANY fitting)
- Scaler, feature selection, models fit on TRAIN only
- Meta-learner (fusion) fit on VALIDATION only
- All reported metrics come from the untouched TEST split
- Saves models in the same format app.py / capture.py already load
- Writes reports/metrics.json
"""
import sys, json, argparse
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, precision_recall_curve)
from utils import (ensure_dirs, save_obj, ISO_PATH, RF_PATH, SCALER_PATH,
                   RL_PATH, REPORTS)
import os

SEED = 42

def map_attack(x):
    x = str(x).upper()
    if "BENIGN" in x or "NORMAL" in x: return "NORMAL"
    if "DOS" in x or "DDOS" in x:      return "DDOS"
    if "SCAN" in x or "PROBE" in x:    return "SCAN"
    if "BRUTE" in x:                   return "BRUTEFORCE"
    return "OTHER"

def load_clean(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    label_col = next((c for c in df.columns
                      if c.lower() in ("label", "prediction", "class", "target")), None)
    if label_col is None:
        raise ValueError("No label column found (expected label/class/target/prediction).")
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")
    df["attack_type"] = df[label_col].astype(str).apply(map_attack)
    df = df.drop(columns=[label_col])
    df["label"] = (df["attack_type"] != "NORMAL").astype(int)
    for c in df.columns:
        if c not in ("label", "attack_type") and df[c].dtype == object:
            df[c] = pd.Categorical(df[c]).codes
    return df

def main(path, top_n):
    ensure_dirs()
    df = load_clean(path)
    feat_cols = [c for c in df.columns if c not in ("label", "attack_type")]

    # 1. split FIRST (stratified on attack type)
    strat = df["attack_type"]
    train, tmp = train_test_split(df, test_size=0.4, stratify=strat, random_state=SEED)
    val, test = train_test_split(tmp, test_size=0.5, stratify=tmp["attack_type"], random_state=SEED)

    # 2. impute + scale with TRAIN statistics only
    med = train[feat_cols].median()
    for part in (train, val, test):
        part[feat_cols] = part[feat_cols].fillna(med)
    scaler = StandardScaler().fit(train[feat_cols])
    save_obj(scaler, SCALER_PATH)
    tr = pd.DataFrame(scaler.transform(train[feat_cols]), columns=feat_cols, index=train.index)
    va = pd.DataFrame(scaler.transform(val[feat_cols]),   columns=feat_cols, index=val.index)
    te = pd.DataFrame(scaler.transform(test[feat_cols]),  columns=feat_cols, index=test.index)

    # 3. feature selection on TRAIN only
    vt = VarianceThreshold(0.01).fit(tr)
    keep = list(tr.columns[vt.get_support()])
    corr = tr[keep].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    keep = [c for c in keep if not any(upper[c] > 0.95)]
    probe = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=SEED, n_jobs=-1)
    probe.fit(tr[keep], train["label"])
    imp = pd.Series(probe.feature_importances_, index=keep).sort_values(ascending=False)
    top = imp.head(top_n).index.tolist()

    # 4. models (train only)
    iso = IsolationForest(n_estimators=300, max_samples=256, random_state=SEED, n_jobs=-1)
    iso.fit(tr.loc[train["label"] == 0, top])
    rf = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight="balanced",
                                random_state=SEED, n_jobs=-1)
    rf.fit(tr[top], train["attack_type"])
    save_obj(iso, ISO_PATH); save_obj(rf, RF_PATH)

    # 5. fusion meta-learner fit on VALIDATION only
    classes = list(rf.classes_)
    n_idx = classes.index("NORMAL")
    iso_va = iso.decision_function(va[top])
    s_min, s_max = float(iso_va.min()), float(iso_va.max())
    def meta_X(X):
        s = np.clip((iso.decision_function(X) - s_min) / (s_max - s_min + 1e-9), 0, 1)
        p = 1.0 - rf.predict_proba(X)[:, n_idx]
        return np.column_stack([s, p])
    meta = LogisticRegression(class_weight="balanced", max_iter=500, random_state=SEED)
    meta.fit(meta_X(va[top]), val["label"])
    # decision threshold chosen on VALIDATION only (max F1), never on test
    pv, rv, tv = precision_recall_curve(val["label"], meta.predict_proba(meta_X(va[top]))[:, 1])
    f1v = 2 * pv[:-1] * rv[:-1] / (pv[:-1] + rv[:-1] + 1e-9)
    thr = float(tv[int(np.argmax(f1v))]) if len(tv) else 0.5
    nmask = (train["label"] == 0).values
    ref_mean = tr.loc[nmask, top].mean().tolist()
    ref_std = (tr.loc[nmask, top].std() + 1e-9).tolist()
    save_obj({"meta": meta, "s_min": s_min, "s_max": s_max,
              "classes": classes, "normal_idx": n_idx,
              "threshold": thr, "features": top, "ref_mean": ref_mean, "ref_std": ref_std}, RL_PATH)

    # 6. evaluate on untouched TEST
    Xm = meta_X(te[top])
    y = test["label"].values
    score = meta.predict_proba(Xm)[:, 1]
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    f1 = f1_score(y, pred, zero_division=0)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        boots.append(f1_score(y[i], pred[i], zero_division=0))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    fpr = fp / (fp + tn + 1e-9)
    suspicious = bool(f1 >= 0.99 or fpr == 0 or (tp + fn) < 30)
    metrics = {
        "dataset": os.path.basename(path),
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "headline": {"metric": "attack-class F1 (test)", "value": round(f1, 4),
                     "ci95": [round(float(lo), 4), round(float(hi), 4)]},
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "auc": round(roc_auc_score(y, score), 4) if len(set(y)) > 1 else None,
        "features_used": top,
        "suspicious_result": suspicious,
        "split": "stratified 60/20/20, seed 42; scaler/selection/models fit on train, fusion on val, metrics on test",
    }
    with open(os.path.join(REPORTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    warn = ("\n> WARNING: result looks too good (F1>=0.99, zero FPR, or <30 test attacks). "
            "Check for duplicates/leakage before quoting it.\n") if suspicious else ""
    md = f"""# Evaluation (held-out test split)

**Headline: attack-class F1 = {f1:.2f} (95% bootstrap CI {lo:.2f} to {hi:.2f})**
Quote the lower bound ({lo:.2f}) if you need one number.
{warn}
- Dataset: `{os.path.basename(path)}`; rows train/val/test = {len(train)}/{len(val)}/{len(test)}
- Method: {metrics["split"]}
- Operating point: fused Isolation Forest + Random Forest, meta-learner, threshold {thr:.2f} tuned on validation only

| | Pred normal | Pred attack |
|---|---|---|
| **Actual normal** | {tn} | {fp} |
| **Actual attack** | {fn} | {tp} |

Not claimed: zero false positives, near-perfect accuracy, or performance on traffic from other networks.
Single dataset, single seed; benchmark flows are not live traffic.
"""
    open(os.path.join(REPORTS, "EVAL.md"), "w").write(md)
    print(md)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--top-n", type=int, default=15)
    a = ap.parse_args()
    main(a.csv, a.top_n)
