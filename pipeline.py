"""
pipeline.py — NetGuard IDS v3.0
Models: Isolation Forest (unsupervised) + Random Forest (supervised)
        + a Logistic Regression meta-learner that fuses the two.

Evaluation methodology
-----------------------
Every prior version of this pipeline trained and evaluated on the same rows,
which is why it reported ~100% accuracy — the model was grading its own
memorized homework. This version enforces a strict split:

  TRAIN  -> fits the Isolation Forest, Random Forest, and meta-learner.
            Attack rows may be oversampled here to balance classes.
  VAL    -> used only to pick the anomaly-score threshold (via a small
            Q-learning search). Never used to fit any model weights.
  TEST   -> touched for the first time at evaluate() time. Every reported
            metric (precision, recall, F1, AUC, confusion matrix) comes
            from this held-out split alone.

If a dataset has no ground-truth label column, there's nothing to hold out
or score against, so the pipeline falls back to a single unlabeled scoring
pass and the UI is expected to hide the evaluation panels accordingly.
"""
import os, io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, roc_auc_score)
from utils import (save_obj, load_obj, ensure_dirs,
                    TRAIN_PROCESSED, VAL_PROCESSED, TEST_PROCESSED,
                    TRAIN_FEATURES, VAL_FEATURES, TEST_FEATURES,
                    ISO_PATH, RF_PATH, SCALER_PATH, RL_PATH,
                    RESULTS_PATH, REPORTS)

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

# Palette matched to the frontend design tokens (see static/css/style.css)
DARK  = "0D1013"   # --ink
PANEL = "15191D"   # --panel
ACC   = "#5D7FA6"  # --signal-focus (steady slate blue)
GRN   = "#4FAE83"  # --signal-ok
WARN  = "#C4534A"  # --signal-bad
AMBER = "#D69A3B"  # --signal-warn
MUTED = "#97A0A8"  # --ink-soft

def _style(fig, ax):
    fig.patch.set_facecolor("#" + DARK)
    axes = np.array(ax).flatten() if isinstance(ax, (list, np.ndarray)) else [ax]
    for a in axes:
        a.set_facecolor("#" + PANEL)
        a.tick_params(colors=MUTED, labelsize=9)
        for spine in a.spines.values():
            spine.set_edgecolor("#232B30")

def _map_attack(x):
    if "BENIGN" in x or "NORMAL" in x:
        return "NORMAL"
    elif "DOS" in x or "DDOS" in x:
        return "DDOS"
    elif "SCAN" in x or "PROBE" in x:
        return "SCAN"
    elif "BRUTE" in x:
        return "BRUTEFORCE"
    return "OTHER"


def preprocess(filepath, progress_cb=None):
    ensure_dirs()
    def log(msg):
        if progress_cb: progress_cb(msg)

    log("Loading CSV file...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    log(f"Loaded {len(df):,} rows x {df.shape[1]} columns")

    before = len(df)
    df.drop_duplicates(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    log(f"Removed {before - len(df):,} duplicate rows")

    label_col = None
    for c in df.columns:
        if c.lower() in ["label", "prediction", "class", "target"]:
            label_col = c; break

    has_labels = label_col is not None
    if has_labels:
        df[label_col] = df[label_col].astype(str).str.upper().str.strip()
        df["attack_type"] = df[label_col].apply(_map_attack)
        df["label"] = df["attack_type"].apply(lambda x: 0 if x == "NORMAL" else 1)
        df.drop(columns=[label_col], inplace=True)
    else:
        df["attack_type"] = "UNKNOWN"
        df["label"] = -1
        log("No label column found — running in unlabeled detection mode (no held-out metrics).")

    for c in df.columns:
        if c in ["label", "attack_type"]:
            continue
        if df[c].dtype == object:
            df[c] = pd.Categorical(df[c]).codes

    feat_cols = [c for c in df.columns if c not in ["label", "attack_type"]]

    if has_labels and df["label"].nunique() > 1:
        log("Splitting into train / validation / test — the model will never see val or test during fitting.")
        try:
            train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)
        except ValueError:
            train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    else:
        train_df, val_df, test_df = df, df.iloc[0:0].copy(), df.iloc[0:0].copy()

    # Balance ONLY the training fold — val/test keep real-world class prevalence
    # so the metrics we report later reflect reality, not an inflated ratio.
    normal_df = train_df[train_df["label"] == 0]
    attack_df = train_df[train_df["label"] == 1]
    if has_labels and len(attack_df) > 0 and len(attack_df) < len(normal_df) * 0.20:
        target = int(len(normal_df) * 0.20)
        attack_df = attack_df.sample(target, replace=True, random_state=42)
        train_df = pd.concat([normal_df, attack_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        log(f"Balanced training fold — Normal: {(train_df['label']==0).sum():,}  Attack: {(train_df['label']==1).sum():,}")

    scaler = StandardScaler()
    train_df = train_df.copy()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    val_df, test_df = val_df.copy(), test_df.copy()
    if len(val_df):
        val_df[feat_cols] = scaler.transform(val_df[feat_cols])
    if len(test_df):
        test_df[feat_cols] = scaler.transform(test_df[feat_cols])
    save_obj(scaler, SCALER_PATH)

    train_df.to_csv(TRAIN_PROCESSED, index=False)
    val_df.to_csv(VAL_PROCESSED, index=False)
    test_df.to_csv(TEST_PROCESSED, index=False)

    log(f"Preprocessing done — train {len(train_df):,} / val {len(val_df):,} / test {len(test_df):,}")

    return {
        "rows": len(df), "cols": df.shape[1],
        "normal": int((df["label"] == 0).sum()) if has_labels else None,
        "attack": int((df["label"] == 1).sum()) if has_labels else None,
        "has_labels": has_labels,
        "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df),
    }


def select_features(top_n=15, progress_cb=None):
    def log(msg):
        if progress_cb: progress_cb(msg)

    train_df = pd.read_csv(TRAIN_PROCESSED)
    val_df   = pd.read_csv(VAL_PROCESSED)
    test_df  = pd.read_csv(TEST_PROCESSED)

    X = train_df[[c for c in train_df.columns if c not in ["label", "attack_type"]]]
    y = train_df["label"]

    vt = VarianceThreshold(threshold=0.01)
    vt.fit(X)
    X = X[X.columns[vt.get_support()]]

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    X = X.drop(columns=[c for c in upper.columns if any(upper[c] > 0.95)])

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=8)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top = imp.head(top_n).index.tolist()

    log(f"Top features (train-only importance): {', '.join(top[:5])}...")

    fig, ax = plt.subplots(figsize=(7, 4)); _style(fig, ax)
    imp.head(top_n).plot(kind="barh", ax=ax, color=ACC)
    ax.set_title("Top Feature Importances", color=ACC, fontsize=12, pad=10)
    ax.set_xlabel("Importance", color=MUTED); ax.invert_yaxis(); ax.tick_params(colors=MUTED)
    plt.tight_layout(); chart = _fig_to_b64(fig); plt.close()

    cols = top + ["label", "attack_type"]
    train_df[cols].to_csv(TRAIN_FEATURES, index=False)
    val_df[cols].to_csv(VAL_FEATURES, index=False)
    test_df[cols].to_csv(TEST_FEATURES, index=False)

    log(f"Feature selection done — kept {top_n} features across all splits.")
    return {"features": top, "chart": chart}


def train_models(progress_cb=None):
    def log(msg):
        if progress_cb: progress_cb(msg)

    df = pd.read_csv(TRAIN_FEATURES)
    feat_cols = [c for c in df.columns if c not in ["label", "attack_type"]]

    X_all = df[feat_cols]
    y_all = df["label"]
    y_multi = df["attack_type"]
    X_normal = df[df["label"] == 0][feat_cols]
    if len(X_normal) == 0:
        raise ValueError("No normal samples found in the training fold.")

    log(f"Training on {len(X_normal):,} normal + {int((y_all==1).sum()):,} attack samples (train fold only)")

    log("Training Isolation Forest...")
    # max_samples: Isolation Forest doesn't need huge per-tree samples to work — its whole
    # mechanism relies on random partitioning isolating outliers in *few* splits, which
    # happens reliably even with small subsamples. The original paper's own recommended
    # default is 256 samples per tree. The previous max_samples=0.8 meant building 500 trees
    # each from ~80% of the full dataset — fine on a local machine, but heavy enough to stall
    # or get OOM-killed on a memory-constrained free-tier host. 256 is faster and, per the
    # Isolation Forest paper, typically detects anomalies just as well or better.
    iso = IsolationForest(n_estimators=150, contamination=0.1, max_samples=256, random_state=42, n_jobs=-1)
    iso.fit(X_normal)
    save_obj(iso, ISO_PATH)

    if (y_all == 1).sum() > 0:
        log("Training Random Forest classifier...")
        # Trimmed from 300 trees / depth 20 — very deep trees over ~78 features and 200k+ rows
        # multiply memory fast. 150 trees at depth 14 trains noticeably lighter with only a
        # small, usually negligible, accuracy cost for this kind of tabular flow data.
        rf = RandomForestClassifier(n_estimators=150, max_depth=14, class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_all, y_multi)
        save_obj(rf, RF_PATH)
        log("Random Forest (multiclass) saved")
    else:
        log("No attack samples in training data — skipping Random Forest, Isolation Forest only.")

    log("Training meta-learner fusion agent...")
    train_meta_learner(progress_cb=progress_cb)

    log("Tuning anomaly threshold on the validation split...")
    select_threshold(progress_cb=progress_cb)

    log("All models trained and saved.")
    return {"status": "ok", "normal_samples": len(X_normal)}


def train_meta_learner(progress_cb=None):
    """Stacking meta-learner: Logistic Regression on [iso_score, rf_attack_proba] -> label.
    Fit on the TRAIN fold only."""
    from sklearn.linear_model import LogisticRegression
    def log(msg):
        if progress_cb: progress_cb(msg)

    df = pd.read_csv(TRAIN_FEATURES)
    feat_cols = [c for c in df.columns if c not in ["label", "attack_type"]]
    y_true = df["label"].values

    iso = load_obj(ISO_PATH)
    try:
        rf = load_obj(RF_PATH)
    except FileNotFoundError:
        log("[META] No Random Forest available — skipping meta-learner.")
        save_obj({"meta": None, "s_min": None, "s_max": None, "classes": [], "normal_idx": 0}, RL_PATH)
        return {"status": "skipped"}

    iso_scores = iso.decision_function(df[feat_cols])
    s_min, s_max = iso_scores.min(), iso_scores.max()
    iso_norm = (iso_scores - s_min) / (s_max - s_min + 1e-9)

    X_rf = df[feat_cols].reindex(columns=rf.feature_names_in_, fill_value=0)
    classes = list(rf.classes_)
    normal_idx = classes.index("NORMAL") if "NORMAL" in classes else 0
    rf_attack_proba = 1.0 - rf.predict_proba(X_rf)[:, normal_idx]

    if len(np.unique(y_true)) < 2:
        log("[META] Only one class present in training data — skipping meta-learner (falls back to OR logic at detect time).")
        save_obj({"meta": None, "s_min": s_min, "s_max": s_max, "classes": classes, "normal_idx": normal_idx}, RL_PATH)
        return {"status": "skipped"}

    X_meta = np.column_stack([iso_norm, rf_attack_proba])
    meta = LogisticRegression(class_weight="balanced", random_state=42, max_iter=500)
    meta.fit(X_meta, y_true)

    save_obj({"meta": meta, "s_min": s_min, "s_max": s_max, "classes": classes, "normal_idx": normal_idx}, RL_PATH)
    log("[META] Meta-learner trained on the training fold.")
    return {"status": "ok"}


def _rl_find_threshold(iso_scores, y_true=None, episodes=40, progress_cb=None):
    """
    Q-learning agent that searches for the ISO Forest threshold percentile
    that maximizes a weighted reward (recall, 1-FPR, agreement, separation).
    Falls back to the 20th percentile if nothing beats it.
    Caller decides which score/label set this runs against — it should
    always be the VALIDATION split, never the one metrics get reported on.
    """
    def log(msg):
        if progress_cb: progress_cb(msg)

    ACTIONS   = [5, 10, 15, 20, 25, 30, 35, 40]
    N_STATES  = 5
    FALLBACK  = 20
    ALPHA     = 0.3
    GAMMA     = 0.85
    EPSILON   = 0.9
    EPS_DECAY = 0.97

    Q = np.zeros((N_STATES, len(ACTIONS)))
    s_min, s_max = iso_scores.min(), iso_scores.max()
    iso_norm = (iso_scores - s_min) / (s_max - s_min + 1e-9)

    def _get_state():
        low = iso_norm[iso_norm < np.percentile(iso_norm, 20)]
        mean_low = float(np.mean(low)) if len(low) > 0 else 0.0
        return min(int(mean_low * N_STATES), N_STATES - 1)

    def _reward(pct):
        threshold = np.percentile(iso_scores, pct)
        preds = np.where(iso_scores < threshold, -1, 1)
        if y_true is not None:
            y_p = (preds == -1).astype(int)
            tp = int(((y_p == 1) & (y_true == 1)).sum())
            fp = int(((y_p == 1) & (y_true == 0)).sum())
            fn = int(((y_p == 0) & (y_true == 1)).sum())
            tn = int(((y_p == 0) & (y_true == 0)).sum())
            rec = tp / (tp + fn + 1e-9)
            fpr = fp / (fp + tn + 1e-9)
            agr = max(preds[preds == 1].size, preds[preds == -1].size) / len(preds)
            sep = float(np.mean(np.abs(iso_norm - 0.5) * 2))
            return float((rec * 0.45) + ((1 - fpr) * 0.30) + (agr * 0.15) + (sep * 0.10))
        anom_rate = (preds == -1).sum() / len(preds)
        return float(1.0 - abs(anom_rate - 0.15) * 3)

    epsilon = EPSILON
    best_pct, best_reward = FALLBACK, _reward(FALLBACK)
    reward_hist = []
    log(f"[THRESHOLD] Searching on validation data — fallback={FALLBACK}th pct, baseline reward={best_reward:.4f}")

    for _ in range(episodes):
        state = _get_state()
        action_idx = np.random.randint(len(ACTIONS)) if np.random.rand() < epsilon else int(np.argmax(Q[state]))
        pct = ACTIONS[action_idx]
        reward = _reward(pct)
        reward_hist.append((pct, reward))
        Q[state, action_idx] += ALPHA * (reward + GAMMA * np.max(Q[state]) - Q[state, action_idx])
        if reward > best_reward:
            best_reward, best_pct = reward, pct
        epsilon *= EPS_DECAY

    fallback_reward = _reward(FALLBACK)
    if best_reward <= fallback_reward:
        log(f"[THRESHOLD] No improvement over fallback — keeping {FALLBACK}th percentile.")
        best_pct = FALLBACK
    else:
        log(f"[THRESHOLD] Best found: {best_pct}th percentile (reward={best_reward:.4f} vs fallback={fallback_reward:.4f})")

    return best_pct, best_reward, reward_hist


def select_threshold(progress_cb=None):
    """Freezes an anomaly-score threshold using the VALIDATION split only.
    detect() later applies this frozen value — it never re-derives a
    threshold from the data it's about to report metrics on."""
    def log(msg):
        if progress_cb: progress_cb(msg)

    val_df = pd.read_csv(VAL_FEATURES)
    agent = load_obj(RL_PATH)

    if len(val_df) == 0:
        log("[THRESHOLD] No validation split available — using default 20th percentile.")
        agent.update({"rl_pct": 20, "rl_reward": 0.0, "threshold_value": None})
        save_obj(agent, RL_PATH)
        return {"rl_pct": 20, "rl_reward": 0.0}

    feat_cols = [c for c in val_df.columns if c not in ["label", "attack_type"]]
    iso = load_obj(ISO_PATH)
    iso_scores = iso.decision_function(val_df[feat_cols])
    y_true = val_df["label"].values if val_df["label"].nunique() > 1 else None

    rl_pct, rl_reward, rl_history = _rl_find_threshold(iso_scores, y_true=y_true, episodes=40, progress_cb=progress_cb)
    threshold_value = float(np.percentile(iso_scores, rl_pct))

    agent.update({
        "rl_pct": rl_pct, "rl_reward": round(rl_reward, 4),
        "rl_history": rl_history, "threshold_value": threshold_value,
    })
    save_obj(agent, RL_PATH)

    log(f"[THRESHOLD] Frozen at {rl_pct}th percentile of validation scores (value={threshold_value:.4f}).")
    return {"rl_pct": rl_pct, "rl_reward": rl_reward, "threshold_value": threshold_value}


def detect(filepath=None, progress_cb=None):
    def log(msg):
        if progress_cb: progress_cb(msg)

    iso = load_obj(ISO_PATH)
    rf = None
    try:
        rf = load_obj(RF_PATH); log("RF model loaded")
    except Exception:
        log("No RF model — Isolation Forest only")

    held_out = True
    if filepath:
        src, held_out = filepath, False
    else:
        test_preview = pd.read_csv(TEST_FEATURES)
        if len(test_preview) > 0:
            src = TEST_FEATURES
        else:
            src, held_out = TRAIN_FEATURES, False
            log("No held-out test split available — scoring training data instead (metrics may be optimistic).")

    df = pd.read_csv(src)
    drop_cols = [c for c in ["label", "attack_type"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    if rf is not None:
        X = X.reindex(columns=rf.feature_names_in_, fill_value=0)

    log(f"Analysing {len(X):,} flows...")
    iso_scores = iso.decision_function(X)
    s_min, s_max = iso_scores.min(), iso_scores.max()
    iso_scores_norm = np.zeros_like(iso_scores) if (s_max - s_min) == 0 else (iso_scores - s_min) / (s_max - s_min)

    agent_preview = {}
    try:
        agent_preview = load_obj(RL_PATH)
    except Exception:
        pass
    rl_pct = agent_preview.get("rl_pct")
    threshold = agent_preview.get("threshold_value")

    if threshold is None:
        rl_pct = rl_pct or 20
        threshold = float(np.percentile(iso_scores, rl_pct))
        log(f"[THRESHOLD] No tuned threshold on file — using {rl_pct}th percentile of this data as a fallback.")
    else:
        log(f"[THRESHOLD] Using threshold tuned on validation data ({rl_pct}th percentile, value={threshold:.4f}).")

    iso_preds = np.where(iso_scores < threshold, -1, 1)

    if rf is not None:
        attack_type = rf.predict(X)
        rf_conv = np.where(attack_type == "NORMAL", 1, -1)
        rf_preds = rf_conv
        classes = list(rf.classes_)
        normal_idx = classes.index("NORMAL") if "NORMAL" in classes else 0
        rf_attack_proba = 1.0 - rf.predict_proba(X)[:, normal_idx]
    else:
        attack_type = ["UNKNOWN"] * len(X)
        rf_preds = np.zeros(len(X), dtype=int)
        rf_conv = np.ones(len(X), dtype=int)
        rf_attack_proba = np.zeros(len(X))

    try:
        meta = agent_preview.get("meta")
        if meta is None:
            raise ValueError("no meta-learner available")
        a_min, a_max = agent_preview["s_min"], agent_preview["s_max"]
        iso_norm_ml = np.clip((iso_scores - a_min) / (a_max - a_min + 1e-9), 0, 1)
        X_meta = np.column_stack([iso_norm_ml, rf_attack_proba])
        meta_pred = meta.predict(X_meta)
        final = np.where(meta_pred == 1, -1, 1)
        log("[META] Meta-learner fusion applied.")
    except Exception as e:
        log(f"[META] Not available, falling back to OR logic: {e}")
        final = np.where((rf_conv == -1) | (iso_preds == -1), -1, 1)

    def _agr(f, r):
        if f == -1 and r == -1: return "High Confidence Anomaly"
        if f == 1 and r == 1: return "High Confidence Normal"
        if r == -1: return "RF Flagged"
        return "ISO Flagged"

    agreement_cats = [_agr(f, r) for f, r in zip(final, rf_conv)]
    agr_scores = []
    for i, r in zip(iso_preds, rf_conv):
        if i == r: agr_scores.append(1.0)
        elif i == -1 and r == 1: agr_scores.append(0.6)
        elif i == 1 and r == -1: agr_scores.append(0.7)
        else: agr_scores.append(0.5)
    agr_scores = np.array(agr_scores)

    sep_scores = np.round(np.abs(iso_scores_norm - 0.5) * 2, 4)
    row_accuracy_pct = np.round((agr_scores * 0.40 + sep_scores * 0.60) * 100, 1)

    results = pd.DataFrame({
        "anomaly_score": iso_scores, "row_accuracy_pct": row_accuracy_pct,
        "iso_pred": iso_preds, "rf_pred": rf_preds,
        "final_pred": final, "agreement": agreement_cats,
        "verdict": ["ANOMALY" if p == -1 else "NORMAL" for p in final],
        "attack_type": attack_type,
    })
    if "label" in df.columns and df["label"].nunique() > 1 and (df["label"] != -1).all():
        results["true_label"] = df["label"].values

    results.to_csv(RESULTS_PATH, index=False)

    total = len(results)
    anomalies = int((results["final_pred"] == -1).sum())
    normal = total - anomalies
    agr_rate = round(results["agreement"].isin(["High Confidence Anomaly", "High Confidence Normal"]).sum() / total * 100, 2) if total else 0

    n_sc = iso_scores_norm[final == 1]
    a_sc = iso_scores_norm[final == -1]
    sep = round(float(abs(n_sc.mean() - a_sc.mean())), 4) if len(n_sc) > 0 and len(a_sc) > 0 else 0.0

    # Compact histogram of the score distribution for the frontend's signature waveform —
    # sending 141k raw scores over polling JSON would be wasteful; a binned summary is plenty.
    bins = 40
    lo, hi = float(iso_scores.min()), float(iso_scores.max())
    if hi - lo == 0: hi = lo + 1e-6
    normal_hist, _ = np.histogram(iso_scores[final == 1], bins=bins, range=(lo, hi))
    anomaly_hist, _ = np.histogram(iso_scores[final == -1], bins=bins, range=(lo, hi))

    log(f"Detection done — {anomalies:,} anomalies in {total:,} flows.")

    return {
        "total": total, "normal": normal, "anomalies": anomalies,
        "pct": round(100 * anomalies / total, 2) if total else 0,
        "agreement_rate": agr_rate, "score_sep": sep,
        "rl_threshold_pct": rl_pct,
        "rl_reward": agent_preview.get("rl_reward", 0),
        "held_out": held_out,
        "score_hist": {"normal": normal_hist.tolist(), "anomaly": anomaly_hist.tolist()},
    }


def evaluate(progress_cb=None):
    def log(msg):
        if progress_cb: progress_cb(msg)
    results = pd.read_csv(RESULTS_PATH)
    if "true_label" not in results.columns:
        log("No ground-truth labels in this split — showing detection breakdown only.")
        return _charts_only(results, log)

    y_true = results["true_label"].values
    y_pred = (results["final_pred"] == -1).astype(int).values
    y_score = -results["anomaly_score"].values

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    try:
        auc = roc_auc_score(y_true, y_score)
        fpr_vals, tpr_vals, _ = roc_curve(y_true, y_score)
    except Exception:
        auc, fpr_vals, tpr_vals = 0, [0, 1], [0, 1]

    # Custom dark->accent colormap (instead of seaborn's default "Blues") so the matrix
    # matches the app palette and stays dark enough that white annotation text is always
    # legible — the default cmap put near-white cells under white text on low counts.
    from matplotlib.colors import LinearSegmentedColormap
    cm_cmap = LinearSegmentedColormap.from_list("netguard_cm", ["#1B2127", ACC])
    fig1, ax1 = plt.subplots(figsize=(4.5, 3.5)); _style(fig1, ax1)
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt="d", cmap=cm_cmap,
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"],
                ax=ax1, linewidths=0.5, linecolor="#232B30", annot_kws={"color": "white", "size": 13},
                cbar=False)
    ax1.set_xlabel("Predicted", color=MUTED, fontsize=9); ax1.set_ylabel("Actual", color=MUTED, fontsize=9)
    ax1.set_title("Confusion Matrix — held-out test", color=ACC, fontsize=11, pad=8)
    plt.tight_layout(); cm_chart = _fig_to_b64(fig1); plt.close()

    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5)); _style(fig2, ax2)
    ax2.plot(fpr_vals, tpr_vals, color=ACC, lw=2, label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], color="#444", lw=1, linestyle="--")
    ax2.fill_between(fpr_vals, tpr_vals, alpha=0.1, color=ACC)
    ax2.set_xlabel("False Positive Rate", color=MUTED, fontsize=9); ax2.set_ylabel("True Positive Rate", color=MUTED, fontsize=9)
    ax2.set_title("ROC Curve — held-out test", color=ACC, fontsize=11, pad=8)
    ax2.legend(facecolor="#" + PANEL, edgecolor="#232B30", labelcolor=MUTED, fontsize=9)
    plt.tight_layout(); roc_chart = _fig_to_b64(fig2); plt.close()

    fig3, ax3 = plt.subplots(figsize=(4.5, 3.0)); _style(fig3, ax3)
    names = ["Precision", "Recall", "F1-Score", "1-FPR"]; values = [prec, rec, f1, 1 - fpr_val]
    colors = [ACC, GRN, AMBER, MUTED]
    bars = ax3.bar(names, values, color=colors, width=0.55, edgecolor="#" + DARK)
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.2f}", ha="center", color="white", fontsize=10)
    ax3.set_ylim(0, 1.15); ax3.set_title("Performance — held-out test", color=ACC, fontsize=11, pad=8)
    plt.tight_layout(); met_chart = _fig_to_b64(fig3); plt.close()

    log(f"Evaluation done — F1={f1:.3f} AUC={auc:.3f} Recall={rec:.3f} (held-out test set)")

    agr = round(results["agreement"].isin(["High Confidence Anomaly", "High Confidence Normal"]).sum() / len(results) * 100, 2) if "agreement" in results.columns else 80.0
    fp_arr = results["final_pred"].values
    iso_s = results["anomaly_score"].values
    s_min, s_max = iso_s.min(), iso_s.max()
    iso_norm = np.zeros_like(iso_s) if (s_max - s_min) == 0 else (iso_s - s_min) / (s_max - s_min)
    n_, a_ = iso_norm[fp_arr == 1], iso_norm[fp_arr == -1]
    sep_norm = min(1.0, abs(n_.mean() - a_.mean())) if len(n_) > 0 and len(a_) > 0 else 0.5

    tuning_score = round((rec * 0.45) + ((1 - fpr_val) * 0.30) + (agr / 100 * 0.15) + (sep_norm * 0.10), 4)

    return {
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "fpr": round(fpr_val, 4), "auc": round(auc, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        # Renamed from the old "weighted_acc" — this is the same heuristic the RL
        # threshold search optimizes, not a claim about model accuracy. It's shown
        # in the UI as a small technical caption, not a competing headline number.
        "tuning_score": tuning_score,
        "wacc_recall": round(rec, 4), "wacc_fpr_inv": round(1 - fpr_val, 4),
        "wacc_agreement": round(agr / 100, 4), "wacc_sep": round(sep_norm, 4),
        "cm_chart": cm_chart, "roc_chart": roc_chart, "met_chart": met_chart,
    }


def _charts_only(results, log):
    total = len(results); anomalies = int((results["final_pred"] == -1).sum())
    fig, ax = plt.subplots(figsize=(4, 4)); _style(fig, ax)
    ax.pie([total - anomalies, anomalies], labels=["Normal", "Anomaly"], colors=[GRN, WARN],
           startangle=90, wedgeprops={"edgecolor": "#" + DARK, "linewidth": 2})
    ax.set_title("Detection Split", color=ACC, fontsize=12)
    chart = _fig_to_b64(fig); plt.close()
    log("Charts generated.")
    return {"pie_chart": chart, "total": total, "anomalies": anomalies}
