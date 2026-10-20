"""
predict_api.py: inference API + dashboard route for NetGuard IDS.
Register in app.py:
    from predict_api import bp as predict_bp
    app.register_blueprint(predict_bp)

GET  /dashboard   POST /predict   POST /predict_csv (file, first 300 rows)
POST /explain     GET /metrics    GET /health
Optional: set ANTHROPIC_API_KEY (and NETGUARD_LLM_MODEL) for LLM-phrased explanations.
"""
import os, json, urllib.request
import numpy as np, pandas as pd
from flask import Blueprint, request, jsonify, send_from_directory
from utils import load_obj, models_exist, ISO_PATH, RF_PATH, SCALER_PATH, RL_PATH, REPORTS

bp = Blueprint("predict", __name__)
BASE = os.path.dirname(os.path.abspath(__file__))
MAX_ROWS = 300
_M = {}

def _models():
    if not _M:
        _M.update(iso=load_obj(ISO_PATH), rf=load_obj(RF_PATH),
                  scaler=load_obj(SCALER_PATH), meta=load_obj(RL_PATH))
    return _M

def _score(df):
    m = _models(); sc = m["scaler"]; a = m["meta"]
    cols = list(sc.feature_names_in_)
    df = df.copy(); df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).reindex(columns=cols)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(pd.Series(sc.mean_, index=cols)).astype(float)   # missing -> typical value
    scaled = pd.DataFrame(sc.transform(df), columns=cols)
    X = scaled[list(m["rf"].feature_names_in_)]
    iso = m["iso"].decision_function(X)
    inorm = np.clip((iso - a["s_min"]) / (a["s_max"] - a["s_min"] + 1e-9), 0, 1)
    p_att = 1.0 - m["rf"].predict_proba(X)[:, a["normal_idx"]]
    fused = a["meta"].predict_proba(np.column_stack([inorm, p_att]))[:, 1]
    return df, X, iso, fused, m["rf"].predict(X)

def _verdict(f, t, s):
    att = f >= _models()["meta"].get("threshold", 0.5)
    return {"verdict": "ANOMALY" if att else "NORMAL",
            "attack_type": str(t) if att else "NORMAL",
            "confidence": round(float(f if att else 1 - f), 4),
            "anomaly_score": round(float(s), 4)}

def _signals(raw, x):
    """Top-3 features that deviate most from the NORMAL training profile (deterministic)."""
    m = _models(); a = m["meta"]; sc = m["scaler"]
    feats = a["features"]; mu = np.array(a["ref_mean"]); sd = np.array(a["ref_std"])
    z = (x[feats].values - mu) / sd
    cols = list(sc.feature_names_in_); out = []
    for i in np.argsort(-np.abs(z))[:3]:
        j = cols.index(feats[i])
        out.append({"feature": feats[i], "value": round(float(raw[feats[i]]), 4),
                    "normal_typical": round(float(mu[i] * sc.scale_[j] + sc.mean_[j]), 4),
                    "z": round(float(z[i]), 2)})
    return out

def _template(v, sig):
    s = "; ".join(f"{g['feature']} = {g['value']} (normal is about {g['normal_typical']}, "
                  f"{abs(g['z']):.1f} SD {'above' if g['z'] > 0 else 'below'})" for g in sig)
    return f"Model verdict {v['verdict']} ({v['confidence']:.0%}). Largest deviations from normal traffic: {s}."

def _llm(facts):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    body = json.dumps({
        "model": os.environ.get("NETGUARD_LLM_MODEL", "claude-haiku-5-5"),
        "max_tokens": 200,
        "system": ("You phrase an intrusion detector's output for an analyst. Use ONLY the facts given. "
                   "Max 2 sentences. Do not name attack techniques beyond the given attack_type, do not "
                   "invent causes, and do not claim certainty: this is a statistical signal, not proof."),
        "messages": [{"role": "user", "content": json.dumps(facts)}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.load(r)["content"][0]["text"].strip()
    except Exception:
        return None

def _need_models():
    return None if models_exist() else (jsonify({"error": "No trained models. Run train.py first."}), 400)

@bp.route("/dashboard")
def dashboard():
    return send_from_directory(os.path.join(BASE, "templates"), "dashboard.html")

@bp.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": models_exist(),
                    "llm_explanations": bool(os.environ.get("ANTHROPIC_API_KEY"))})

@bp.route("/metrics")
def metrics():
    rp = os.path.join(REPORTS, "metrics.json"); cp = os.path.join(REPORTS, "metrics_chrono.json")
    if not os.path.exists(rp):
        return jsonify({"error": "Run train.py first"}), 404
    r = json.load(open(rp))
    if os.path.exists(cp):          # prefer the stricter time-ordered result as the headline
        c = json.load(open(cp)); c["random_split"] = r["headline"]
        return jsonify(c)
    return jsonify(r)

@bp.route("/predict", methods=["POST"])
def predict():
    err = _need_models()
    if err: return err
    data = request.get_json(silent=True)
    if not data: return jsonify({"error": "Send a JSON list of flow objects"}), 400
    if isinstance(data, dict): data = [data]
    if len(data) > 5000: return jsonify({"error": "Max 5000 flows per request"}), 413
    _, _, iso, fused, types = _score(pd.DataFrame(data))
    return jsonify([_verdict(f, t, s) for f, t, s in zip(fused, types, iso)])

@bp.route("/predict_csv", methods=["POST"])
def predict_csv():
    err = _need_models()
    if err: return err
    if "file" not in request.files: return jsonify({"error": "No file"}), 400
    df = pd.read_csv(request.files["file"], nrows=MAX_ROWS)
    raw, _, iso, fused, types = _score(df)
    rows = [{"i": i, **_verdict(f, t, s), "flow": raw.iloc[i].to_dict()}
            for i, (f, t, s) in enumerate(zip(fused, types, iso))]
    return jsonify({"n": len(rows), "flagged": sum(r["verdict"] == "ANOMALY" for r in rows), "rows": rows})

@bp.route("/explain", methods=["POST"])
def explain():
    err = _need_models()
    if err: return err
    flow = (request.get_json(silent=True) or {}).get("flow")
    if not flow: return jsonify({"error": "Send {\"flow\": {...}}"}), 400
    raw, X, iso, fused, types = _score(pd.DataFrame([flow]))
    v = _verdict(fused[0], types[0], iso[0]); sig = _signals(raw.iloc[0], X.iloc[0])
    text = _llm({**v, "top_deviations": sig})
    return jsonify({**v, "signals": sig, "explanation": text or _template(v, sig),
                    "source": "llm" if text else "template"})
