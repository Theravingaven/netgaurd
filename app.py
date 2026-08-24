"""
app.py — NetGuard IDS · Flask backend v3.0
"""
import os, threading, time
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from utils import ensure_dirs, models_exist, DATA, RESULTS_PATH
import pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.config["SECRET_KEY"] = "netguard-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

ensure_dirs()

_state = {
    "phase": "idle" if not models_exist() else "ready",
    "step": 0, "step_msg": "", "error": None, "result": None, "log": [],
}
_lock = threading.Lock()

def _set(**kw):
    with _lock:
        _state.update(kw)

def _log(msg):
    with _lock:
        _state["log"].append(msg)
        _state["step_msg"] = msg


def _run_pipeline(filepath, top_n=15):
    try:
        _set(phase="running", step=1, step_msg="Reading your file…", error=None, result=None, log=[])
        pre = pipeline.preprocess(filepath, progress_cb=_log)

        _set(step=2)
        feat = pipeline.select_features(top_n=top_n, progress_cb=_log)

        _set(step=3)
        pipeline.train_models(progress_cb=_log)

        _set(step=4)
        det = pipeline.detect(progress_cb=_log)

        _set(step=5)
        eval_r = pipeline.evaluate(progress_cb=_log)

        result = {
            # dataset story — what was ingested, what the model actually trained/tested on
            "total_connections": pre["rows"],
            "has_labels": pre["has_labels"],
            "train_rows": pre["train_rows"],
            "val_rows": pre["val_rows"],
            "test_rows": pre["test_rows"],
            "normal_count": pre["normal"],
            "attack_count": pre["attack"],

            # detection pass (on the held-out test split, when one exists)
            "scanned": det["total"],
            "flagged": det["anomalies"],
            "safe": det["normal"],
            "threat_pct": det["pct"],
            "held_out": det["held_out"],
            "score_hist": det["score_hist"],
            "rl_threshold_pct": det["rl_threshold_pct"],
            "rl_reward": det["rl_reward"],
            "agreement_rate": det["agreement_rate"],
            "score_sep": det["score_sep"],

            # feature importance
            "top_features": feat["features"][:8],
            "feat_chart": feat["chart"],
        }

        # evaluate() returns either full metrics (labeled + held-out) or a fallback pie chart
        result.update(eval_r)
        result["evaluated"] = "precision" in eval_r

        _set(phase="done", step=5, result=result)

    except Exception as e:
        import traceback
        _set(phase="error", error=str(e), step_msg=traceback.format_exc())


@app.route("/")
def index():
    return render_template("index.html", models_ready=models_exist())


@app.route("/scan", methods=["POST"])
def scan():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a .csv file"}), 400
    path = os.path.join(DATA, secure_filename(f.filename))
    ensure_dirs()
    f.save(path)
    top_n = int(request.form.get("top_n", 15))
    _set(phase="running", step=0, step_msg="Starting…", error=None, result=None, log=[])
    threading.Thread(target=_run_pipeline, args=(path, top_n), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/progress")
def progress():
    with _lock:
        return jsonify(dict(_state))


@app.route("/download")
def download():
    if not os.path.exists(RESULTS_PATH):
        return jsonify({"error": "No results yet"}), 404
    return send_file(RESULTS_PATH, as_attachment=True, download_name="netguard_results.csv")


# --- Connection ledger: paginated, filterable, searchable read of the last scan's results ---
_results_cache = {"mtime": None, "df": None}

def _load_results_df():
    if not os.path.exists(RESULTS_PATH):
        return None
    mtime = os.path.getmtime(RESULTS_PATH)
    if _results_cache["mtime"] != mtime:
        _results_cache["df"] = pd.read_csv(RESULTS_PATH)
        _results_cache["mtime"] = mtime
    return _results_cache["df"]


@app.route("/connections")
def connections():
    df = _load_results_df()
    if df is None:
        return jsonify({"rows": [], "total": 0, "page": 1, "page_size": 50, "has_labels": False})

    page = max(1, int(request.args.get("page", 1)))
    page_size = min(200, max(1, int(request.args.get("page_size", 50))))
    filt = request.args.get("filter", "all")
    search = request.args.get("search", "").strip().upper()

    view = df
    if filt == "normal":
        view = view[view["final_pred"] == 1]
    elif filt == "suspicious":
        view = view[view["final_pred"] == -1]
    elif filt == "disagreement":
        view = view[view["iso_pred"] != view["rf_pred"]]

    if search:
        view = view[view["attack_type"].astype(str).str.upper().str.contains(search)]

    total = len(view)
    start = (page - 1) * page_size
    page_df = view.iloc[start:start + page_size]

    has_labels = "true_label" in df.columns
    rows = []
    for i, r in page_df.iterrows():
        rows.append({
            "idx": int(i) + 1,
            "verdict": "Suspicious" if r["final_pred"] == -1 else "Normal",
            "attack_type": r["attack_type"],
            "row_accuracy_pct": float(r["row_accuracy_pct"]),
            "agreement": r["agreement"],
            "iso_pred": "Anomaly" if r["iso_pred"] == -1 else "Normal",
            "rf_pred": "Anomaly" if r["rf_pred"] == -1 else "Normal",
            "true_label": (("Attack" if r["true_label"] == 1 else "Normal") if has_labels else None),
        })

    return jsonify({"rows": rows, "total": total, "page": page, "page_size": page_size, "has_labels": has_labels})


_capture_running = False

@app.route("/start_capture", methods=["POST"])
def start_capture():
    global _capture_running
    if _capture_running:
        return jsonify({"status": "already running"})
    if not models_exist():
        return jsonify({"error": "No trained models found. Run a CSV scan first."}), 400
    _capture_running = True
    import capture as cap
    threading.Thread(target=cap.run_capture, args=(socketio, lambda: _capture_running), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/stop_capture", methods=["POST"])
def stop_capture():
    global _capture_running
    _capture_running = False
    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  NetGuard IDS v3.0 — Zero-Day Threat Scanner")
    print("  Open:  http://localhost:5000")
    print("=" * 50 + "\n")
    # allow_unsafe_werkzeug: this app is a local single-user dev tool run via `python app.py`
    # or run.bat, never a production deployment — newer flask-socketio blocks the dev server
    # by default unless this is set explicitly.
    socketio.run(app, debug=False, port=5000, allow_unsafe_werkzeug=True)
