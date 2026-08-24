    
"""capture.py — NetGuard live packet sniffer
Requires: scapy, admin/root privileges"""
import numpy as np
import time
from collections import defaultdict
from utils import ISO_PATH, RF_PATH, SCALER_PATH, load_obj # Changed SVM to RF
FLOW_TIMEOUT = 5

def _key(pkt):
    try:
        from scapy.layers.inet import IP, TCP, UDP
        if IP in pkt:
            sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
            dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
            return (pkt[IP].src, pkt[IP].dst, sport, dport, pkt[IP].proto)
    except: pass
    return None

def _features(f):
    n = len(f["lengths"])
    lengths = f["lengths"]
    times   = f["times"]
    iats    = [times[i+1]-times[i] for i in range(len(times)-1)]
    duration = (times[-1] - times[0]) if len(times) > 1 else 1e-6
    return {
        "flow_duration":         duration,
        "total_fwd_packets":     n,
        "total_length_fwd_pkts": sum(lengths),
        "fwd_pkt_len_mean":      sum(lengths)/n if n else 0,
        "fwd_pkt_len_std":       float(np.std(lengths)) if n>1 else 0,
        "flow_iat_mean":         float(np.mean(iats)) if iats else 0,
        "flow_iat_std":          float(np.std(iats)) if len(iats)>1 else 0,
        "protocol":              f["proto"],
        "destination_port":      f["dport"],
        "fwd_pkt_len_max":       max(lengths) if lengths else 0,
        "fwd_pkt_len_min":       min(lengths) if lengths else 0,
        "packet_rate":           n / duration,
        "byte_rate":             sum(lengths) / duration,
    }

COLS = ["flow_duration","total_fwd_packets","total_length_fwd_pkts",
        "fwd_pkt_len_mean","fwd_pkt_len_std","flow_iat_mean","flow_iat_std",
        "protocol","destination_port","fwd_pkt_len_max","fwd_pkt_len_min","packet_rate","byte_rate"]
def _score(flow, key, iso, rf, scaler, socketio):
    if len(flow["lengths"]) < 2: return
    try:
        feat_dict = _features(flow)
        
        # 1. Feature Alignment (Using RF features)
        trained_features = rf.feature_names_in_ 
        X = np.array([[feat_dict.get(c, 0) for c in trained_features]])
        import warnings
        with warnings.catch_warnings():
         warnings.simplefilter("ignore")
         X_pad = np.zeros((1, scaler.n_features_in_))
         X_pad[0, :X.shape[1]] = X[0, :]
         X_scaled = scaler.transform(X_pad)
        # trim to RF's expected features
         Xs = X_scaled[:, :len(rf.feature_names_in_)]
        
        # 2. Prediction
        ip = int(iso.predict(Xs)[0])   # Isolation Forest
        label = rf.predict(Xs)[0]      # Random Forest (Attack Name)
        
        # 3. Decision Logic (YOUR BLOCK)
        if label == "NORMAL" and ip == 1:
            display_verdict = "NORMAL"
            color = "#06D6A0"
            agreement = "High Confidence Normal"
        elif label != "NORMAL" and ip == -1:
            display_verdict = label
            color = "#FF6B35"
            agreement = "High Confidence Anomaly"
        else:
            # One flags it, one doesn't
            display_verdict = label if label != "NORMAL" else "UNCERTAIN"
            color = "#FFB627"
            agreement = "Mixed Signals"

        # 4. Send to Dashboard
        socketio.emit("live_flow", {
            "src": key[0], 
            "dst": key[1],
            "sport": key[2], 
            "dport": key[3], 
            "proto": key[4],
            "verdict": display_verdict, # Sends "DOS", "SCAN", etc.
            "color": color,
            "agreement": agreement,    # Needed for your JS color logic
            "attack_type": label,
            "score": round(float(iso.decision_function(Xs)[0]), 4),
            "pkts": len(flow["lengths"]),
        })

    except Exception as e:
        print(f"Scoring Error: {e}")

def run_capture(socketio, is_running):
    try:
        from scapy.all import sniff, conf
        conf.verb = 0
    except ImportError:
        socketio.emit("capture_error", {"msg": "Scapy not installed. Run: pip install scapy"}); return
    try:
        # Load RF (Random Forest) instead of the old SVM
        iso = load_obj(ISO_PATH)
        rf = load_obj(RF_PATH) 
        scaler = load_obj(SCALER_PATH)
    except FileNotFoundError as e:
        socketio.emit("capture_error", {"msg": str(e)}); return

    flows = defaultdict(lambda: {"lengths":[],"times":[],"proto":0,"dport":0})
    last_seen = {}

    def process(pkt):
        k = _key(pkt)
        if not k: return
        now = time.time()
        f = flows[k]
        f["lengths"].append(len(pkt)); f["times"].append(now)
        f["proto"] = k[4]; f["dport"] = k[3]
        last_seen[k] = now
        for fk in [x for x,t in list(last_seen.items()) if now-t>FLOW_TIMEOUT and x!=k]:
            _score(flows[fk], fk, iso, rf, scaler, socketio)
            del flows[fk]; del last_seen[fk]

    socketio.emit("capture_started", {"msg": "Live capture running…"})
    sniff(prn=process, store=False, stop_filter=lambda _: not is_running())
    socketio.emit("capture_stopped", {"msg": "Capture stopped."})
