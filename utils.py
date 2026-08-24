"""
utils.py — shared paths, save/load helpers
"""
import os, joblib

BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, "dataset")
MODELS  = os.path.join(BASE, "models")
REPORTS = os.path.join(BASE, "reports")

# Full-dataset processed file (kept for reference, not used for train/eval directly)
PROCESSED       = os.path.join(DATA, "processed.csv")

# Split files — training never sees val/test, metrics never touch train.
TRAIN_PROCESSED = os.path.join(DATA, "train_processed.csv")
VAL_PROCESSED   = os.path.join(DATA, "val_processed.csv")
TEST_PROCESSED  = os.path.join(DATA, "test_processed.csv")

TRAIN_FEATURES  = os.path.join(DATA, "train_features.csv")
VAL_FEATURES    = os.path.join(DATA, "val_features.csv")
TEST_FEATURES   = os.path.join(DATA, "test_features.csv")

ISO_PATH     = os.path.join(MODELS,  "isolation_forest.pkl")
RF_PATH      = os.path.join(MODELS,  "random_forest.pkl")
SCALER_PATH  = os.path.join(MODELS,  "scaler.pkl")
RL_PATH      = os.path.join(MODELS,  "rl_fusion_agent.pkl")
RESULTS_PATH = os.path.join(REPORTS, "results.csv")

def ensure_dirs():
    for d in [DATA, MODELS, REPORTS]:
        os.makedirs(d, exist_ok=True)

def save_obj(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_obj(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    return joblib.load(path)

def models_exist():
    return os.path.exists(ISO_PATH) and os.path.exists(RF_PATH) and os.path.exists(RL_PATH)
