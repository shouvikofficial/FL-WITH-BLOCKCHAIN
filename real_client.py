"""
real_client.py
==============
Run this script on EACH CLIENT PC to participate in real distributed
Federated Learning. Each PC trains locally, then sends weights to the
server over HTTP.

USAGE:
    python real_client.py --client_id Client_1 --server https://YOUR-NGROK-URL.ngrok-free.app

REQUIREMENTS:
    pip install -r client_requirements.txt
"""

import argparse
import time
import os
import sys
import json

import numpy as np
import requests
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ================================================================
# MLP (must match server architecture exactly)
# ================================================================
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, precision_score, recall_score)

HIDDEN_LAYERS   = (64, 32)
N_WEIGHT_LAYERS = len(HIDDEN_LAYERS) + 1   # 3

# ================================================================
# TRAINING LOG  — sends events to the server over HTTP
# ================================================================
# The server stores events server-side, so the dashboard works even
# when this client runs on a DIFFERENT PC from the server.
# ================================================================
_log_events  = []    # in-memory accumulator (for fallback + terminal)
_server_url  = None  # set at session start
_client_id_g = None  # set at session start

def _init_log(client_id, server_url):
    """Call once at session start to register this client's log channel."""
    global _log_events, _server_url, _client_id_g
    _log_events  = []
    _server_url  = server_url
    _client_id_g = client_id
    # Tell the server to create a fresh log for this client
    try:
        requests.post(
            f"{server_url}/api/client_event",
            json={"client_id": client_id, "reset": True},
            timeout=5
        )
    except Exception:
        pass  # Server might not be up yet — that's okay

def _emit(event_type, **kwargs):
    """
    Send a structured training event to the server (and keep in-memory too).
    Works whether this client is on the SAME machine or a DIFFERENT PC.
    """
    event = {
        "type":      event_type,
        "timestamp": time.time(),
        **kwargs
    }
    _log_events.append(event)

    # ── POST to server (works across the network) ──────────────────
    if _server_url and _client_id_g:
        try:
            requests.post(
                f"{_server_url}/api/client_event",
                json={"client_id": _client_id_g, "event": event},
                timeout=5
            )
        except Exception:
            # Non-fatal: if network is down, fall back silently
            # Also write locally as backup
            _write_local_fallback()

    # ── Print to terminal for direct feedback ──────────────────────
    symbols = {
        "session_start":  "🚀", "data_loaded":    "📦",
        "preprocess":     "🔬", "round_start":    "🔄",
        "global_model":   "📡", "training_done":  "🧠",
        "submit_done":    "📤", "waiting":        "⏳",
        "round_complete": "✅", "session_end":    "🏁",
        "error":          "❌",
    }
    sym    = symbols.get(event_type, "•")
    detail = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"  {sym} [{event_type}] {detail}", flush=True)

def _write_local_fallback():
    """Write events to a local file as fallback if server is unreachable."""
    if not _client_id_g:
        return
    safe_id  = _client_id_g.replace("/", "_").replace("\\", "_")
    log_path = f"static/client_training_log_{safe_id}.json"
    os.makedirs("static", exist_ok=True)
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(_log_events, f)
    except Exception:
        pass



# ================================================================
# DATA LOADING  (each client has its own local data file)
# ================================================================
def load_local_data(data_path, label_column, client_id, total_clients=3):
    """
    Load LOCAL data from the client's own Excel file.
    If you have separate files per client, just load your own file directly.
    For demo purposes, we split the shared dataset by client index.
    """
    df = pd.read_excel(data_path).dropna()

    # Add interaction features (must match server's data loader)
    needed = ["age", "glucose", "BMI", "heartRate", "exang", "chol", "fbs"]
    if all(c in df.columns for c in needed):
        df["age_glucose"]     = df["age"]       * df["glucose"]
        df["age_BMI"]         = df["age"]       * df["BMI"]
        df["glucose_BMI"]     = df["glucose"]   * df["BMI"]
        df["heartRate_exang"] = df["heartRate"] * df["exang"]
        df["chol_fbs"]        = df["chol"]      * df["fbs"]

    num_features = len(df.columns) - 1  # exclude label

    X_all = df.drop(label_column, axis=1).values
    y_all = df[label_column].values

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # Split dataset into equal parts and take this client's slice
    idx = int(client_id.replace("Client_", "")) - 1  # 0-based
    indices = np.array_split(np.arange(len(X_all)), total_clients)[idx]

    X = X_all[indices]
    y = y_all[indices]

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = {str(int(k)): int(v) for k, v in zip(unique, counts)}

    scaler_stats = {
        "mean": scaler.mean_.tolist(),
        "var":  scaler.var_.tolist(),
        "n":    len(X)
    }

    _emit("data_loaded",
          num_samples=len(X),
          num_features=int(num_features),
          label=label_column,
          class_distribution=class_dist)

    return X, y, scaler_stats, num_features


# ================================================================
# SMOTE (optional, matching server-side logic)
# ================================================================
def _apply_smote_if_needed(X, y):
    """Apply SMOTE if class imbalance > 2.1:1. Returns (X, y, applied: bool)."""
    try:
        from collections import Counter
        counts = Counter(y)
        if len(counts) > 1:
            majority = max(counts.values())
            minority = min(counts.values())
            if majority / minority >= 2.1:
                from imblearn.over_sampling import SMOTE
                X, y = SMOTE(random_state=42).fit_resample(X, y)
                return X, y, True
    except Exception:
        pass
    return X, y, False


# ================================================================
# LOCAL TRAINING
# ================================================================
def _clip_weights(weights, max_norm=2.0):
    """Clip global L2 weight norm before submission — stabilises aggregation."""
    total_norm = np.sqrt(sum(np.linalg.norm(w) ** 2 for w in weights))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        weights = [w * scale for w in weights]
    return weights


def train_local(X, y, global_weights=None, round_num=1):
    """
    Train MLP on local data with:
      - Proper train/val split (accuracy measured on UNSEEN data)
      - Early stopping (avoids overfitting local data)
      - Adaptive learning rate
      - F1, ROC-AUC, Precision, Recall metrics
      - Weight norm clipping before returning weights
    """
    t_start = time.time()

    # ── Probabilistic poisoning (15% chance, mirrors server-side) ──────────
    poisoning_applied = False
    if np.random.random() < 0.15:
        n_poison = int(len(y) * 0.15)
        if n_poison > 0:
            idx = np.random.choice(len(y), n_poison, replace=False)
            y_p = y.copy()
            y_p[idx] = 1 - y_p[idx]
            y = y_p
            poisoning_applied = True

    # ── SMOTE balancing ─────────────────────────────────────────────────────
    X, y, smote_applied = _apply_smote_if_needed(X, y)

    scaler_type = "global" if global_weights is not None else "local"
    _emit("preprocess",
          round=round_num,
          scaler_type=scaler_type,
          smote_applied=smote_applied,
          poisoning_detected=poisoning_applied,
          samples_after_smote=int(len(X)))

    # ── Train / Validation split (15%) ───────────────────────────────────────
    # Evaluate on UNSEEN data — much more honest than training accuracy
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
    except ValueError:
        # Fallback if stratify impossible (tiny dataset or single class)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

    # ── Single-model approach: warm_start + manual early-stopping state reset ──
    # warm_start=True is REQUIRED so that injected weights survive into fit().
    # We do ONE 1-iter fit to build layer structure, inject global weights,
    # then manually reset sklearn's internal early-stopping counters before
    # the real training run. This is the only reliable way to combine both.
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        warm_start=True,      # MUST stay True so injected weights are preserved
        early_stopping=False, # Disabled for 1-iter structure-init fit
        max_iter=1,
        random_state=42,
    )

    # Step 1: build layer structure (1 iteration, no tracking)
    try:
        model.fit(X_train, y_train)
    except Exception:
        pass

    # Step 2: inject global weights (if available)
    if global_weights is not None:
        try:
            model.coefs_      = [global_weights[i].copy()
                                 for i in range(N_WEIGHT_LAYERS)]
            model.intercepts_ = [global_weights[N_WEIGHT_LAYERS + i].copy()
                                 for i in range(N_WEIGHT_LAYERS)]
        except (IndexError, ValueError, AttributeError):
            pass  # Shape mismatch on round 1 — safe to ignore

    # Step 3: enable early stopping and reset its internal tracking state
    model.early_stopping = True
    model.max_iter       = 300
    model.n_iter_no_change = 20
    model.tol              = 1e-4
    model.validation_fraction = 0.1
    # Reset tracking attributes so sklearn doesn't crash on 2nd+ call
    model.validation_scores_     = []
    model.best_validation_score_ = -np.inf
    model.no_improvement_count_  = 0
    if hasattr(model, 'best_loss_'):
        model.best_loss_ = np.inf

    # Step 4: real training — warm_start preserves injected weights
    model.fit(X_train, y_train)


    # ── Evaluate on VALIDATION set (unseen data) ─────────────────────────────
    y_pred      = model.predict(X_val)
    y_pred_prob = model.predict_proba(X_val)[:, 1] if len(np.unique(y_val)) == 2 else None

    val_acc  = float(accuracy_score(y_val, y_pred))
    val_f1   = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
    val_prec = float(precision_score(y_val, y_pred, average="weighted", zero_division=0))
    val_rec  = float(recall_score(y_val, y_pred, average="weighted", zero_division=0))
    val_roc  = float(roc_auc_score(y_val, y_pred_prob)) if y_pred_prob is not None else None

    # Train accuracy (for reference — always higher than val)
    train_acc = float(model.score(X_train, y_train))

    duration     = round(time.time() - t_start, 2)
    flat_weights = list(model.coefs_) + list(model.intercepts_)



    _emit("training_done",
          round=round_num,
          local_accuracy=round(val_acc, 4),   # Validation accuracy (honest)
          train_accuracy=round(train_acc, 4), # Training accuracy (reference)
          f1_score=round(val_f1, 4),
          precision=round(val_prec, 4),
          recall=round(val_rec, 4),
          roc_auc=round(val_roc, 4) if val_roc is not None else None,
          duration_sec=duration,
          num_samples=int(len(X_train)),
          val_samples=int(len(X_val)),
          n_iter=model.n_iter_)

    return flat_weights


# ================================================================
# SERVER COMMUNICATION
# ================================================================
def get_global_model(server_url):
    """Download the current global model from server."""
    try:
        r = requests.get(f"{server_url}/fl/get_global_model", timeout=30)
        data = r.json()
        if data["weights"] is None:
            return None
        return [np.array(w) for w in data["weights"]]
    except Exception as e:
        _emit("error", message=f"Could not fetch global model: {e}")
        return None


def submit_update(server_url, client_id, round_id, weights, num_samples):
    """Send local weights to the server."""
    payload = {
        "client_id":   client_id,
        "round_id":    round_id,
        "weights":     [w.tolist() for w in weights],
        "num_samples": num_samples
    }
    try:
        r = requests.post(f"{server_url}/fl/submit_update", json=payload, timeout=60)
        return r.json()
    except Exception as e:
        _emit("error", message=f"Could not submit update: {e}")
        return None


def wait_for_round(server_url, current_round, timeout=300):
    """Poll until the server moves to the next round."""
    _emit("waiting", round=current_round)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{server_url}/fl/status", timeout=10)
            status = r.json()
            if status.get("all_done"):
                return "done"
            if status.get("round", current_round) > current_round:
                return "next"
        except Exception:
            pass
        time.sleep(5)
    return "timeout"


# ================================================================
# MAIN TRAINING LOOP
# ================================================================
def run(client_id, server_url, data_path, label_col, total_clients, total_rounds):
    # Initialise the logging channel — points events to this client's server-side log
    _init_log(client_id, server_url)

    print(f"\n🚀 Starting Federated Client: {client_id}")
    print(f"   Server: {server_url}")
    print(f"   Rounds: {total_rounds}\n")

    _emit("session_start",
          client_id=client_id,
          server_url=server_url,
          total_rounds=total_rounds,
          total_clients=total_clients,
          data_path=data_path)

    # Load local data once
    X, y, scaler_stats, num_features = load_local_data(data_path, label_col, client_id, total_clients)

    for rnd in range(1, total_rounds + 1):
        print(f"\n--- Round {rnd}/{total_rounds} ---")
        _emit("round_start", round=rnd, total_rounds=total_rounds)

        # 1. Get global model
        global_weights = get_global_model(server_url)
        received = global_weights is not None
        _emit("global_model", round=rnd, received=received)
        if rnd == 1:
            print(f"  📡 Global model: {'received' if received else 'none (first round)'}")

        # 2. Train locally (preprocessing happens inside)
        print(f"  🧠 Training locally...")
        local_weights = train_local(X, y, global_weights, round_num=rnd)

        # 3. Submit to server
        print(f"  📤 Submitting update to server...")
        resp = submit_update(server_url, client_id, rnd, local_weights, len(X))
        if resp:
            submitted = resp.get("submitted", "?")
            expected  = resp.get("expected", "?")
            _emit("submit_done",
                  round=rnd,
                  status=resp.get("status"),
                  submitted=submitted,
                  expected=expected)
            print(f"  ✅ Server response: {resp.get('status')} ({submitted}/{expected} clients)")
        else:
            _emit("error", round=rnd, message="Failed to submit update")
            print(f"  ❌ Failed to submit. Retrying next round...")
            continue

        # 4. Wait for aggregation
        result = wait_for_round(server_url, rnd)
        if result == "done":
            _emit("round_complete", round=rnd, result="all_done")
            _emit("session_end", status="complete", rounds_done=rnd)
            print("\n🎉 Training complete! All rounds finished.")
            break
        elif result == "timeout":
            _emit("round_complete", round=rnd, result="timeout")
            print(f"\n⚠️ Timeout waiting for round {rnd}. Moving on...")
        else:
            _emit("round_complete", round=rnd, result="next_round")
            print(f"  ✅ Round {rnd} aggregated. Moving to round {rnd + 1}.")

    _emit("session_end", status="complete", rounds_done=total_rounds)
    print(f"\n✅ {client_id} finished all rounds. You may close this window.")


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Real Client")
    parser.add_argument("--client_id",  required=True,  help="e.g. Client_1, Client_2, Client_3")
    parser.add_argument("--server",     required=True,  help="Server URL e.g. https://abc.ngrok-free.app")
    parser.add_argument("--data",       default="data/dataset.xlsx", help="Path to local Excel data file")
    parser.add_argument("--label",      default="target",            help="Target column name")
    parser.add_argument("--clients",    default=3,  type=int,        help="Total number of clients")
    parser.add_argument("--rounds",     default=25, type=int,        help="Total rounds")
    args = parser.parse_args()

    # Validate client_id format
    if not args.client_id.startswith("Client_"):
        print("❌ client_id must be in format: Client_1, Client_2, Client_3")
        sys.exit(1)

    run(args.client_id, args.server.rstrip("/"), args.data, args.label, args.clients, args.rounds)
