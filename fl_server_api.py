"""
fl_server_api.py
================
Real distributed FL server endpoints.
Clients on other PCs POST their weight updates here.
Server waits for all clients, then aggregates + logs to blockchain.
"""

import numpy as np
import json
import os
import threading
import time
from flask import Blueprint, request, jsonify
from collections import defaultdict

# ================================================================
# IMPORTS FROM EXISTING PROJECT
# ================================================================
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from security.defense import aggregate_weights, DEFENSE_METHOD
from server.server import detect_attack, evaluate_global_model
from blockchain.blockchain import log_update

# ================================================================
# CONFIG
# ================================================================
EXPECTED_CLIENTS = int(os.getenv("FL_NUM_CLIENTS", 3))
TOTAL_ROUNDS     = int(os.getenv("FL_ROUNDS", 25))
METRICS_FILE     = "static/metrics.json"
LOG_FILE         = "static/training_log.txt"
ROUND_TIMEOUT    = int(os.getenv("FL_ROUND_TIMEOUT", 300))   # 5 min per round

fl_api = Blueprint("fl_api", __name__)

# ================================================================
# SHARED STATE (thread-safe with locks)
# ================================================================
_lock = threading.Lock()

state = {
    "round":           0,
    "global_weights":  None,
    "global_scaler":   None,
    "updates":         {},          # {client_id: update_dict}
    "round_complete":  False,
    "all_done":        False,
    "malicious":       set(),
    "suspicion":       defaultdict(int),
    "client_prev_hash":{},          # Track previous transaction hashes per client
    "metrics": {
        "accuracy": [], "f1": [], "precision": [],
        "recall": [],   "roc": [], "mcc": []
    }
}


def _log(msg):
    print(msg, flush=True)
    os.makedirs("static", exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _write_metrics(status, completed=False):
    """Dump current metrics to static/metrics.json for the dashboard."""
    os.makedirs("static", exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump({
            "status": status,
            "completed": completed,
            "round": state["round"],
            "metrics": {k: state["metrics"][k] for k in state["metrics"]},
            "malicious_attackers": list(state["malicious"])
        }, f)


# ================================================================
# ROUND AGGREGATION  (runs in background thread)
# ================================================================
def _aggregate_round():
    """Called once all clients have submitted for the current round."""
    round_num = state["round"]
    updates   = list(state["updates"].values())

    _log(f"\n🌍 Aggregating Round {round_num}/{TOTAL_ROUNDS}")

    # Detect attacks
    clean_weights, sample_counts = [], []
    malicious_this_round = set()

    for upd in updates:
        cid     = upd["client_id"]
        weights = upd["weights"]

        if detect_attack(weights, state["global_weights"]):
            state["suspicion"][cid] += 1
        else:
            state["suspicion"][cid] = 0

        if state["suspicion"][cid] >= 2:
            _log(f"  ⚠️ CONFIRMED MALICIOUS: {cid}")
            state["malicious"].add(cid)
            malicious_this_round.add(cid)

        # Determine status
        if cid in malicious_this_round or cid in state["malicious"]:
            status_str = "Malicious"
        elif state["suspicion"][cid] > 0:
            status_str = "Flagged/Suspicious"
        else:
            status_str = "Clean"

        # Blockchain log
        _log(f"  🔗 Logging {cid} to Blockchain...")
        try:
            log_update(
                client_id=cid, 
                weights=weights, 
                round_id=round_num
            )
        except Exception as e:
            _log(f"  ❌ Blockchain error: {e}")

        if cid not in state["malicious"]:
            clean_weights.append(weights)
            sample_counts.append(upd["num_samples"])

    # Fail-open
    if not clean_weights:
        _log("  ⚠️ All clients flagged — using all weights this round.")
        clean_weights   = [u["weights"]     for u in updates]
        sample_counts   = [u["num_samples"] for u in updates]

    # Aggregate
    state["global_weights"] = aggregate_weights(clean_weights, sample_counts)

    # Evaluate
    try:
        from data.data_loader import load_and_split_data
        client_data = load_and_split_data("data/dataset.xlsx", "target", EXPECTED_CLIENTS)
        acc, f1, prec, rec, roc, mcc = evaluate_global_model(state["global_weights"], client_data)
        state["metrics"]["accuracy"].append(acc)
        state["metrics"]["f1"].append(f1)
        state["metrics"]["precision"].append(prec)
        state["metrics"]["recall"].append(rec)
        state["metrics"]["roc"].append(roc)
        state["metrics"]["mcc"].append(mcc)
        _log(f"  📊 Round {round_num} → Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    except Exception as e:
        _log(f"  ❌ Evaluation error: {e}")

    # Clear for next round
    state["updates"] = {}

    if round_num >= TOTAL_ROUNDS:
        state["all_done"] = True
        _write_metrics("Training Completed Successfully ✅", completed=True)
        _log("\n✅ All rounds complete!")
    else:
        state["round"] += 1
        state["round_complete"] = True
        _write_metrics(f"Round {round_num} complete. Waiting for clients on round {state['round']}...")
        _log(f"  ✅ Round {round_num} done. Next: Round {state['round']}")


# ================================================================
# ENDPOINTS
# ================================================================

@fl_api.route("/fl/status", methods=["GET"])
def fl_status():
    """Dashboard/client polling: current round state."""
    with _lock:
        return jsonify({
            "round":           state["round"],
            "total_rounds":    TOTAL_ROUNDS,
            "clients_submitted": list(state["updates"].keys()),
            "expected_clients":  EXPECTED_CLIENTS,
            "all_done":          state["all_done"],
            "malicious":         list(state["malicious"])
        })


@fl_api.route("/fl/get_global_model", methods=["GET"])
def get_global_model():
    """Client downloads the current global weights."""
    with _lock:
        if state["global_weights"] is None:
            return jsonify({"weights": None, "round": state["round"]})
        serialized = [w.tolist() for w in state["global_weights"]]
        return jsonify({"weights": serialized, "round": state["round"]})


@fl_api.route("/fl/submit_update", methods=["POST"])
def submit_update():
    """
    Client submits its local weight update.
    Triggers aggregation when all clients have submitted.
    """
    data      = request.get_json(force=True)
    client_id = data.get("client_id")
    round_id  = data.get("round_id")
    weights   = [np.array(w) for w in data.get("weights", [])]
    num_samples = int(data.get("num_samples", 0))

    if not client_id or not weights:
        return jsonify({"error": "Missing client_id or weights"}), 400

    with _lock:
        if state["all_done"]:
            return jsonify({"status": "done", "message": "Training already complete."})

        if round_id != state["round"]:
            return jsonify({
                "status": "wrong_round",
                "expected": state["round"],
                "got": round_id
            }), 400

        if client_id in state["updates"]:
            return jsonify({"status": "already_submitted"})

        state["updates"][client_id] = {
            "client_id":  client_id,
            "weights":    weights,
            "num_samples": num_samples
        }
        submitted = len(state["updates"])
        _log(f"  📥 Received update from {client_id} ({submitted}/{EXPECTED_CLIENTS})")
        _write_metrics(f"Round {state['round']}: {submitted}/{EXPECTED_CLIENTS} clients submitted...")

        should_aggregate = (submitted >= EXPECTED_CLIENTS)

    if should_aggregate:
        # Run aggregation in background so we don't block this request
        t = threading.Thread(target=_aggregate_round, daemon=True)
        t.start()

    return jsonify({
        "status": "ok",
        "submitted": submitted,
        "expected": EXPECTED_CLIENTS
    })


def reset_fl_session():
    """Reset the FL session state and metrics."""
    with _lock:
        state["round"]          = 1
        state["global_weights"] = None
        state["global_scaler"]  = None
        state["updates"]        = {}
        state["round_complete"] = False
        state["all_done"]       = False
        state["malicious"]      = set()
        state["suspicion"]      = defaultdict(int)
        state["client_prev_hash"]= {}
        state["metrics"]        = {"accuracy": [], "f1": [], "precision": [],
                                   "recall": [], "roc": [], "mcc": []}

    os.makedirs("static", exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("🚀 Real Distributed FL Session Started\n")

    _write_metrics(f"Waiting for {EXPECTED_CLIENTS} clients on Round 1...")

@fl_api.route("/fl/start", methods=["POST"])
def fl_start():
    """Reset and start a new FL session (called from dashboard)."""
    reset_fl_session()
    return jsonify({"status": "ok", "message": f"FL session started. Waiting for {EXPECTED_CLIENTS} clients."})
