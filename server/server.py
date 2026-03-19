import numpy as np
import os
import json
import sys
from collections import defaultdict
from data.data_loader import load_and_split_data
from client.client import train_local_model
from blockchain.blockchain import log_update
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from evaluation.metrics import evaluate_model
from security.defense import aggregate_weights, DEFENSE_METHOD

LOG_FILE = "static/training_log.txt"

def log_print(msg):
    print(msg)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

# ======================================================
# CONFIGURATION
# ======================================================
DATA_PATH    = "data/dataset.xlsx"
LABEL_COLUMN = "target"

NUM_CLIENTS  = 3
ROUNDS       = 25   # 🔧 Increased from 16 → 25 for better convergence

# 🔐 Malicious client registry (DEFENSE)
malicious_clients   = set()
suspicion_counter   = defaultdict(int)

# 📊 Metrics history
accuracy_history  = []
f1_history        = []
precision_history = []
recall_history    = []
roc_auc_history   = []
mcc_history       = []


# ======================================================
# ATTACK DETECTION (tighter threshold: 5.0 → 3.0)
# ======================================================
def detect_attack(weights, global_weights, alpha=3.0):
    """
    Detect poisoning via relative weight deviation.
    Tighter alpha (3.0 vs old 5.0) catches subtler poisoning.
    """
    if global_weights is None:
        return False

    diff  = np.linalg.norm(weights[0] - global_weights[0])
    base  = np.linalg.norm(global_weights[0])
    score = diff / (base + 1e-8)

    return score > alpha

# NOTE: The federated_average function was removed.
# All aggregation logic (FedAvg, Median, Trimmed Mean) is now handled
# by security.defense.aggregate_weights.


# MLP architecture — must match client.py exactly
_HIDDEN_LAYERS    = (64, 32)
_N_WEIGHT_LAYERS  = len(_HIDDEN_LAYERS) + 1   # 3


# ======================================================
# GLOBAL MODEL EVALUATION (MLP)
# ======================================================
def evaluate_global_model(global_weights, client_data):
    """
    Reconstruct an MLP from the federated global weights and evaluate it.
    Uses a tiny dummy fit to initialize weight shapes, then overwrites.
    """
    X_all, y_all = [], []
    for client in client_data:
        for sample in client:
            x, y = sample
            X_all.append(x)
            y_all.append(y)

    X_all = np.asarray(X_all, dtype=float)
    y_all = np.asarray(y_all, dtype=int)

    if global_weights is None:
        raise ValueError("Global weights are None during evaluation")

    # Build MLP with same architecture as clients
    model = MLPClassifier(
        hidden_layer_sizes=_HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        max_iter=1,            # intentional: shape init only
        alpha=0.001,
        warm_start=True,
        random_state=42
    )

    # Quick dummy fit on tiny sample → initializes coefs_ / intercepts_ shapes
    # max_iter=1 is intentional — we just need shape init, not convergence
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_all[:20], y_all[:20])

    # Overwrite with global federated weights
    model.coefs_      = [global_weights[i].copy()                       for i in range(_N_WEIGHT_LAYERS)]
    model.intercepts_ = [global_weights[_N_WEIGHT_LAYERS + i].copy()   for i in range(_N_WEIGHT_LAYERS)]

    accuracy, f1, precision, recall, roc_auc, mcc = evaluate_model(model, X_all, y_all)
    return accuracy, f1, precision, recall, roc_auc, mcc


# ======================================================
# MAIN FEDERATED LEARNING LOOP
# ======================================================
def run_federated_learning():
    """
    Server-side orchestration of federated learning.
    """

    client_data = load_and_split_data(
        DATA_PATH,
        LABEL_COLUMN,
        NUM_CLIENTS
    )

    global_weights = None
    global_scaler  = None
    
    # 🧹 Clear old log
    os.makedirs("static", exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("System Initialized.\nFederation Starting...\n")

    for round_num in range(ROUNDS):
        log_print(f"\n🌍 Federated Round {round_num + 1}/{ROUNDS}")

        malicious_clients.clear()
        all_client_updates = [] # Store all updates for potential re-aggregation
        local_weights   = []    # Filtered weights for aggregation
        scaler_stats_list = []
        sample_counts   = []

        for i in range(NUM_CLIENTS):
            client_id = f"Client_{i+1}"

            client_update = train_local_model(
                client_id=client_id,    # 🔧 Pass the string (e.g., "Client_3") not an int
                client_df=client_data[i],
                global_weights=global_weights,
                global_scaler=global_scaler
            )
            all_client_updates.append(client_update)

            weights     = client_update["weights"]
            scaler_stats = client_update["scaler_stats"]
            num_samples  = client_update["num_samples"]

            log_print(f"  [{client_id}] samples={num_samples} | training done")

            # 🔎 TEMPORAL ATTACK DETECTION (alpha=3.0, 2 consecutive rounds)
            if detect_attack(weights, global_weights):
                suspicion_counter[client_id] += 1
            else:
                suspicion_counter[client_id] = 0

            if suspicion_counter[client_id] >= 2:
                log_print(f"  ⚠️ CONFIRMED MALICIOUS: {client_id}")
                malicious_clients.add(client_id)

            # 🔐 Blockchain logging
            log_print(f"  🔗 LOGGING to Blockchain...")
            log_update(
                client_id=client_id,
                weights=weights,
                round_id=round_num + 1
            )

            # 🛡️ Only aggregate clean clients
            if client_id not in malicious_clients:
                local_weights.append(weights)
                if scaler_stats is not None:
                    scaler_stats_list.append(scaler_stats)
                    sample_counts.append(num_samples)
            else:
                log_print(f"  🚫 Ignoring weights from malicious {client_id}")

        # 🔧 FAIL-OPEN: if all clients flagged, use all weights
        if not local_weights:
            log_print("  ⚠️ All clients flagged. Proceeding with all updates this round.")
            local_weights = [u["weights"] for u in all_client_updates]
            sample_counts = [u["num_samples"] for u in all_client_updates]
            scaler_stats_list = [u["scaler_stats"] for u in all_client_updates if u["scaler_stats"] is not None]


        # 2. Server Aggregation (Robust Defense)
        # ==========================
        # Check if an attack was detected in this round based on weight deviance
        # This check uses the original weights from all clients, not just the filtered ones.
        attack_detected = any(detect_attack(u["weights"], global_weights) for u in all_client_updates)
        if attack_detected:
            log_print(f"  ⚠️ Attack signature detected! Defending via {DEFENSE_METHOD.title()} Aggregation.")

        # Use robust aggregation from defense.py
        global_weights = aggregate_weights(local_weights, sample_counts=sample_counts)

        # Federated scaler (weighted mean/var)
        if scaler_stats_list:
            total_n = sum(s["n"] for s in scaler_stats_list)
            mean = sum(s["mean"] * s["n"] for s in scaler_stats_list) / total_n
            var  = sum(s["var"]  * s["n"] for s in scaler_stats_list) / total_n
            global_scaler = (mean, var)

        accuracy, f1, precision, recall, roc_auc, mcc = evaluate_global_model(
            global_weights, client_data
        )

        accuracy_history.append(accuracy)
        f1_history.append(f1)
        precision_history.append(precision)
        recall_history.append(recall)
        roc_auc_history.append(roc_auc)
        mcc_history.append(mcc)

        log_print(
            f"  📊 Round {round_num + 1} → "
            f"Acc={accuracy:.4f}  F1={f1:.4f}  "
            f"Prec={precision:.4f}  Rec={recall:.4f}"
        )

        # 🌐 EXPORT METRICS FOR WEB DASHBOARD
        os.makedirs("static", exist_ok=True)
        try:
            with open("static/metrics.json", "w") as f:
                json.dump({
                    "status": f"Training Round {round_num + 1}...",
                    "completed": False,
                    "round": round_num + 1,
                    "metrics": {
                        "accuracy": accuracy_history,
                        "f1": f1_history,
                        "precision": precision_history,
                        "recall": recall_history,
                        "roc": roc_auc_history,
                        "mcc": mcc_history
                    },
                    "malicious_attackers": list(malicious_clients)
                }, f)
        except Exception as e:
            print(f"Failed to write metrics.json: {e}")

    # 🔥 FINAL SUMMARY (last 5 rounds average)
    n = min(5, ROUNDS)
    log_print(f"\n📈 Final Performance (Average of Last {n} Rounds):")
    log_print(f"  Accuracy  : {np.mean(accuracy_history[-n:]):.4f}")
    log_print(f"  F1-score  : {np.mean(f1_history[-n:]):.4f}")
    log_print(f"  Precision : {np.mean(precision_history[-n:]):.4f}")
    log_print(f"  Recall    : {np.mean(recall_history[-n:]):.4f}")
    log_print(f"  ROC-AUC   : {np.mean(roc_auc_history[-n:]):.4f}")
    log_print(f"  MCC       : {np.mean(mcc_history[-n:]):.4f}")

    log_print("\n✅ Federated Learning Completed Successfully")
    
    # 🌐 FINAL BROADCAST FOR WEB DASHBOARD
    try:
        with open("static/metrics.json", "w") as f:
            json.dump({
                "status": "Training Completed Successfully ✅",
                "completed": True,
                "round": ROUNDS,
                "metrics": {
                    "accuracy": accuracy_history,
                    "f1": f1_history,
                    "precision": precision_history,
                    "recall": recall_history,
                    "roc": roc_auc_history,
                    "mcc": mcc_history
                },
                "malicious_attackers": list(malicious_clients)
            }, f)
    except Exception as e:
        pass


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    run_federated_learning()
