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

HIDDEN_LAYERS   = (64, 32)
N_WEIGHT_LAYERS = len(HIDDEN_LAYERS) + 1   # 3


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

    X_all = df.drop(label_column, axis=1).values
    y_all = df[label_column].values

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # Split dataset into equal parts and take this client's slice
    idx = int(client_id.replace("Client_", "")) - 1  # 0-based
    indices = np.array_split(np.arange(len(X_all)), total_clients)[idx]

    X = X_all[indices]
    y = y_all[indices]
    print(f"📦 Loaded {len(X)} local samples for {client_id}")

    scaler_stats = {
        "mean": scaler.mean_.tolist(),
        "var":  scaler.var_.tolist(),
        "n":    len(X)
    }
    return X, y, scaler_stats


# ================================================================
# LOCAL TRAINING
# ================================================================
def train_local(X, y, global_weights=None):
    """Train MLP on local data, warm-starting from global weights."""
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        max_iter=200,
        alpha=0.001,
        warm_start=True,
        random_state=42
    )
    model.fit(X, y)

    # Load global weights if provided (federated warm-start)
    if global_weights is not None:
        try:
            model.coefs_      = [global_weights[i].copy()                    for i in range(N_WEIGHT_LAYERS)]
            model.intercepts_ = [global_weights[N_WEIGHT_LAYERS + i].copy() for i in range(N_WEIGHT_LAYERS)]
        except (IndexError, ValueError):
            pass   # Shape mismatch on first round — safe to ignore

    model.fit(X, y)

    flat_weights = list(model.coefs_) + list(model.intercepts_)
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
        print(f"❌ Could not fetch global model: {e}")
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
        print(f"❌ Could not submit update: {e}")
        return None


def wait_for_round(server_url, current_round, timeout=300):
    """Poll until the server moves to the next round."""
    print(f"  ⏳ Waiting for other clients (round {current_round})...")
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
    print(f"\n🚀 Starting Federated Client: {client_id}")
    print(f"   Server: {server_url}")
    print(f"   Rounds: {total_rounds}\n")

    # Load local data once
    X, y, scaler_stats = load_local_data(data_path, label_col, client_id, total_clients)

    for rnd in range(1, total_rounds + 1):
        print(f"\n--- Round {rnd}/{total_rounds} ---")

        # 1. Get global model
        global_weights = get_global_model(server_url)
        if rnd == 1:
            print(f"  📡 Global model: {'received' if global_weights else 'none (first round)'}")

        # 2. Train locally
        print(f"  🧠 Training locally...")
        local_weights = train_local(X, y, global_weights)

        # 3. Submit to server
        print(f"  📤 Submitting update to server...")
        resp = submit_update(server_url, client_id, rnd, local_weights, len(X))
        if resp:
            print(f"  ✅ Server response: {resp.get('status')} ({resp.get('submitted')}/{resp.get('expected')} clients)")
        else:
            print(f"  ❌ Failed to submit. Retrying next round...")
            continue

        # 4. Wait for aggregation
        result = wait_for_round(server_url, rnd)
        if result == "done":
            print("\n🎉 Training complete! All rounds finished.")
            break
        elif result == "timeout":
            print(f"\n⚠️ Timeout waiting for round {rnd}. Moving on...")
        else:
            print(f"  ✅ Round {rnd} aggregated. Moving to round {rnd + 1}.")

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
