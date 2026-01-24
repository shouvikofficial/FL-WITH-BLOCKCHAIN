import numpy as np
from data.data_loader import load_and_split_data
from client.client import train_local_model
from blockchain.blockchain import log_update
from sklearn.linear_model import LogisticRegression
from evaluation.metrics import evaluate_model

# ======================================================
# CONFIGURATION
# ======================================================
DATA_PATH = "data/dataset.xlsx"
LABEL_COLUMN = "target"

NUM_CLIENTS = 3
ROUNDS = 5

# 🔐 Malicious client registry (DEFENSE)
malicious_clients = set()

# ======================================================
# SIMPLE ATTACK DETECTION (Model Poisoning)
# ======================================================
def detect_attack(weights, threshold=10.0):
    """
    Detect poisoning using weight norm explosion.
    """
    norm = np.linalg.norm(weights[0])
    return norm > threshold


# ======================================================
# FEDERATED AVERAGING (FedAvg)
# ======================================================
def federated_average(weights_list):
    """
    Perform Federated Averaging on client model weights.
    """
    avg_weights = []
    for layer_weights in zip(*weights_list):
        avg_weights.append(np.mean(layer_weights, axis=0))
    return avg_weights


# ======================================================
# GLOBAL MODEL EVALUATION (ROBUST & FIXED)
# ======================================================
def evaluate_global_model(global_weights, client_data):
    """
    Evaluate the aggregated global model on all client samples.
    """

    X_all = []
    y_all = []

    for client in client_data:
        for sample in client:
            x, y = sample
            X_all.append(x)
            y_all.append(y)

    X_all = np.asarray(X_all, dtype=float)
    y_all = np.asarray(y_all, dtype=int)

    if global_weights is None:
        raise ValueError("Global weights are None during evaluation")

    model = LogisticRegression(max_iter=300)

    # REQUIRED when setting weights manually
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = X_all.shape[1]

    model.coef_ = global_weights[0]
    model.intercept_ = global_weights[1]

    return evaluate_model(model, X_all, y_all)


# ======================================================
# MAIN FEDERATED LEARNING LOOP (DEFENDED)
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

    for round_num in range(ROUNDS):
        print(f"\n🌍 Federated Round {round_num + 1}")
        local_weights = []

        for i in range(NUM_CLIENTS):
            client_id = f"Client_{i+1}"

            weights = train_local_model(
                client_id=i + 1,
                client_data=client_data[i],
                global_weights=global_weights
            )

            print(f"[{client_id}] Training completed")

            # 🔎 ATTACK DETECTION
            if detect_attack(weights):
                print(f"⚠️ ATTACK DETECTED: Model poisoning by {client_id}")
                malicious_clients.add(client_id)

            # 🔐 Blockchain logging (ALWAYS log)
            log_update(
                client_id=client_id,
                weights=weights,
                round_id=round_num + 1
            )

            # 🛡️ DEFENSE: exclude malicious updates
            if client_id not in malicious_clients:
                local_weights.append(weights)
            else:
                print(f"🚫 Ignoring weights from malicious {client_id}")

        # Safety: ensure at least one honest client
        if not local_weights:
            raise RuntimeError("All clients marked malicious. Stopping training.")

        # Aggregate updates
        global_weights = federated_average(local_weights)

        # Evaluate global model
        accuracy, f1 = evaluate_global_model(global_weights, client_data)
        print(
            f"📊 Round {round_num + 1} → "
            f"Accuracy={accuracy:.4f}, F1={f1:.4f}"
        )

    print("\n✅ Federated Learning Completed Successfully")


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    run_federated_learning()
