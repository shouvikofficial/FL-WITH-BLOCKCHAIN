import numpy as np
import sys
import os
from sklearn.neural_network import MLPClassifier
from client.client_preprocessing import local_preprocess
from security.attack import poison_weights

# ======================================================
# MLP ARCHITECTURE (must match server.py exactly!)
# ======================================================
HIDDEN_LAYERS    = (64, 32)   # 🔧 Reverted to original highly-performant architecture
N_WEIGHT_LAYERS  = len(HIDDEN_LAYERS) + 1   # 3


# ======================================================
# LOCAL CLIENT TRAINING
# ======================================================
def train_local_model(client_id, client_df, global_weights=None, global_scaler=None):
    """
    Federated MLP client:
      1. Preprocess local data
      2. Fit MLP to initialize weight shapes
      3. Overwrite with global weights (warm-start)
      4. Continue training (warm_start=True keeps weights)
      5. Return all layer weights + scaler stats + sample count
    """

    # ------------------------------------------------
    # 1️⃣ LOCAL PREPROCESSING
    # ------------------------------------------------
    X, y, scaler_stats = local_preprocess(
        client_df,
        global_scaler=global_scaler
    )

    # ------------------------------------------------
    # 2️⃣ MLP INITIALIZATION
    # ------------------------------------------------
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        max_iter=200,           # 🔧 Reverted to 200
        alpha=0.001,
        warm_start=True,
        random_state=42,
        early_stopping=False
    )

    # ------------------------------------------------
    # 3️⃣ INITIAL FIT — initializes coefs_ / intercepts_ shapes
    # ------------------------------------------------
    model.fit(X, y)

    # ------------------------------------------------
    # 4️⃣ LOAD GLOBAL WEIGHTS (warm-start from server)
    # ------------------------------------------------
    if global_weights is not None:
        try:
            model.coefs_      = [global_weights[i].copy()                    for i in range(N_WEIGHT_LAYERS)]
            model.intercepts_ = [global_weights[N_WEIGHT_LAYERS + i].copy() for i in range(N_WEIGHT_LAYERS)]
        except (IndexError, ValueError):
            pass   # shape mismatch on first round — safe to ignore

    # ------------------------------------------------
    # 5️⃣ CONTINUE TRAINING (warm_start keeps loaded weights)
    # ------------------------------------------------
    model.fit(X, y)

    # ------------------------------------------------
    # 6️⃣ EXTRACT ALL LAYER WEIGHTS
    #    Layout: [coef_0, coef_1, coef_2, ic_0, ic_1, ic_2]
    # ------------------------------------------------
    flat_weights = list(model.coefs_) + list(model.intercepts_)
    
    # Apply poisoning if this client is the designated attacker
    flat_weights = poison_weights(client_id, flat_weights)

    return {
        "weights":      flat_weights,
        "scaler_stats": scaler_stats,
        "num_samples":  len(X)
    }
