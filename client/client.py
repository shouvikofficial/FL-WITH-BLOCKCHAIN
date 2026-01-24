import numpy as np
from sklearn.linear_model import LogisticRegression
from security.attack import poison_weights

# ======================================================
# LOCAL CLIENT TRAINING
# ======================================================
def train_local_model(client_id, client_data, global_weights=None):
    """
    Train local model on client data.
    Supports:
    - Federated initialization
    - Attack injection
    """

    # -------------------------------
    # Prepare local dataset
    # -------------------------------
    X = np.array([x for x, y in client_data], dtype=float)
    y = np.array([y for x, y in client_data], dtype=int)

    # -------------------------------
    # Initialize model
    # -------------------------------
    model = LogisticRegression(
        max_iter=300,
        solver="lbfgs"
    )

    # -------------------------------
    # IMPORTANT: Initialize model correctly
    # -------------------------------
    # We must fit once to initialize internal structures
    model.fit(X, y)

    # -------------------------------
    # Apply global weights (AFTER fit)
    # -------------------------------
    if global_weights is not None:
        model.coef_ = global_weights[0].copy()
        model.intercept_ = global_weights[1].copy()

    # -------------------------------
    # Re-train locally (FL step)
    # -------------------------------
    model.fit(X, y)

    # -------------------------------
    # Extract weights
    # -------------------------------
    weights = [
        model.coef_.copy(),
        model.intercept_.copy()
    ]

    # -------------------------------
    # 🔥 ATTACK INJECTION HOOK
    # -------------------------------
    weights = poison_weights(client_id, weights)


    return weights
