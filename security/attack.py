import numpy as np

# ======================================================
# ATTACK CONFIGURATION
# ======================================================
ATTACK_ENABLED      = True          # 🔁 Toggle attack ON/OFF
MALICIOUS_CLIENT_ID = "Client_3"    # ❌ Hardcoded malicious client (for testing)

# Attack types: "noise", "sign_flip", "free_rider", "label_flip"
ATTACK_TYPE         = "label_flip"  

ATTACK_STRENGTH     = 8.0           # Noise magnitude multiplier

# ======================================================
# DATA POISONING ATTACK (Label Flipping)
# ======================================================
def poison_data(client_id, X, y):
    """
    Simulate a data poisoning attack by flipping labels.
    If positive (1), changes to negative (0) and vice versa.
    """
    if not ATTACK_ENABLED or ATTACK_TYPE != "label_flip":
        return X, y

    if client_id == MALICIOUS_CLIENT_ID:
        print(f"\n  [😈 MALICIOUS] {client_id} is executing a LABEL-FLIP attack!")
        # Assuming binary classification (0 and 1)
        # y = 1 - y flips 1 to 0, and 0 to 1
        poisoned_y = 1 - y
        return X, poisoned_y

    return X, y

# ======================================================
# MODEL POISONING ATTACKS
# ======================================================
def poison_weights(client_id, weights, global_weights=None):
    """
    Simulate malicious weight manipulation attacks.
    """
    if not ATTACK_ENABLED or ATTACK_TYPE == "label_flip":
        # Label flip doesn't poison weights directly (it poisons training data)
        return weights

    if client_id == MALICIOUS_CLIENT_ID:
        if ATTACK_TYPE == "noise":
            print(f"\n  [😈 MALICIOUS] {client_id} is poisoning weights (NOISE)!")
            poisoned_weights = []
            for layer in weights:
                noise = np.random.normal(0, ATTACK_STRENGTH, layer.shape)
                poisoned_weights.append(layer + noise)
            return poisoned_weights

        elif ATTACK_TYPE == "sign_flip":
            print(f"\n  [😈 MALICIOUS] {client_id} is poisoning weights (SIGN-FLIP)!")
            # Multiply weights by a negative scalar (e.g., -2)
            return [layer * -2.0 for layer in weights]

        elif ATTACK_TYPE == "free_rider":
            print(f"\n  [😈 MALICIOUS] {client_id} is FREE-RIDING (returning global weights)!")
            if global_weights is not None:
                # Need flat structure similar to returning weights, but let's just 
                # add slight noise to global_weights to evade naive identical checks.
                free_rider_weights = []
                for layer in global_weights:
                    noise = np.random.normal(0, 0.001, layer.shape)
                    free_rider_weights.append(layer + noise)
                return free_rider_weights
            else:
                return weights  # fallback if no global weights

    return weights
