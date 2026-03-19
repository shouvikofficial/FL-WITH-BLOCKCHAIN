import numpy as np

# ======================================================
# ATTACK CONFIGURATION
# ======================================================
ATTACK_ENABLED = True          # 🔁 Toggle attack ON/OFF
MALICIOUS_CLIENT_ID = None     # ❌ No fixed malicious client
ATTACK_STRENGTH = 8.0          # Noise magnitude

# ======================================================
# MODEL POISONING ATTACK
# ======================================================
def poison_weights(client_id, weights):
    """
    Simulate a malicious client by poisoning model weights.
    This mimics a real-world model update manipulation attack.
    NOTE:
    - No client is hard-coded as malicious
    - Poisoning occurs only if attack is explicitly enabled elsewhere
    """

    if not ATTACK_ENABLED:
        return weights

    # ❌ Identity-based poisoning DISABLED
    # Malicious behavior should come from poisoned DATA, not client ID
    return weights
