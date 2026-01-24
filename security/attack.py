import numpy as np

# ======================================================
# ATTACK CONFIGURATION
# ======================================================
ATTACK_ENABLED = True          # 🔁 Toggle attack ON/OFF
MALICIOUS_CLIENT_ID = 3        # Client acting as attacker
ATTACK_STRENGTH = 8.0          # Noise magnitude

# ======================================================
# MODEL POISONING ATTACK
# ======================================================
def poison_weights(client_id, weights):
    """
    Simulate a malicious client by poisoning model weights.
    This mimics a real-world model update manipulation attack.
    """

    if not ATTACK_ENABLED:
        return weights

    if client_id != MALICIOUS_CLIENT_ID:
        return weights

    print("🚨 ATTACK DETECTED: Model poisoning by Client", client_id)

    poisoned_weights = []
    for w in weights:
        noise = ATTACK_STRENGTH * np.random.randn(*w.shape)
        poisoned_weights.append(w + noise)

    return poisoned_weights
