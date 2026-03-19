import numpy as np

# ======================================================
# ATTACK CONFIGURATION
# ======================================================
ATTACK_ENABLED      = False         # 🔁 Toggle attack ON/OFF (Turned OFF after testing)
MALICIOUS_CLIENT_ID = "Client_3"    # ❌ Hardcoded malicious client (for testing)
ATTACK_STRENGTH     = 8.0           # Noise magnitude multiplier

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

    if client_id == MALICIOUS_CLIENT_ID:
        print(f"\n  [😈 MALICIOUS] {client_id} is poisoning its weights!")
        poisoned_weights = []
        for layer in weights:
            # Inject random gaussian noise scaled by attack strength
            noise = np.random.normal(0, ATTACK_STRENGTH, layer.shape)
            poisoned_weights.append(layer + noise)
        return poisoned_weights
        
    return weights
