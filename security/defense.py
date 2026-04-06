import numpy as np
import warnings
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# DEFENSE CONFIGURATION
# ======================================================
DEFENSE_ENABLED = True     # 🔁 Toggle defense ON/OFF
DEFENSE_METHOD = "trust_weighted"  # options: "mean", "median", "trimmed", "krum", "trust_weighted"

TRIM_RATIO = 0.2           # used only for trimmed mean (discards top/bottom 20%)

# ======================================================
# DIFFERENTIAL PRIVACY (DP)
# ======================================================
def apply_dp_to_weights(weights, clip_norm=1.0, noise_multiplier=0.01):
    """
    Applies Gaussian noise to weights for Differential Privacy.
    Limits the influence of any single client.
    """
    dp_weights = []
    for layer in weights:
        # 1. Clip
        l2_norm = np.linalg.norm(layer)
        clip_factor = min(1.0, clip_norm / (l2_norm + 1e-8))
        clipped_layer = layer * clip_factor
        
        # 2. Add calibration noise
        noise = np.random.normal(0, noise_multiplier * clip_norm, layer.shape)
        dp_weights.append(clipped_layer + noise)
    return dp_weights

# ======================================================
# AGGREGATION ALGORITHMS
# ======================================================
def simple_mean(weights_list, sample_counts=None):
    """
    Standard Federated Averaging (weighted by sample counts if provided).
    """
    if sample_counts is None or len(sample_counts) != len(weights_list):
        # Uniform mean fallback
        avg = []
        for layer_idx in range(len(weights_list[0])):
            layer_avg = np.mean([w[layer_idx] for w in weights_list], axis=0)
            avg.append(layer_avg)
        return avg

    # Weighted mean
    total_samples = sum(sample_counts)
    avg = []
    for layer_idx in range(len(weights_list[0])):
        layer_sum = np.zeros_like(weights_list[0][layer_idx], dtype=np.float64)
        for i, w in enumerate(weights_list):
            layer_sum += w[layer_idx] * sample_counts[i]
        avg.append(layer_sum / total_samples)
    return avg


def coordinate_median(weights_list):
    """
    Coordinate-wise Median.
    Takes the median value for every single parameter across all clients.
    Highly robust against arbitrary weight poisoning.
    """
    median_weights = []
    for layer_idx in range(len(weights_list[0])):
        # Stack all client updates for this layer into a single array
        layer_stack = np.stack([w[layer_idx] for w in weights_list], axis=0)
        # Calculate median along the client dimension (axis 0)
        layer_med = np.median(layer_stack, axis=0)
        median_weights.append(layer_med)
    return median_weights


def trimmed_mean(weights_list, trim_ratio=TRIM_RATIO):
    """
    Coordinate-wise Trimmed Mean.
    Discards the highest and lowest `trim_ratio` fraction of values
    for every single parameter, then averages the rest.
    """
    num_clients = len(weights_list)
    trim_count = int(num_clients * trim_ratio)
    
    if trim_count == 0 or num_clients <= 2 * trim_count:
        # Not enough clients to trim, fallback to median
        return coordinate_median(weights_list)

    trimmed_weights = []
    for layer_idx in range(len(weights_list[0])):
        layer_stack = np.stack([w[layer_idx] for w in weights_list], axis=0)
        # Sort along client dimension
        sorted_stack = np.sort(layer_stack, axis=0)
        # Trim top and bottom
        trimmed_stack = sorted_stack[trim_count : -trim_count]
        # Average the remaining
        layer_avg = np.mean(trimmed_stack, axis=0)
        trimmed_weights.append(layer_avg)
        
    return trimmed_weights


def krum_aggregation(weights_list, f=1):
    """
    Krum Aggregation Algorithm.
    Selects the single update that is closest to its (n - f - 2) neighbors.
    f: assumed number of malicious clients.
    """
    n = len(weights_list)
    if n <= 2:
        return simple_mean(weights_list)
    
    # Flatten weights for distance calculation
    flat_weights = [np.concatenate([w.flatten() for w in weights]) for weights in weights_list]
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(flat_weights[i] - flat_weights[j])
            distances[i, j] = dist
            distances[j, i] = dist
            
    scores = []
    # k neighbors to sum distances over (n - f - 2)
    k = max(1, n - f - 2)
    
    for i in range(n):
        # Sort distances for client i, skip distance to itself (index 0 which is 0.0)
        sorted_dists = np.sort(distances[i])[1:k+1]
        scores.append(np.sum(sorted_dists))
        
    best_client_idx = np.argmin(scores)
    print(f"  [🛡️ KRUM] Selected client {best_client_idx+1} as the most reliable update.")
    return weights_list[best_client_idx]


def multi_krum_aggregation(weights_list, f=1):
    """
    Multi-Krum Aggregation Algorithm.
    Selects the top (n - f) mostly reliable updates and averages them.
    Faster training than standard Krum while retaining security.
    """
    n = len(weights_list)
    if n <= 2:
        return simple_mean(weights_list)
    
    flat_weights = [np.concatenate([w.flatten() for w in weights]) for weights in weights_list]
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(flat_weights[i] - flat_weights[j])
            distances[i, j] = dist
            distances[j, i] = dist
            
    scores = []
    k = max(1, n - f - 2)
    
    for i in range(n):
        sorted_dists = np.sort(distances[i])[1:k+1]
        scores.append(np.sum(sorted_dists))
        
    # Multi-Krum: Select top 'm' clients with lowest scores (m = n - f)
    m = max(1, n - f)
    best_client_indices = np.argsort(scores)[:m]
    
    print(f"  [🛡️ MULTI-KRUM] Selected clients {[i+1 for i in best_client_indices]} out of {n} as reliable.")
    
    selected_weights = [weights_list[i] for i in best_client_indices]
    return simple_mean(selected_weights)


def foolsgold_scores(weights_list):
    """
    FoolsGold Collusion Detection.
    Computes cosine similarity between clients. 
    If clients send suspiciously identical updates (collusion), they get a score close to 0.
    Returns: list of scores [0.0 to 1.0] representing integrity.
    """
    n = len(weights_list)
    if n <= 1:
        return [1.0] * n
        
    # Flatten weights
    flat_weights = np.array([np.concatenate([w.flatten() for w in weights]) for weights in weights_list])
    
    # Calculate pairwise cosine similarity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = cosine_similarity(flat_weights)
    
    np.fill_diagonal(cs, 0) # ignore self-similarity
    
    scores = []
    for i in range(n):
        max_sim = np.max(cs[i])
        # FoolsGold penalty: 1.0 - max_similarity. 
        # If similarity is 0.99, score is 0.01 (heavily penalized).
        score = 1.0 - max(0.0, max_sim)
        scores.append(score)
        
    return scores



def trust_weighted_mean(weights_list, trust_scores=None):
    """
    Averages weights using a Blockchain Trust Score as the aggregator weight.
    Clients with < 50 score are effectively penalized.
    Clients with < 20 score should have been dropped before this function.
    """
    if trust_scores is None or len(trust_scores) != len(weights_list):
        return simple_mean(weights_list)

    print(f"  [🛡️ TRUST AGGREGATION] Using Trust Scores: {trust_scores}")
    total_trust = sum(trust_scores)
    if total_trust == 0:
        return simple_mean(weights_list)
        
    avg = []
    for layer_idx in range(len(weights_list[0])):
        layer_sum = np.zeros_like(weights_list[0][layer_idx], dtype=np.float64)
        for i, w in enumerate(weights_list):
            layer_sum += w[layer_idx] * trust_scores[i]
        avg.append(layer_sum / total_trust)
    return avg


# ======================================================
# ROBUST AGGREGATION ROUTER
# ======================================================
def aggregate_weights(weights_list, sample_counts=None, trust_scores=None):
    """
    Route to the requested aggregation method based on configuration.
    """
    if not DEFENSE_ENABLED or DEFENSE_METHOD == "mean":
        return simple_mean(weights_list, sample_counts)

    if DEFENSE_METHOD == "median":
        return coordinate_median(weights_list)

    if DEFENSE_METHOD == "trimmed":
        return trimmed_mean(weights_list, TRIM_RATIO)

    if DEFENSE_METHOD == "krum":
        # Assumes 1 malicious client max for krum parameter f
        return krum_aggregation(weights_list, f=1)

    if DEFENSE_METHOD == "multi_krum":
        return multi_krum_aggregation(weights_list, f=1)

    if DEFENSE_METHOD == "trust_weighted":
        return trust_weighted_mean(weights_list, trust_scores)

    raise ValueError(f"Unknown defense method: {DEFENSE_METHOD}")
