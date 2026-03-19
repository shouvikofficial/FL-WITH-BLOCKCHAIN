import numpy as np

# ======================================================
# DEFENSE CONFIGURATION
# ======================================================
DEFENSE_ENABLED = True     # 🔁 Toggle defense ON/OFF
DEFENSE_METHOD = "median"  # options: "mean", "median", "trimmed"

TRIM_RATIO = 0.2           # used only for trimmed mean (discards top/bottom 20%)

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


# ======================================================
# ROBUST AGGREGATION ROUTER
# ======================================================
def aggregate_weights(weights_list, sample_counts=None):
    """
    Route to the requested aggregation method based on configuration.
    """
    if not DEFENSE_ENABLED or DEFENSE_METHOD == "mean":
        return simple_mean(weights_list, sample_counts)

    if DEFENSE_METHOD == "median":
        return coordinate_median(weights_list)

    if DEFENSE_METHOD == "trimmed":
        return trimmed_mean(weights_list, TRIM_RATIO)

    raise ValueError(f"Unknown defense method: {DEFENSE_METHOD}")
