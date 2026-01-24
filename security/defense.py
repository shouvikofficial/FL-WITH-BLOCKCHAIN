import numpy as np

# ======================================================
# DEFENSE CONFIGURATION
# ======================================================
DEFENSE_ENABLED = True     # 🔁 Toggle defense ON/OFF
DEFENSE_METHOD = "median" # options: "mean", "median", "trimmed"

TRIM_RATIO = 0.2           # used only for trimmed mean

# ======================================================
# ROBUST AGGREGATION
# ======================================================
def aggregate_weights(weights_list):
    """
    Aggregate client weights using a robust defense strategy.
    """

    if not DEFENSE_ENABLED or DEFENSE_METHOD == "mean":
        return simple_mean(weights_list)

    if DEFENSE_METHOD == "median":
        return coordinate_median(weights_list)

    if DEFENSE_METHOD == "trimmed":
        return trimmed_mean(weights_list)

    raise ValueError("Unknown defense method")
