import numpy as np
from blockchain.blockchain import log_update

dummy_weights = [
    np.array([1.0, 2.0, 3.0]),
    np.array([0.5])
]

log_update("Hospital_1", dummy_weights)

print("✅ Blockchain logging SUCCESSFUL")
