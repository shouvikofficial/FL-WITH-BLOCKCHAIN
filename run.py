"""
run.py — Launch the Federated Learning server from the project root.
Usage:
    venv\Scripts\python.exe run.py
"""
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import sys
import os

# Ensure the project root is always on the Python path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from server.server import run_federated_learning

if __name__ == "__main__":
    run_federated_learning()
