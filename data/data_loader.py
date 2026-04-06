import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ======================================================
# FEATURE INTERACTIONS (PERFORMANCE BOOST)
# ======================================================
def add_interactions(df):
    """
    Add 5 domain-informed interaction features.
    These are validated in the centralized baseline (centralized.py).
    """
    df = df.copy()
    df["age_glucose"]     = df["age"]       * df["glucose"]
    df["age_BMI"]         = df["age"]       * df["BMI"]
    df["glucose_BMI"]     = df["glucose"]   * df["BMI"]
    df["heartRate_exang"] = df["heartRate"] * df["exang"]
    df["chol_fbs"]        = df["chol"]      * df["fbs"]
    return df


# ======================================================
# SERVER VALIDATION DATA LOADER
# ======================================================
def load_server_validation_data(path, label_column, samples=100):
    """
    Load a small hold-out set for the server to perform validation
    checks against potentially malicious clients.
    """
    df = pd.read_excel(path)
    df = df.dropna()

    if all(c in df.columns for c in ["age", "glucose", "BMI", "heartRate", "exang", "chol", "fbs"]):
        df = add_interactions(df)

    X = df.drop(label_column, axis=1).values
    y = df[label_column].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Take the last `samples` as server validation data
    return X[-samples:], y[-samples:]


# ======================================================
# DATA LOADER WITH INTERACTIONS
# ======================================================
def load_and_split_data(path, label_column, num_clients=3):
    # Load Excel
    df = pd.read_excel(path)
    df = df.dropna()

    # 🔥 ADD INTERACTIONS BEFORE SPLITTING
    if all(c in df.columns for c in ["age", "glucose", "BMI", "heartRate", "exang", "chol", "fbs"]):
        df = add_interactions(df)

    X = df.drop(label_column, axis=1).values
    y = df[label_column].values

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Shuffle indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Split indices evenly across clients
    split_indices = np.array_split(indices, num_clients)

    # Create client datasets (LISTS, not NumPy arrays)
    client_data = []
    for idx in split_indices:
        client_data.append(
            [(X[i], y[i]) for i in idx]
        )

    return client_data
