import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ======================================================
# FEATURE INTERACTIONS (MUST MATCH data_loader.py)
# ======================================================
def add_interactions(df):
    """
    5 interaction features — must match data_loader.py exactly.
    Applied client-side when global_scaler is used (round 2+).
    """
    df = df.copy()
    df["age_glucose"]     = df["age"]       * df["glucose"]
    df["age_BMI"]         = df["age"]       * df["BMI"]
    df["glucose_BMI"]     = df["glucose"]   * df["BMI"]
    df["heartRate_exang"] = df["heartRate"] * df["exang"]
    df["chol_fbs"]        = df["chol"]      * df["fbs"]
    return df


# ======================================================
# DATA POISONING SIMULATION (label-flip)
# ======================================================
def inject_fake_data(X, y, poison_ratio=0.15):
    """
    Label-flipping attack — simulates a malicious client.
    Reduced from 0.30 → 0.15 to reflect realistic threat level.
    """
    n_poison = int(len(y) * poison_ratio)
    if n_poison == 0:
        return X, y

    idx = np.random.choice(len(y), n_poison, replace=False)
    y_poisoned = y.copy()
    y_poisoned[idx] = 1 - y_poisoned[idx]

    return X, y_poisoned


# ======================================================
# LOCAL PREPROCESSING (FEDERATED-SAFE)
# ======================================================
def local_preprocess(df, global_scaler=None, apply_smote=True):
    """
    Federated-safe preprocessing pipeline:
      1. Convert list → DataFrame if needed
      2. Fill missing values
      3. Add interaction features (if raw columns available)
      4. Scale (global or local)
      5. Probabilistic data poisoning (15% chance, 15% flip rate)
      6. SMOTE if class imbalance > 2.1:1
    """

    # ---------------------------------
    # 1. Convert list of (x, y) → DataFrame
    # ---------------------------------
    if isinstance(df, list):
        X_arr = np.array([x for x, _ in df], dtype=float)
        y_arr = np.array([y for _, y in df], dtype=int)
        df = pd.DataFrame(X_arr)
        df["target"] = y_arr

    # ---------------------------------
    # 2. Missing values
    # ---------------------------------
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # ---------------------------------
    # 3. Feature interactions (if named columns exist)
    # ---------------------------------
    named_cols = {"age", "glucose", "BMI", "heartRate", "exang", "chol", "fbs"}
    if named_cols.issubset(set(df.columns)):
        df = add_interactions(df)

    X = df.drop("target", axis=1)
    y = df["target"]

    # ---------------------------------
    # 4. Scaling
    # ---------------------------------
    if global_scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scaler_params = {
            "mean": scaler.mean_,
            "var":  scaler.var_,
            "n":    len(X)
        }
    else:
        mean, var = global_scaler
        X_scaled = (X.values - mean) / np.sqrt(var + 1e-8)
        scaler_params = None

    # ---------------------------------
    # 5. 🔥 DATA POISONING (15% chance, reduced from 30%)
    # ---------------------------------
    if np.random.random() < 0.15:
        X_scaled, y_vals = inject_fake_data(X_scaled, y.values, poison_ratio=0.15)
        y = pd.Series(y_vals)

    # ---------------------------------
    # 6. SMOTE (client-side, if imbalanced)
    # ---------------------------------
    if apply_smote:
        class_counts = y.value_counts()
        if len(class_counts) > 1:
            class_ratio = class_counts.max() / class_counts.min()
            if class_ratio >= 2.1:
                X_scaled, y = SMOTE(random_state=42).fit_resample(X_scaled, y)

    return X_scaled, np.array(y), scaler_params
