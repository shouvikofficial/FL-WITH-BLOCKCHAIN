import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef
)


# ======================================================
# EXTENDED MODEL EVALUATION
# ======================================================
def evaluate_model(model, X, y):
    """
    Returns: accuracy, f1, precision, recall, roc_auc, mcc
    - ROC-AUC: area under curve (class probability needed)
    - MCC: Matthews Correlation Coefficient (best for imbalanced binary)
    """
    y_pred = model.predict(X)

    acc       = accuracy_score(y, y_pred)
    f1        = f1_score(y, y_pred, average="binary", zero_division=0)
    precision = precision_score(y, y_pred, average="binary", zero_division=0)
    recall    = recall_score(y, y_pred, average="binary", zero_division=0)
    mcc       = matthews_corrcoef(y, y_pred)

    # ROC-AUC requires probability scores
    roc_auc = 0.0
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, y_proba)
        except Exception:
            roc_auc = 0.0

    return acc, f1, precision, recall, roc_auc, mcc
