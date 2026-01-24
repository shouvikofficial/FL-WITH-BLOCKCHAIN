import pandas as pd
import numpy as np
import os
import warnings
import joblib
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import shap
from lime.lime_tabular import LimeTabularExplainer


# ======================================================
# UTILS
# ======================================================
def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def add_interactions(df):
    df = df.copy()
    df["age_glucose"] = df["age"] * df["glucose"]
    df["age_BMI"] = df["age"] * df["BMI"]
    df["glucose_BMI"] = df["glucose"] * df["BMI"]
    df["heartRate_exang"] = df["heartRate"] * df["exang"]
    df["chol_fbs"] = df["chol"] * df["fbs"]
    return df


# ======================================================
# CENTRALIZED RETRAINING
# ======================================================
def retrain_model():

    MAIN_FILE = "data/dataset.xlsx"
    NEW_FILE = "NewData.xlsx"

    if not os.path.exists(MAIN_FILE):
        raise FileNotFoundError("❌ MainData.xlsx not found")

    main_df = pd.read_excel(MAIN_FILE)
    new_df = pd.read_excel(NEW_FILE) if os.path.exists(NEW_FILE) else pd.DataFrame()

    df = pd.concat([main_df, new_df], ignore_index=True)
    df = df.dropna(subset=["target"])

    if df.empty:
        raise ValueError("❌ No valid data found")

    feature_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "heartRate", "exang", "oldpeak", "BMI", "diaBP", "glucose", "Smkr"
    ]

    X = df[feature_columns].copy()
    y = df["target"].astype(int)

    X.fillna(X.mean(numeric_only=True), inplace=True)
    X = add_interactions(X)

    # ---------------- Scaling + SMOTE ----------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class_ratio = y.value_counts().max() / y.value_counts().min()
    if class_ratio >= 2.1:
        smote = SMOTE(random_state=42)
        X_scaled, y = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ---------------- Models ----------------
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss"),
        "GradientBoosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
        "SVC": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "MLP": MLPClassifier(max_iter=1000)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)
            y_proba = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()

        results.append({
            "name": name,
            "model": model,
            "f1": f1_score(y_test, y_pred),
            "roc": roc_auc_score(y_test, y_proba)
        })

    # ---------------- Ensemble ----------------
    top3 = sorted(results, key=lambda x: x["f1"], reverse=True)[:3]
    ensemble = VotingClassifier(
        estimators=[(r["name"], r["model"]) for r in top3],
        voting="soft"
    )
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]

    print("\n📊 FINAL ENSEMBLE PERFORMANCE")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    # ---------------- SHAP ----------------
    try:
        explainer = shap.Explainer(top3[0]["model"], X_train)
        shap_values = explainer(X_test[:100])
        shap.summary_plot(shap_values, X_test[:100], show=False)
        plot_to_base64()
        print("✅ SHAP generated")
    except Exception as e:
        print("⚠️ SHAP failed:", e)

    # ---------------- LIME ----------------
    try:
        lime_explainer = LimeTabularExplainer(
            X_train,
            feature_names=X.columns.tolist(),
            class_names=["No Risk", "Risk"],
            mode="classification"
        )
        lime_explainer.explain_instance(
            X_test[0], ensemble.predict_proba
        ).as_pyplot_figure()
        plot_to_base64()
        print("✅ LIME generated")
    except Exception as e:
        print("⚠️ LIME failed:", e)

    # ---------------- SAVE ----------------
    joblib.dump(ensemble, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Model & scaler saved")

    df.to_excel(MAIN_FILE, index=False)
    pd.DataFrame(columns=df.columns).to_excel(NEW_FILE, index=False)
    print("✅ Data merged & NewData cleared")


# ======================================================
# ENTRY
# ======================================================
if __name__ == "__main__":
    retrain_model()
