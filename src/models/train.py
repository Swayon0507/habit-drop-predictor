import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                             brier_score_loss, confusion_matrix)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR  = "reports/figures"
MODEL_DIR   = "src/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "roll7_rate", "roll14_rate", "roll3_rate",
    "streak", "days_since_miss", "weekend_penalty",
    "habit_age", "is_weekend", "dayofweek", "mood",
    "habit_encoded"
]
TARGET_COL = "will_drop"

def load_data():
    print("\n[1] Loading combined dataset")
    df = pd.read_csv("data/processed/combined_dataset.csv")
    print(f"    Shape: {df.shape}")
    print(f"    Drop events: {df[TARGET_COL].sum()} ({df[TARGET_COL].mean()*100:.1f}%)")
    return df

def prepare_features(df):
    print("\n[2] Preparing features")
    le = LabelEncoder()
    df["habit_encoded"] = le.fit_transform(df["habit"])
    df["mood"] = df["mood"].fillna(df["mood"].median())
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    groups = df["user_id"]
    print(f"    X shape: {X.shape} | y shape: {y.shape}")
    return X, y, groups, le

def split_data(X, y, groups):
    print("\n[3] Splitting data by user groups")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"    Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"    Train drop rate: {y_train.mean()*100:.1f}% | Test drop rate: {y_test.mean()*100:.1f}%")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("\n[4] Applying SMOTE to fix class imbalance...")
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"    Before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"    After  SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

    print("\n[5] Training XGBoost model with calibration")
    base_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    )
    calibrated_model = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
    calibrated_model.fit(X_res, y_res)
    print("    Model trained with SMOTE + probability calibration")
    return calibrated_model

def evaluate_model(model, X_test, y_test):
    print("\n[5] Evaluating model")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    brier   = brier_score_loss(y_test, y_proba)
    print(f"\n    AUC-ROC Score : {auc:.4f}")
    print(f"    Brier Score   : {brier:.4f}")
    print(f"\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stable","Will Drop"]))
    print(f"    Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    return y_proba, auc, brier

def plot_feature_importance(model, X_train):
    print("\n[6] Generating SHAP plots")
    try:
        base = model.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(base)
        shap_vals = explainer.shap_values(X_train)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_train, feature_names=FEATURE_COLS, show=False, plot_type="bar")
        plt.title("Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_feature_importance.png", dpi=150)
        plt.close()
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_train, feature_names=FEATURE_COLS, show=False)
        plt.title("SHAP Summary Plot")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150)
        plt.close()
        print(f"    Saved SHAP plots to {OUTPUT_DIR}/")
    except Exception as e:
        print(f"    SHAP skipped: {e}")

def plot_risk_distribution(y_proba, y_test):
    plt.figure(figsize=(10, 5))
    plt.hist(y_proba[y_test==0], bins=30, alpha=0.6, label="Stable",    color="green")
    plt.hist(y_proba[y_test==1], bins=30, alpha=0.6, label="Will Drop", color="red")
    plt.axvline(x=0.5, color="black", linestyle="--", label="Threshold 0.5")
    plt.xlabel("Predicted Drop Probability")
    plt.ylabel("Count")
    plt.title("Risk Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/risk_distribution.png", dpi=150)
    plt.close()
    print(f"    Saved: {OUTPUT_DIR}/risk_distribution.png")

def save_model(model, le):
    print("\n[7] Saving model")
    joblib.dump(model, f"{MODEL_DIR}/habit_drop_model.pkl")
    joblib.dump(le,    f"{MODEL_DIR}/label_encoder.pkl")
    print(f"    Saved: {MODEL_DIR}/habit_drop_model.pkl")

def run_training():
    print("="*60)
    print("  HABIT DROP PREDICTOR - MODEL TRAINING")
    print("="*60)
    df                               = load_data()
    X, y, groups, le                 = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y, groups)
    model                            = train_model(X_train, y_train)
    y_proba, auc, brier              = evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, X_train)
    plot_risk_distribution(y_proba, y_test)
    save_model(model, le)
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print(f"  AUC-ROC: {auc:.4f}")
    print("  Next step: streamlit dashboard")
    print("="*60)

if __name__ == "__main__":
    run_training()
