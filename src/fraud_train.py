"""
Fraud Detection Model Training Pipeline.

This module implements the training pipeline for credit card fraud detection
using ensemble machine learning methods. It supports multiple model types
with MLflow experiment tracking for reproducibility.

Training Pipeline:
    1. Load and validate transaction dataset
    2. Train multiple classifiers (Logistic Regression, Random Forest)
    3. Evaluate models using fraud-specific metrics (F1, ROC-AUC)
    4. Select best model based on F1 score
    5. Register model in MLflow for deployment

Models:
    - Logistic Regression: Baseline model with class balancing
    - Random Forest: Ensemble model for improved fraud detection

Version: 1.0.0
"""

import os
import time
import json
import hashlib
import pathlib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import joblib


# MLflow Configuration
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:mlruns"))
mlflow.set_experiment("Fraud-Detection-Classification")

DATA_FP = pathlib.Path("data/credit_card_transactions.csv")
MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading fraud detection dataset...")
df = pd.read_csv(DATA_FP)

# Features and target
FEATURE_COLS = [
    'amount', 'hour', 'day_of_week', 'merchant_category',
    'distance_from_home', 'distance_from_last_transaction',
    'ratio_to_median_purchase', 'repeat_retailer',
    'used_chip', 'used_pin', 'online_order'
]

X = df[FEATURE_COLS]
y = df['is_fraud']

# Train-test split (stratified to maintain fraud ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Fraud rate in training: {y_train.mean()*100:.2f}%")

# Log dataset hash for reproducibility
with open(DATA_FP, "rb") as f:
    data_sha256 = hashlib.sha256(f.read()).hexdigest()

results = []


def run_and_log(model_name, model):
    """
    Train a model and log metrics to MLflow.

    Args:
        model_name: Identifier for the model type
        model: Sklearn classifier instance

    Returns:
        Tuple of (model_name, trained_model, f1_score, roc_auc)
    """
    with mlflow.start_run(run_name=model_name):
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print('='*50)

        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predict
        t0 = time.time()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - t0

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        avg_inference_time = inference_time / max(len(X_test), 1)

        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))

        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.log_metric("avg_inference_time", avg_inference_time)
        mlflow.log_metric("true_negatives", int(cm[0, 0]))
        mlflow.log_metric("false_positives", int(cm[0, 1]))
        mlflow.log_metric("false_negatives", int(cm[1, 0]))
        mlflow.log_metric("true_positives", int(cm[1, 1]))

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        print(f"Confusion Matrix:")
        print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
        print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
        print(f"Training time: {train_time:.2f}s")
        print(f"Avg inference: {avg_inference_time*1000:.4f}ms")

        return model_name, model, f1, roc_auc


# Train models
print("\n" + "="*60)
print("FRAUD DETECTION MODEL TRAINING")
print("="*60)

# Model 1: Logistic Regression (baseline)
results.append(run_and_log(
    "LogisticRegression",
    LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
))

# Model 2: Random Forest (better for imbalanced data)
results.append(run_and_log(
    "RandomForest",
    RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
))

# Select best model by F1 score (important for imbalanced classification)
best_model_name, best_model, best_f1, best_auc = max(results, key=lambda x: x[2])

print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name}")
print(f"F1 Score: {best_f1:.4f}, ROC AUC: {best_auc:.4f}")
print("="*60)

# Save best model for deployment
joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

# Save feature order for inference
with open(MODELS_DIR / "feature_order.json", "w") as f:
    json.dump(FEATURE_COLS, f)

# Save model metadata
metadata = {
    "model_name": best_model_name,
    "task": "fraud_detection",
    "f1_score": best_f1,
    "roc_auc": best_auc,
    "features": FEATURE_COLS,
    "data_hash": data_sha256
}
with open(MODELS_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nModel saved to {MODELS_DIR / 'best_model.pkl'}")
print(f"Feature order saved to {MODELS_DIR / 'feature_order.json'}")

# Register best model in MLflow
with mlflow.start_run(run_name=f"{best_model_name}-Production"):
    mlflow.log_params({
        "selected_by": "f1_score",
        "best_f1": best_f1,
        "best_roc_auc": best_auc,
    })
    mlflow.sklearn.log_model(
        best_model,
        "model",
        registered_model_name="FraudDetectionModel",
    )

print("\nTraining complete. Model ready for deployment.")
