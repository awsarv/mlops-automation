import os
import time
import json
import hashlib
import pathlib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# Respect env if server is used; default to local file store
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:mlruns"))
mlflow.set_experiment("California-Housing-Regression")

DATA_FP = pathlib.Path("data/california_housing.csv")
MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_FP)
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Log dataset hash (reproducibility)
with open(DATA_FP, "rb") as f:
    data_sha256 = hashlib.sha256(f.read()).hexdigest()

results = []


def run_and_log(model_name, model):
    with mlflow.start_run(run_name=model_name):
        if hasattr(model, "random_state"):
            model.random_state = 42
        model.fit(X_train, y_train)

        t0 = time.time()
        preds = model.predict(X_test)
        inf_elapsed = time.time() - t0

        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, preds)
        avg_inf_time = inf_elapsed / max(len(X_test), 1)

        mlflow.log_param("model_name", model_name)
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())
        mlflow.log_param("data_sha256", data_sha256)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("avg_inference_time", avg_inf_time)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(
            f"{model_name}: MSE={mse:.4f}, RMSE={rmse:.4f}, "
            f"R2={r2:.4f}, avg_inf={avg_inf_time:.6f}s"
        )
        return model_name, model, mse, r2


# Train two baseline models
results.append(run_and_log("LinearRegression", LinearRegression()))
results.append(
    run_and_log(
        "DecisionTreeRegressor",
        DecisionTreeRegressor(max_depth=5),
    )
)

# Pick best by lowest MSE
best_model_name, best_model, best_mse, best_r2 = min(
    results, key=lambda x: x[2]
)

# Persist artifacts for serving
joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
feature_order = list(X.columns)
with open(MODELS_DIR / "feature_order.json", "w") as f:
    json.dump(feature_order, f)

print(
    f"Best model: {best_model_name} "
    f"(MSE={best_mse:.4f}, R2={best_r2:.4f}) saved for deployment."
)

# Register best in the MLflow Model Registry (works when using server)
with mlflow.start_run(run_name=f"{best_model_name}-Best-Register"):
    mlflow.log_params(
        {
            "selected_by": "mse",
            "best_mse": best_mse,
            "best_r2": best_r2,
        }
    )
    mlflow.sklearn.log_model(
        best_model,
        "model",
        registered_model_name="BestHousingModel",
    )

print("Training complete.")
