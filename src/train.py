import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time

# Use env var for MLflow URI, default to ./mlruns for local, /mlflow/mlruns for docker
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:mlruns"))

df = pd.read_csv("data/california_housing.csv")
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

mlflow.set_experiment("California-Housing-Regression")
results = []

# ----- Linear Regression -----
with mlflow.start_run(run_name="LinearRegression") as run:
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    start_time = time.time()
    _ = model.predict(X_test)
    elapsed_time = time.time() - start_time
    avg_inference_time = elapsed_time / len(X_test)
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("avg_inference_time", avg_inference_time)
    mlflow.sklearn.log_model(model, "model")
    results.append(("LinearRegression", model, mse, r2))
    print(
        f"Linear Regression: MSE={mse:.4f}, R2={r2:.4f}, "
        f"Avg inference time={avg_inference_time:.6f}s"
    )

# ----- Decision Tree -----
max_depth = 5  # You can tune this for more runs
with mlflow.start_run(run_name="DecisionTreeRegressor") as run:
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    start_time = time.time()
    _ = model.predict(X_test)
    elapsed_time = time.time() - start_time
    avg_inference_time = elapsed_time / len(X_test)
    mlflow.log_param("model", "DecisionTreeRegressor")
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("avg_inference_time", avg_inference_time)
    mlflow.sklearn.log_model(model, "model")
    results.append(("DecisionTreeRegressor", model, mse, r2))
    print(
        f"Decision Tree: MSE={mse:.4f}, R2={r2:.4f}, "
        f"Avg inference time={avg_inference_time:.6f}s"
    )

# ----- Select Best Model -----
best = min(results, key=lambda x: x[2])  # Choose best by lowest MSE
best_model_name, best_model, best_mse, best_r2 = best

joblib.dump(best_model, "models/best_model.pkl")
print(
    f"Best model: {best_model_name} "
    f"(MSE={best_mse:.4f}, R2={best_r2:.4f}) "
    "saved for deployment."
)

# ----- Register best model in MLflow Model Registry (Bonus) -----
with mlflow.start_run(
    run_name=f"{best_model_name}-Best-Register"
):
    mlflow.sklearn.log_model(
        best_model,
        "model",
        registered_model_name="BestHousingModel"
    )

print(
    "Training complete! "
    "Check MLflow UI on http://<your-ec2-ip>:5000 "
    "to compare model performance."
)
