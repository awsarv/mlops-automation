import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time

df = pd.read_csv("data/california_housing.csv")
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("California-Housing-Regression")

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
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("avg_inference_time", avg_inference_time)
    mlflow.sklearn.log_model(model, "model")
    joblib.dump(model, "models/best_model.pkl")
    print(
        "Linear Regression: MSE={:.4f}, R2={:.4f}, "
        "Avg inference time={:.6f}s".format(
            mse, r2, avg_inference_time
        )
    )

with mlflow.start_run(run_name="DecisionTreeRegressor") as run:
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    start_time = time.time()
    _ = model.predict(X_test)
    elapsed_time = time.time() - start_time
    avg_inference_time = elapsed_time / len(X_test)
    mlflow.log_param("model", "DecisionTreeRegressor")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("avg_inference_time", avg_inference_time)
    mlflow.sklearn.log_model(model, "model")
    if r2 > 0.7:
        joblib.dump(model, "models/best_model.pkl")
    print(
        "Decision Tree: MSE={:.4f}, R2={:.4f}, "
        "Avg inference time={:.6f}s".format(
            mse, r2, avg_inference_time
        )
    )

print(
    "Training complete! "
    "Check MLflow UI to compare model performance and speed."
)
