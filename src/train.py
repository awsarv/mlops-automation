from DecisionTreeRegressor import run_decision_tree_regression
from LinearRegression import run_linear_regression as train_linear_regression
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import argparse

# ----- Parse --max_depth argument to set the hyperparameter of DecisionTreeRegressor -----
parser = argparse.ArgumentParser()
parser.add_argument(
    "--max_depth", type=int, default=5, help="Max depth for DecisionTreeRegressor"
)
args = parser.parse_args()
max_depth = args.max_depth
print(f"max_depth set to {max_depth}")


# ----- Load Data -----
df = pd.read_csv("data/california_housing.csv")
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a new MLflow Experiment
mlflow.set_experiment("California-Housing-Regression")


results = []

# ----- Linear Regression -----
with mlflow.start_run(run_name="LinearRegression") as run:
    model, model_name, mse, r2, avg_inference_time = train_linear_regression(
        X_train, y_train, X_test, y_test
    )
    mlflow.log_param("model", model_name)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("avg_inference_time", avg_inference_time)

    mlflow.set_tag(
        "Training Info",
        "This run trains a Linear Regression model on California housing data.",
    )

    mlflow.sklearn.log_model(
        sk_model=model, name=model_name, input_example=X_test.iloc[:5]
    )
    results.append((model_name, model, mse, r2))
    print(
        f"Linear Regression: MSE={(mse * 100):.2f}%, R2={(r2 * 100):.2f}%, "
        f"Avg inference time={avg_inference_time:.6f}s"
    )


# ----- Decision Tree -----
with mlflow.start_run(run_name="DecisionTreeRegressor") as run:
    model, model_name, mse, r2, avg_inference_time = run_decision_tree_regression(
        X_train, y_train, X_test, y_test, max_depth=max_depth
    )
    mlflow.log_param("model", model_name)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("avg_inference_time", avg_inference_time)

    mlflow.set_tag(
        "Training Info",
        "This run trains a Decision Tree Regressor model on California housing data.",
    )

    mlflow.sklearn.log_model(
        sk_model=model, name=model_name, input_example=X_test.iloc[:5]
    )
    results.append((model_name, model, mse, r2))
    print(
        f"Decision Tree: MSE={(mse * 100):.2f}%, R2={(r2 * 100):.2f}%, "
        f"Avg inference time={avg_inference_time:.6f}s"
    )

# ----- Select Best Model -----
best_result = min(results, key=lambda x: x[2])  # Choose best by lowest MSE
best_model_name, best_model, best_mse, best_r2 = best_result

joblib.dump(best_model, "models/best_model.pkl")
print(
    f"✅ Best model: {best_model_name} "
    f"(MSE={(best_mse * 100):.2f}%, R2={(best_r2 * 100):.2f}%) "
    "saved for deployment."
)

# ----- Register best model in MLflow Model Registry (Bonus) -----
with mlflow.start_run(run_name=f"{best_model_name}-Best-Register"):

    mlflow.set_tag(
        "Training Info", "This run registers the best performing model for deployment."
    )
    mlflow.set_tag("Model Type", best_model_name)
    mlflow.set_tag(
        "Max Depth", max_depth if best_model_name == "DecisionTreeRegressor" else None
    )
    mlflow.set_tag("Framework", "scikit-learn")
    mlflow.set_tag("Dataset", "california_housing.csv")
    mlflow.set_tag(
        "Purpose",
        "This model predicts California housing prices using regression algorithms and logs the best performing model for deployment.",
    )

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name=best_model_name,
        registered_model_name="BestHousingModel",
        input_example=X_test.iloc[:5],
        signature=mlflow.models.infer_signature(
            X_test.iloc[:5], {
                "prediction": best_model.predict(X_test.iloc[:5])}
        ),
    )


print(
    "Training complete! "
    "Check MLflow UI on http://<your-ec2-ip>:5000 "
    "to compare model performance."
)
