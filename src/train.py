from DecisionTreeRegressor import run_decision_tree_regression
from LinearRegression import run_linear_regression as train_linear_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import argparse
import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import os
import pandas as pd
import psutil
import seaborn as sns
import time
import warnings

warnings.filterwarnings('ignore')

# Set matplotlib backend for Windows compatibility
matplotlib.use('Agg')

# ----- Parse arguments for hyperparameter tuning -----
parser = argparse.ArgumentParser()
parser.add_argument(
    "--max_depth_list",
    nargs='+',
    type=int,
    default=[3, 5, 7, 10, 15],
    help="List of max depths for DecisionTreeRegressor"
)
parser.add_argument(
    "--test_size",
    type=float,
    default=0.2,
    help="Test size for train-test split"
)
parser.add_argument(
    "--random_state",
    type=int,
    default=42,
    help="Random state for reproducibility"
)
parser.add_argument(
    "--cv_folds",
    type=int,
    default=5,
    help="Number of cross-validation folds"
)
args = parser.parse_args()

print(f"Hyperparameter grid: max_depth_list={args.max_depth_list}")
print(f"Test size: {args.test_size}")
print(f"CV folds: {args.cv_folds}")


def log_system_metrics():
    """Log system metrics to MLflow"""
    # CPU and memory usage
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    mlflow.log_metric("system_cpu_percent", cpu_percent)
    mlflow.log_metric("system_memory_percent", memory.percent)
    mlflow.log_metric("system_memory_available_gb", memory.available / (1024**3))

    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_available_gb": memory.available / (1024**3)
    }


def log_data_profile(X, y, split_name=""):
    """Log data profiling metrics"""
    prefix = f"data_{split_name}_" if split_name else "data_"

    # Data shape and basic stats
    mlflow.log_metric(f"{prefix}samples", len(X))
    mlflow.log_metric(f"{prefix}features", X.shape[1])
    mlflow.log_metric(f"{prefix}target_mean", y.mean())
    mlflow.log_metric(f"{prefix}target_std", y.std())
    mlflow.log_metric(f"{prefix}target_min", y.min())
    mlflow.log_metric(f"{prefix}target_max", y.max())


def create_learning_curve_plot(model, X_train, y_train, cv_folds, model_name):
    """Create and log learning curve plot"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=cv_folds,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error', n_jobs=-1
    )

    train_scores_mean = -train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    val_scores_std = val_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training MSE')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='blue')

    plt.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation MSE')
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color='red')

    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Learning Curve - {model_name}')
    plt.legend()
    plt.grid(True)

    # Save plot with better Windows compatibility
    plot_filename = f"learning_curve_{model_name.replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    try:
        mlflow.log_artifact(plot_filename, "plots")
        os.remove(plot_filename)  # Clean up after logging
    except Exception as e:
        print(f"Warning: Could not log learning curve plot: {e}")

    plt.close()

    # Log learning curve metrics
    for i, size in enumerate(train_sizes):
        mlflow.log_metric(f"learning_curve_train_mse", train_scores_mean[i], step=int(size))
        mlflow.log_metric(f"learning_curve_val_mse", val_scores_mean[i], step=int(size))



def create_residual_plot(y_true, y_pred, model_name):
    """Create and log residual plot"""
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'Residuals vs Predicted - {model_name}')
    ax1.grid(True)

    # Residuals histogram
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Residuals Distribution - {model_name}')
    ax2.grid(True)

    plt.tight_layout()

    # Save plot with better Windows compatibility
    plot_filename = f"residuals_{model_name.replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    try:
        mlflow.log_artifact(plot_filename, "plots")
        os.remove(plot_filename)
    except Exception as e:
        print(f"Warning: Could not log residual plot: {e}")

    plt.close()


def create_advanced_model_analysis(X_train, y_train, X_test, y_test, model_name):
    """Create advanced model analysis with additional visualizations not covered by other functions"""

    try:
        # 1. Feature Correlation Matrix
        correlation_matrix = X_train.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True,
                    cmap='coolwarm', center=0, fmt='.2f')
        plt.title(f'Feature Correlation Matrix - {model_name}')
        plt.tight_layout()

        plot_filename = f"correlation_matrix_{model_name.replace(' ', '_').replace('-', '_')}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        try:
            mlflow.log_artifact(plot_filename, "advanced_plots")
            os.remove(plot_filename)
        except Exception as e:
            print(f"Warning: Could not log correlation matrix: {e}")
        plt.close()

        # 2. Validation Curves for Decision Tree (if applicable)
        if "DecisionTree" in model_name or "Tree" in model_name:
            param_range = np.arange(1, 16)  # max_depth from 1 to 15

            train_scores, val_scores = validation_curve(
                DecisionTreeRegressor(
                    random_state=42,
                    min_samples_split=2,
                    min_samples_leaf=1
                ), X_train, y_train,
                param_name='max_depth', param_range=param_range,
                cv=3, scoring='neg_mean_squared_error', n_jobs=-1
            )

            train_scores_mean = -train_scores.mean(axis=1)
            train_scores_std = train_scores.std(axis=1)
            val_scores_mean = -val_scores.mean(axis=1)
            val_scores_std = val_scores.std(axis=1)

            # Log validation curve metrics
            for i, depth in enumerate(param_range):
                mlflow.log_metric("validation_train_mse", train_scores_mean[i], step=depth)
                mlflow.log_metric("validation_val_mse", val_scores_mean[i], step=depth)

            # Plot validation curve
            plt.figure(figsize=(12, 8))
            plt.plot(param_range, train_scores_mean, 'o-', color='blue', label='Training MSE')
            plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='blue')

            plt.plot(param_range, val_scores_mean, 'o-', color='red', label='Validation MSE')
            plt.fill_between(param_range, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color='red')

            plt.xlabel('Max Depth')
            plt.ylabel('Mean Squared Error')
            plt.title(f'Validation Curve - {model_name}')
            plt.legend()
            plt.grid(True)

            plot_filename = f"validation_curve_{model_name.replace(' ', '_').replace('-', '_')}.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            try:
                mlflow.log_artifact(plot_filename, "advanced_plots")
                os.remove(plot_filename)
            except Exception as e:
                print(f"Warning: Could not log validation curve: {e}")
            plt.close()

        # 3. Distribution Analysis (Target, Actual vs Predicted, Residuals)
        # Get predictions for distribution analysis using the existing function
        temp_model, _, _, _, _ = run_decision_tree_regression(
            X_train, y_train, X_test, y_test, max_depth=10, random_state=42
        )
        y_pred_dist = temp_model.predict(X_test)

        plt.figure(figsize=(15, 5))

        # Target distribution
        plt.subplot(1, 3, 1)
        plt.hist(y_train, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Target Values')
        plt.ylabel('Frequency')
        plt.title(f'Target Distribution - {model_name}')
        plt.grid(True, alpha=0.3)

        # Actual vs Predicted scatter
        plt.subplot(1, 3, 2)
        plt.scatter(y_test, y_pred_dist, alpha=0.6, color='green')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted - {model_name}')
        plt.grid(True, alpha=0.3)

        # Residuals distribution
        plt.subplot(1, 3, 3)
        residuals_dist = y_test - y_pred_dist
        plt.hist(residuals_dist, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title(f'Residuals Distribution - {model_name}')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_filename = f"distribution_analysis_{model_name.replace(' ', '_').replace('-', '_')}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        try:
            mlflow.log_artifact(plot_filename, "advanced_plots")
            os.remove(plot_filename)
        except Exception as e:
            print(f"Warning: Could not log distribution analysis: {e}")
        plt.close()

        # Log advanced metrics
        mlflow.log_metric("correlation_max", correlation_matrix.abs().max().max())
        mlflow.log_metric("correlation_mean", correlation_matrix.abs().mean().mean())
        mlflow.log_metric("target_skewness", y_train.skew())
        mlflow.log_metric("target_kurtosis", y_train.kurtosis())
        mlflow.log_metric("residuals_mean", residuals_dist.mean())
        mlflow.log_metric("residuals_std", residuals_dist.std())

    except Exception as e:
        print(f"Warning: Could not complete advanced analysis: {e}")


def enhanced_model_evaluation(model, X_train, y_train, X_test, y_test, model_name, hyperparams, cv_folds):
    """Enhanced model evaluation with comprehensive metrics"""
    start_time = time.time()

    # Training predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Basic metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Cross-validation scores
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
    cv_mse_mean = -cv_scores.mean()
    cv_mse_std = cv_scores.std()

    # Inference time
    inference_start = time.time()
    _ = model.predict(X_test)
    inference_time = time.time() - inference_start
    avg_inference_time = inference_time / len(X_test)

    training_time = time.time() - start_time

    # Log all metrics
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)
    mlflow.log_metric("cv_mse_mean", cv_mse_mean)
    mlflow.log_metric("cv_mse_std", cv_mse_std)
    mlflow.log_metric("avg_inference_time", avg_inference_time)
    mlflow.log_metric("total_inference_time", inference_time)
    mlflow.log_metric("training_time", training_time)

    # Overfitting metrics
    overfitting_ratio = test_mse / train_mse
    mlflow.log_metric("overfitting_ratio", overfitting_ratio)

    # Log hyperparameters
    for param, value in hyperparams.items():
        mlflow.log_param(param, value)

    # Create visualizations
    create_learning_curve_plot(model, X_train, y_train, cv_folds, model_name)
    create_residual_plot(y_test, y_test_pred, model_name)
    create_advanced_model_analysis(
        X_train, y_train, X_test, y_test, model_name)

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Log top feature importances as metrics
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
            mlflow.log_metric(
                f"feature_importance_{row['feature']}", row['importance'])

    return {
        'model': model,
        'model_name': model_name,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'cv_mse_mean': cv_mse_mean,
        'hyperparams': hyperparams
    }


# ----- Load and prepare data -----
print("Loading and preparing data...")
# Try different possible paths for the data file
data_paths = [
    "data/california_housing.csv",  # From project root
    "../data/california_housing.csv",  # From src directory
    "data\\california_housing.csv",  # Windows path from root
    "..\\data\\california_housing.csv"  # Windows path from src
]

df = None
for path in data_paths:
    try:
        df = pd.read_csv(path)
        print(f"Data loaded from: {path}")
        break
    except FileNotFoundError:
        continue

if df is None:
    print("Error: Could not find california_housing.csv in any of these locations:")
    for path in data_paths:
        print(f"   - {path}")
    print("Please ensure the data file exists in the data/ directory")
    exit(1)

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# Optional: Feature scaling (can help with some algorithms)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# Create MLflow Experiment
mlflow.set_experiment("California-Housing-Regression-Enhanced")
print(f"MLflow experiment set to: California-Housing-Regression-Enhanced")

# Store all results for comparison
all_results = []

print(f"Starting comprehensive model training with hyperparameter tuning...")
print(f"Data shape: {X.shape}")
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ----- Linear Regression Experiments -----
print("Training Linear Regression models...")

# Standard Linear Regression
with mlflow.start_run(run_name="LinearRegression-Standard") as run:
    print("  -> Training Linear Regression (Standard features)")

    # Log system metrics at start
    system_metrics = log_system_metrics()

    # Log data profile
    log_data_profile(X_train, y_train, "train")
    log_data_profile(X_test, y_test, "test")

    # Train model
    model, model_name, mse, r2, avg_inference_time = train_linear_regression(
        X_train, y_train, X_test, y_test
    )

    # Enhanced evaluation
    result = enhanced_model_evaluation(
        model, X_train, y_train, X_test, y_test,
        f"{model_name}-Standard",
        {"feature_scaling": False, "model_type": "LinearRegression"},
        args.cv_folds
    )

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        name=f"{model_name}-Standard",
        input_example=X_test.iloc[:5],
        signature=mlflow.models.infer_signature(
            X_test.iloc[:5], model.predict(X_test.iloc[:5]))
    )

    mlflow.set_tag("Training Info", "Linear Regression with standard features")
    mlflow.set_tag("Model Type", "LinearRegression")
    mlflow.set_tag("Feature Scaling", "No")
    mlflow.set_tag("Framework", "scikit-learn")
    mlflow.set_tag("Dataset", "california_housing.csv")

    all_results.append(result)
    print(f"    -> MSE: {result['test_mse']:.4f}, R2: {result['test_r2']:.4f}")

# Scaled Linear Regression
with mlflow.start_run(run_name="LinearRegression-Scaled") as run:
    print("  -> Training Linear Regression (Scaled features)")

    # Log system metrics
    system_metrics = log_system_metrics()

    # Train model with scaled features
    model, model_name, mse, r2, avg_inference_time = train_linear_regression(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    # Enhanced evaluation
    result = enhanced_model_evaluation(
        model, X_train_scaled, y_train, X_test_scaled, y_test,
        f"{model_name}-Scaled",
        {"feature_scaling": True, "model_type": "LinearRegression"},
        args.cv_folds
    )

    # Log scaler as artifact
    scaler_path = "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(scaler_path)
    os.remove(scaler_path)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        name=f"{model_name}-Scaled",
        input_example=X_test_scaled.iloc[:5],
        signature=mlflow.models.infer_signature(
            X_test_scaled.iloc[:5], model.predict(X_test_scaled.iloc[:5]))
    )

    mlflow.set_tag("Training Info", "Linear Regression with scaled features")
    mlflow.set_tag("Model Type", "LinearRegression")
    mlflow.set_tag("Feature Scaling", "Yes")
    mlflow.set_tag("Framework", "scikit-learn")
    mlflow.set_tag("Dataset", "california_housing.csv")

    all_results.append(result)
    print(f"    -> MSE: {result['test_mse']:.4f}, R2: {result['test_r2']:.4f}")

# ----- Decision Tree Regression with Hyperparameter Tuning -----
print(f"Training Decision Tree models with hyperparameter tuning...")
print(f"  -> Testing max_depth values: {args.max_depth_list}")

for max_depth in args.max_depth_list:
    with mlflow.start_run(run_name=f"DecisionTreeRegressor-MaxDepth-{max_depth}") as run:
        print(f"  -> Training Decision Tree (max_depth={max_depth})")

        # Log system metrics
        system_metrics = log_system_metrics()

        # Train model
        model, model_name, mse, r2, avg_inference_time = run_decision_tree_regression(
            X_train, y_train, X_test, y_test, max_depth=max_depth
        )

        # Enhanced evaluation
        result = enhanced_model_evaluation(
            model, X_train, y_train, X_test, y_test,
            f"{model_name}-MaxDepth-{max_depth}",
            {"max_depth": max_depth, "model_type": "DecisionTreeRegressor"},
            args.cv_folds
        )

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            name=f"{model_name}-MaxDepth-{max_depth}",
            input_example=X_test.iloc[:5],
            signature=mlflow.models.infer_signature(
                X_test.iloc[:5], model.predict(X_test.iloc[:5]))
        )

        mlflow.set_tag("Training Info", f"Decision Tree Regressor with max_depth={max_depth}")
        mlflow.set_tag("Model Type", "DecisionTreeRegressor")
        mlflow.set_tag("Max Depth", max_depth)
        mlflow.set_tag("Framework", "scikit-learn")
        mlflow.set_tag("Dataset", "california_housing.csv")

        all_results.append(result)
        print(
            f"    -> MSE: {result['test_mse']:.4f}, R2: {result['test_r2']:.4f}")

# ----- Select and Register Best Model -----
print(f"Selecting best model from {len(all_results)} experiments...")

# Find best model by lowest CV MSE (more robust than test MSE)
best_result = min(all_results, key=lambda x: x['cv_mse_mean'])
best_model = best_result['model']
best_model_name = best_result['model_name']
best_mse = best_result['test_mse']
best_r2 = best_result['test_r2']
best_hyperparams = best_result['hyperparams']

print(f"Best model: {best_model_name}")
print(f"   Test MSE: {best_mse:.4f}")
print(f"   Test R2: {best_r2:.4f}")
print(
    f"   CV MSE: {best_result['cv_mse_mean']:.4f} +/- {best_result.get('cv_mse_std', 0):.4f}")
print(f"   Hyperparameters: {best_hyperparams}")

# Save best model locally
# Get the project root directory (parent of src folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
models_dir = os.path.join(project_root, "models")
model_path = os.path.join(models_dir, "best_model.pkl")

os.makedirs(models_dir, exist_ok=True)
joblib.dump(best_model, model_path)
print(f"Best model saved to {model_path}")

# ----- Register best model in MLflow Model Registry -----
with mlflow.start_run(run_name=f"{best_model_name}-Best-Register-Enhanced") as run:
    print(f"Registering best model in MLflow Model Registry...")

    # Log system metrics
    system_metrics = log_system_metrics()

    # Re-evaluate best model with full metrics
    if "Scaled" in best_model_name:
        X_eval, y_eval = X_test_scaled, y_test
        input_example = X_test_scaled.iloc[:5]
    else:
        X_eval, y_eval = X_test, y_test
        input_example = X_test.iloc[:5]

    # Log comprehensive metrics for the best model
    y_pred = best_model.predict(X_eval)

    # Performance comparison with all models
    for i, result in enumerate(all_results):
        mlflow.log_metric(f"comparison_model_{i}_mse", result['test_mse'])
        mlflow.log_metric(f"comparison_model_{i}_r2", result['test_r2'])

    # Log best model metrics
    for param, value in best_hyperparams.items():
        mlflow.log_param(f"best_{param}", value)

    mlflow.log_metric("best_test_mse", best_mse)
    mlflow.log_metric("best_test_r2", best_r2)
    mlflow.log_metric("best_cv_mse", best_result['cv_mse_mean'])
    mlflow.log_metric("total_experiments", len(all_results))

    # Set comprehensive tags
    mlflow.set_tag(
        "Training Info", "Best performing model selected from comprehensive hyperparameter tuning")
    mlflow.set_tag("Model Type", best_hyperparams.get('model_type', 'Unknown'))
    mlflow.set_tag("Selection Criteria", "Lowest Cross-Validation MSE")
    mlflow.set_tag("Total Experiments", len(all_results))
    mlflow.set_tag("Framework", "scikit-learn")
    mlflow.set_tag("Dataset", "california_housing.csv")
    mlflow.set_tag("Feature Scaling", str(
        best_hyperparams.get('feature_scaling', False)))

    for param, value in best_hyperparams.items():
        mlflow.set_tag(f"Best {param.title()}", str(value))

    # Register model with comprehensive signature
    mlflow.sklearn.log_model(
        sk_model=best_model,
        name=f"Best-{best_model_name}",
        registered_model_name="BestHousingModel",
        input_example=input_example,
        signature=mlflow.models.infer_signature(
            input_example,
            best_model.predict(input_example)
        ),
    )

    print(f"Model registered successfully in MLflow Model Registry!")

# ----- Summary Report -----
print("=" * 80)
print("TRAINING COMPLETE - COMPREHENSIVE SUMMARY")
print("=" * 80)
print(f"Total experiments conducted: {len(all_results)}")
print(f"Best model: {best_model_name}")
print(f"Performance metrics:")
print(f"   • Test MSE: {best_mse:.6f}")
print(f"   • Test R2: {best_r2:.6f}")
print(
    f"   • CV MSE: {best_result['cv_mse_mean']:.6f} +/- {best_result.get('cv_mse_std', 0):.6f}")
print(f"Best hyperparameters: {best_hyperparams}")
print(f"Model saved locally: {model_path}")
print(f"Model registered in MLflow: BestHousingModel")
print(f"Check MLflow UI for detailed analysis and visualizations")
print("=" * 80)

print(
    "Training complete! "
    "Check MLflow UI on http://localhost:5000 "
    "to explore comprehensive model metrics, system metrics, and visualizations."
)
