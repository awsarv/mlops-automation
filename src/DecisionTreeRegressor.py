from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import time
import numpy as np


def run_decision_tree_regression(X_train, y_train, X_test, y_test,
                                 max_depth=5, random_state=42):
    # Create model with additional parameters for better performance
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=random_state,
        min_samples_split=2,
        min_samples_leaf=1
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    preds = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Measure inference time more accurately
    inference_times = []
    for _ in range(10):  # Multiple runs for better average
        start_time = time.time()
        _ = model.predict(X_test)
        elapsed_time = time.time() - start_time
        inference_times.append(elapsed_time)

    avg_inference_time = np.mean(inference_times) / len(X_test)

    return model, "DecisionTreeRegressor", mse, r2, avg_inference_time
