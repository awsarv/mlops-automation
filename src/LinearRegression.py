import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def run_linear_regression(X_train, y_train, X_test, y_test):
    # Create and train model
    model = LinearRegression()
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

    return model, "LinearRegression", mse, r2, avg_inference_time
