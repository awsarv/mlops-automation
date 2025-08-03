import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def run_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    start_time = time.time()
    _ = model.predict(X_test)
    elapsed_time = time.time() - start_time
    avg_inference_time = elapsed_time / len(X_test)

    return model, "LinearRegression", mse, r2, avg_inference_time
