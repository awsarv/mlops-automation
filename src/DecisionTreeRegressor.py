from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import time


def run_decision_tree_regression(X_train, y_train, X_test, y_test, max_depth):
    # You can tune this for more runs
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    start_time = time.time()
    _ = model.predict(X_test)
    elapsed_time = time.time() - start_time
    avg_inference_time = elapsed_time / len(X_test)

    return model, "DecisionTreeRegressor", mse, r2, avg_inference_time
