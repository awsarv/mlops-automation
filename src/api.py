from fastapi import FastAPI, Response
from pydantic import BaseModel
import os
import json
import logging
import sqlite3
import numpy as np
import joblib
import pandas as pd  # only used by /retrain
from sklearn.linear_model import LinearRegression
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST


# ----- Paths/Env -----
LOG_PATH = os.getenv("LOG_PATH", "/app/logs/api.log")
DB_PATH = os.getenv("DB_PATH", "/app/logs/api_requests.db")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/best_model.pkl")
FEATURE_ORDER_PATH = os.getenv(
    "FEATURE_ORDER_PATH", "/app/models/feature_order.json"
)

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# ----- Logging -----
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ----- SQLite -----
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("PRAGMA journal_mode=WAL;")
c.execute("PRAGMA synchronous=NORMAL;")
c.execute(
    """
CREATE TABLE IF NOT EXISTS requests(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    MedInc REAL,
    HouseAge REAL,
    AveRooms REAL,
    AveBedrms REAL,
    Population REAL,
    AveOccup REAL,
    Latitude REAL,
    Longitude REAL,
    Prediction REAL,
    Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""
)
conn.commit()


# ----- Metrics -----
LATENCY = Histogram(
    "inference_latency_seconds", "Prediction latency in seconds"
)
PREDICTIONS = Counter(
    "predictions_total", "Total prediction requests", ["status"]
)


# ----- App/Schema -----
app = FastAPI(title="Housing Inference API")


class Features(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


# ----- Model & feature order -----
if os.path.exists(FEATURE_ORDER_PATH):
    with open(FEATURE_ORDER_PATH) as f:
        FEATURE_ORDER = json.load(f)
else:
    FEATURE_ORDER = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(features: Features):
    with LATENCY.time():
        try:
            row = [getattr(features, col) for col in FEATURE_ORDER]
            data = np.array([row], dtype=float)
            prediction = float(model.predict(data)[0])

            logging.info(
                "Request=%s Prediction=%s",
                features.model_dump(),
                prediction,
            )

            c.execute(
                (
                    "INSERT INTO requests ("
                    "MedInc, HouseAge, AveRooms, AveBedrms, Population, "
                    "AveOccup, Latitude, Longitude, Prediction"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                tuple(row) + (prediction,),
            )
            conn.commit()

            PREDICTIONS.labels("ok").inc()
            return {"prediction": prediction}
        except Exception:
            logging.exception("Prediction failed")
            PREDICTIONS.labels("error").inc()
            raise


@app.post("/retrain")
def retrain():
    # Retrain in-process for demo; for real use, make this an async job
    df = pd.read_csv("data/california_housing.csv")
    x_mat = df.drop("MedHouseVal", axis=1)
    y_vec = df["MedHouseVal"]

    retrained_model = LinearRegression()
    retrained_model.fit(x_mat, y_vec)

    joblib.dump(retrained_model, MODEL_PATH)
    with open(FEATURE_ORDER_PATH, "w") as f:
        json.dump(list(x_mat.columns), f)

    global model, FEATURE_ORDER
    model = retrained_model
    FEATURE_ORDER = list(x_mat.columns)

    logging.info("Retraining completed; model reloaded.")
    return {"status": "retrained"}
