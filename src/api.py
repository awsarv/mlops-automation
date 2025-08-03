from fastapi import FastAPI, Response, Body
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import sqlite3
import os
from prometheus_client import Counter, generate_latest

os.makedirs("/app/logs", exist_ok=True)

# ----- Load paths from environment -----
log_path = os.getenv("LOG_PATH", "/app/logs/api.log")
db_path = os.getenv("DB_PATH", "/app/logs/api_requests.db")

# ----- Model Loading -----
model = joblib.load("models/best_model.pkl")

# ----- API and Schema Setup -----
app = FastAPI()


class Features(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


# ----- File Logging Setup -----
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ----- SQLite Logging Setup -----
conn = sqlite3.connect(db_path, check_same_thread=False)
c = conn.cursor()
c.execute(
    (
        "CREATE TABLE IF NOT EXISTS requests ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "MedInc REAL,"
        "HouseAge REAL,"
        "AveRooms REAL,"
        "AveBedrms REAL,"
        "Population REAL,"
        "AveOccup REAL,"
        "Latitude REAL,"
        "Longitude REAL,"
        "Prediction REAL,"
        "Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"
    )
)
conn.commit()

# ----- Prometheus Metric Setup -----
PREDICTIONS = Counter(
    "predictions_total",
    "Total prediction requests served"
)


@app.post(
    "/predict",
    response_model=dict,
    responses={
        200: {
            "description": "Successful Prediction",
            "content": {
                "application/json": {
                    "example": {"prediction": 4.778788739495791}
                }
            },
        }
    },
)
def predict(features: Features = Body(...)):
    """
    Receives JSON with California Housing features,
    returns predicted median house value.
    Logs request to file and SQLite, updates Prometheus counter.
    """
    data = np.array(
        [[
            features.MedInc,
            features.HouseAge,
            features.AveRooms,
            features.AveBedrms,
            features.Population,
            features.AveOccup,
            features.Latitude,
            features.Longitude
        ]]
    )
    prediction = float(model.predict(data)[0])

    # File logging
    logging.info(
        "Request: %s | Prediction: %s",
        features.dict(),
        prediction
    )

    # SQLite logging
    c.execute(
        (
            "INSERT INTO requests ("
            "MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,"
            "Latitude, Longitude, Prediction"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        (
            features.MedInc,
            features.HouseAge,
            features.AveRooms,
            features.AveBedrms,
            features.Population,
            features.AveOccup,
            features.Latitude,
            features.Longitude,
            prediction
        )
    )
    conn.commit()

    # Prometheus metric
    PREDICTIONS.inc()

    return {"prediction": prediction}


@app.get("/metrics")
def metrics():
    """
    Exposes Prometheus metrics for monitoring.
    """
    return Response(generate_latest(), media_type="text/plain")


@app.post("/retrain")
def retrain():
    """
    Demo endpoint for model retraining trigger.
    """
    logging.info("Retraining triggered!")
    return {"status": "Retraining triggered (not implemented fully in demo)"}
