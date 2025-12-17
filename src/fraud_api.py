"""
Fraud Detection API Service.

This module implements a real-time credit card fraud detection REST API
using FastAPI. It provides endpoints for fraud prediction, health monitoring,
and Prometheus metrics collection.

Architecture:
    - FastAPI for REST API framework
    - SQLite for request logging and audit trail
    - Prometheus client for metrics exposition
    - Joblib for model serialization

Security Features:
    - Input validation via Pydantic models
    - Request logging for audit compliance
    - Structured error handling

Author: Arvind Kumar
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
import os
import json
import logging
import sqlite3
import numpy as np
import joblib
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST


# Configuration
LOG_PATH = os.getenv("LOG_PATH", "/app/logs/api.log")
DB_PATH = os.getenv("DB_PATH", "/app/logs/fraud_requests.db")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/best_model.pkl")
FEATURE_ORDER_PATH = os.getenv("FEATURE_ORDER_PATH", "/app/models/feature_order.json")
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# Logging Configuration
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# Database Setup for Audit Trail
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("PRAGMA journal_mode=WAL;")
cursor.execute("PRAGMA synchronous=NORMAL;")
cursor.execute("""
CREATE TABLE IF NOT EXISTS fraud_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    amount REAL,
    hour INTEGER,
    day_of_week INTEGER,
    merchant_category INTEGER,
    distance_from_home REAL,
    distance_from_last_transaction REAL,
    ratio_to_median_purchase REAL,
    repeat_retailer INTEGER,
    used_chip INTEGER,
    used_pin INTEGER,
    online_order INTEGER,
    fraud_probability REAL,
    is_flagged INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()


# Prometheus Metrics
LATENCY = Histogram(
    "fraud_inference_latency_seconds",
    "Fraud prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Total fraud prediction requests",
    ["status", "result"]
)
FRAUD_FLAGGED = Counter(
    "fraud_transactions_flagged_total",
    "Total transactions flagged as fraudulent"
)
AMOUNT_HISTOGRAM = Histogram(
    "transaction_amount_dollars",
    "Transaction amounts in dollars",
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
)
MODEL_CONFIDENCE = Histogram(
    "fraud_model_confidence",
    "Model confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)
ACTIVE_MODEL = Gauge(
    "fraud_model_loaded",
    "Indicates if the fraud model is loaded (1=yes, 0=no)"
)


# FastAPI Application
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection service",
    version="1.0.0"
)


# Request/Response Models
class Transaction(BaseModel):
    """Credit card transaction features for fraud detection."""
    amount: float = Field(..., ge=0, description="Transaction amount in dollars")
    hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    merchant_category: int = Field(..., ge=0, le=14, description="Merchant category code")
    distance_from_home: float = Field(..., ge=0, description="Distance from home in miles")
    distance_from_last_transaction: float = Field(..., ge=0, description="Distance from last transaction")
    ratio_to_median_purchase: float = Field(..., ge=0, description="Ratio to median purchase amount")
    repeat_retailer: int = Field(..., ge=0, le=1, description="Is repeat retailer (0/1)")
    used_chip: int = Field(..., ge=0, le=1, description="Used chip (0/1)")
    used_pin: int = Field(..., ge=0, le=1, description="Used PIN (0/1)")
    online_order: int = Field(..., ge=0, le=1, description="Is online order (0/1)")

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 125.50,
                "hour": 14,
                "day_of_week": 2,
                "merchant_category": 5,
                "distance_from_home": 15.3,
                "distance_from_last_transaction": 5.2,
                "ratio_to_median_purchase": 1.2,
                "repeat_retailer": 1,
                "used_chip": 1,
                "used_pin": 1,
                "online_order": 0
            }
        }


class FraudPrediction(BaseModel):
    """Fraud detection prediction response."""
    transaction_id: str
    fraud_probability: float
    is_fraudulent: bool
    risk_level: str
    recommendation: str
    timestamp: str


# Model Loading
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_ORDER_PATH) as f:
        FEATURE_ORDER = json.load(f)
    ACTIVE_MODEL.set(1)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    FEATURE_ORDER = [
        'amount', 'hour', 'day_of_week', 'merchant_category',
        'distance_from_home', 'distance_from_last_transaction',
        'ratio_to_median_purchase', 'repeat_retailer',
        'used_chip', 'used_pin', 'online_order'
    ]
    ACTIVE_MODEL.set(0)


def get_risk_level(probability: float) -> tuple:
    """Determine risk level and recommendation based on fraud probability."""
    if probability >= 0.9:
        return "CRITICAL", "BLOCK transaction immediately. Contact customer."
    elif probability >= 0.7:
        return "HIGH", "BLOCK transaction. Require additional verification."
    elif probability >= 0.5:
        return "MEDIUM", "FLAG for review. Consider SMS verification."
    elif probability >= 0.3:
        return "LOW", "ALLOW with monitoring. Log for analysis."
    else:
        return "MINIMAL", "ALLOW transaction. Normal activity."


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=FraudPrediction)
def predict_fraud(transaction: Transaction):
    """
    Predict if a credit card transaction is fraudulent.

    Returns fraud probability, risk level, and recommendation.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with LATENCY.time():
        try:
            # Prepare features
            features = [getattr(transaction, col) for col in FEATURE_ORDER]
            X = np.array([features], dtype=float)

            # Predict
            fraud_probability = float(model.predict_proba(X)[0, 1])
            is_fraudulent = fraud_probability >= FRAUD_THRESHOLD
            risk_level, recommendation = get_risk_level(fraud_probability)

            # Generate transaction ID
            transaction_id = f"TXN-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

            # Log metrics
            AMOUNT_HISTOGRAM.observe(transaction.amount)
            MODEL_CONFIDENCE.observe(fraud_probability)

            if is_fraudulent:
                FRAUD_FLAGGED.inc()
                PREDICTIONS.labels(status="ok", result="fraud").inc()
                logger.warning(
                    f"FRAUD DETECTED: {transaction_id} amount=${transaction.amount:.2f} "
                    f"prob={fraud_probability:.4f}"
                )
            else:
                PREDICTIONS.labels(status="ok", result="legitimate").inc()
                logger.info(
                    f"Transaction {transaction_id}: amount=${transaction.amount:.2f} "
                    f"prob={fraud_probability:.4f} risk={risk_level}"
                )

            # Store in database
            cursor.execute("""
                INSERT INTO fraud_predictions (
                    amount, hour, day_of_week, merchant_category,
                    distance_from_home, distance_from_last_transaction,
                    ratio_to_median_purchase, repeat_retailer,
                    used_chip, used_pin, online_order,
                    fraud_probability, is_flagged
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (*features, fraud_probability, int(is_fraudulent)))
            conn.commit()

            return FraudPrediction(
                transaction_id=transaction_id,
                fraud_probability=round(fraud_probability, 4),
                is_fraudulent=is_fraudulent,
                risk_level=risk_level,
                recommendation=recommendation,
                timestamp=datetime.utcnow().isoformat()
            )

        except Exception as e:
            PREDICTIONS.labels(status="error", result="none").inc()
            logger.exception(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": type(model).__name__,
        "features": FEATURE_ORDER,
        "n_features": len(FEATURE_ORDER),
        "fraud_threshold": FRAUD_THRESHOLD,
        "model_path": MODEL_PATH
    }


@app.get("/stats")
def get_stats():
    """Get fraud detection statistics."""
    cursor.execute("""
        SELECT
            COUNT(*) as total_predictions,
            SUM(is_flagged) as total_flagged,
            AVG(fraud_probability) as avg_probability,
            AVG(amount) as avg_amount,
            MAX(amount) as max_amount
        FROM fraud_predictions
        WHERE timestamp > datetime('now', '-24 hours')
    """)
    row = cursor.fetchone()

    return {
        "period": "last_24_hours",
        "total_predictions": row[0] or 0,
        "total_flagged": row[1] or 0,
        "flag_rate": (row[1] or 0) / max(row[0] or 1, 1),
        "avg_fraud_probability": round(row[2] or 0, 4),
        "avg_transaction_amount": round(row[3] or 0, 2),
        "max_transaction_amount": round(row[4] or 0, 2)
    }
