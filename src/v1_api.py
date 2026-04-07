# v1_api.py  (Python 3.12 compatible)

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
import sqlite3
from datetime import datetime

app = FastAPI(title="Fraud Detection API - V1")

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "fraud_model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.joblib"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

class Transaction(BaseModel):
    customer_id: int
    amount: float
    country: str
    channel: str
    hour: int

def build_feature_vector(tx: Transaction) -> pd.DataFrame:
    df = pd.DataFrame([tx.dict()])
    df = pd.get_dummies(df, columns=["country", "channel"], drop_first=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]

@app.post("/score_transaction")
def score_transaction(tx: Transaction):
    X = build_feature_vector(tx)
    proba = float(model.predict_proba(X)[0, 1])
    is_fraud = proba >= 0.5

    explanation = "Low risk transaction."
    if is_fraud:
        explanation = "High risk transaction based on amount, country, channel, and time pattern."

    conn = sqlite3.connect("database/alerts.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO alerts (transaction_id, fraud_probability, is_fraud, explanation, created_at) VALUES (?, ?, ?, ?, ?)",
        (None, proba, int(is_fraud), explanation, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

    return {
        "fraud_probability": proba,
        "is_fraud": is_fraud,
        "explanation": explanation,
    }