from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# -----------------------------
# Load your model + agent + RAG
# -----------------------------
# IMPORTANT: These imports must match your project structure.
# If any of these fail, the debug prints will show it.

print(">>> LOADED v2_api.py (Version 2 API)")

try:
    from feature_builder import build_feature_vector
    print(">>> Imported feature_builder successfully")
except Exception as e:
    print(">>> ERROR importing feature_builder:", repr(e))

try:
    from model_loader import model  # or wherever your model is loaded
    print(">>> Loaded ML model successfully")
except Exception as e:
    print(">>> ERROR loading ML model:", repr(e))

try:
    from agents.agent import agentic_decision
    print(">>> Imported agentic_decision successfully")
except Exception as e:
    print(">>> ERROR importing agentic_decision:", repr(e))


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Fraud Detection API - Version 2")


# -----------------------------
# Request Schema
# -----------------------------
class Transaction(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    merchant: str
    location: str
    country: str
    channel: str
    hour: int
    timestamp: str


# -----------------------------
# V2 Endpoint (Debug Version)
# -----------------------------
@app.post("/v2/score_transaction")
def score_transaction_v2(tx: Transaction):
    try:
        print("\n\n==============================")
        print(">>> Received TX:", tx.model_dump())
        print("==============================")

        # -----------------------------
        # 1. Build feature vector
        # -----------------------------
        try:
            X = build_feature_vector({
                "customer_id": tx.customer_id,
                "amount": tx.amount,
                "country": tx.country,
                "channel": tx.channel,
                "hour": tx.hour
            })
            print(">>> Feature vector built:", X)
        except Exception as e:
            print(">>> ERROR in build_feature_vector:", repr(e))
            raise

        # -----------------------------
        # 2. ML Model Prediction
        # -----------------------------
        try:
            fraud_score = float(model.predict_proba(X)[0, 1])
            print(">>> Fraud score:", fraud_score)
        except Exception as e:
            print(">>> ERROR in model.predict_proba:", repr(e))
            raise

        # -----------------------------
        # 3. Agentic Reasoning (RAG + LLM)
        # -----------------------------
        try:
            print(">>> Calling agentic_decision...")
            agent_response = agentic_decision(tx.model_dump(), fraud_score)
            print(">>> Agent response:", agent_response)
        except Exception as e:
            print(">>> ERROR in agentic_decision:", repr(e))
            raise

        # -----------------------------
        # 4. Return Final Response
        # -----------------------------
        return {
            "fraud_score": fraud_score,
            "agent_analysis": agent_response
        }

    except Exception as e:
        print(">>> ERROR in /v2/score_transaction:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))