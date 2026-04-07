from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from src.v1_api import build_feature_vector, model
from src.agents.agent import agentic_decision


app = FastAPI(title="Fraud Detection API - Version 2")


class Transaction(BaseModel):
    transaction_id: str
    customer_id: int
    amount: float
    merchant: str
    location: str
    country: str
    channel: str
    hour: int
    timestamp: str


@app.post("/v2/score_transaction")
def score_transaction_v2(tx: Transaction) -> Dict[str, Any]:
    """
    Version 2 scoring endpoint:
    - Reuses V1 RandomForest model + feature pipeline
    - Adds GenAI / RAG / agentic reasoning via agentic_decision()
    """
    try:
        print("\n================ V2 REQUEST ================")
        print(">>> Incoming transaction:", tx.model_dump())

        # 1) Build feature vector using the SAME logic as V1
        X = build_feature_vector(tx)
        print(">>> Feature vector shape:", X.shape)
        print(">>> Feature vector columns:", list(X.columns))

        # 2) Get fraud probability from the RandomForest model
        fraud_score = float(model.predict_proba(X)[0, 1])
        print(">>> Fraud score:", fraud_score)

        # 3) Call the agent (RAG + LLM + policy/case reasoning)
        tx_dict = tx.model_dump()
        print(">>> Calling agentic_decision with tx + fraud_score...")
        agent_response = agentic_decision(tx_dict, fraud_score)
        print(">>> Agent response:", agent_response)

        # 4) Return combined response
        return {
            "fraud_score": fraud_score,
            "agent_analysis": agent_response,
        }

    except Exception as e:
        print(">>> ERROR in /v2/score_transaction:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))