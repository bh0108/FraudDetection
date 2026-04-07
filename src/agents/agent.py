from .tools import get_policy_context, get_case_context
from .reasoning import llm_reason

def agentic_decision(transaction, fraud_score):
    query = f"""
    Transaction: {transaction}
    Fraud Score: {fraud_score}

    Retrieve relevant policies and cases.
    """

    policies = get_policy_context(query)
    cases = get_case_context(query)

    prompt = f"""
    You are a fraud analyst AI.

    Transaction: {transaction}
    Fraud Score: {fraud_score}

    Relevant Policies:
    {policies}

    Relevant Cases:
    {cases}

    Provide:
    - Final decision (fraud or not)
    - Explanation
    - Policy justification
    - Case justification
    """

    return llm_reason(prompt)