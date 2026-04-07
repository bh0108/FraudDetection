from src.rag.vectorstore_policy import load_policy_store
from src.rag.vectorstore_cases import load_case_store
from src.rag.retrieval import retrieve_top_k

policy_store = load_policy_store()
case_store = load_case_store()

def get_policy_context(query: str):
    return retrieve_top_k(policy_store, query)

def get_case_context(query: str):
    return retrieve_top_k(case_store, query)