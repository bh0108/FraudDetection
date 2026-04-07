from src.rag.vectorstore_policy import load_policy_store
from src.rag.retrieval import retrieve_top_k

store = load_policy_store()
results = retrieve_top_k(store, "high risk transaction")

print("Retrieved:", len(results))
for r in results:
    print("----")
    print(r[:300])