from src.rag.loaders import load_documents
from src.rag.splitters import get_splitter
from src.rag.vectorstore_policy import create_policy_store

docs = load_documents("documents/policies")
splitter = get_splitter()
chunks = splitter.split_documents(docs)

create_policy_store(chunks)
print("Policy vectorstore created.")