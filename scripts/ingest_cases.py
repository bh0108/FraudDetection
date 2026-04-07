from src.rag.loaders import load_documents
from src.rag.splitters import get_splitter
from src.rag.vectorstore_cases import create_case_store

docs = load_documents("documents/cases")
splitter = get_splitter()
chunks = splitter.split_documents(docs)

create_case_store(chunks)
print("Case vectorstore created.")