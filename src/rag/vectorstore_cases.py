from langchain_community.vectorstores import Chroma
from .embeddings import get_embeddings

def create_case_store(docs, persist_dir="vectorstores/case_index"):
    embeddings = get_embeddings()
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    store.persist()
    return store

def load_case_store(persist_dir="vectorstores/case_index"):
    embeddings = get_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )