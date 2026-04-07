from langchain_community.vectorstores import Chroma
from .embeddings import get_embeddings

def create_policy_store(docs, persist_dir="vectorstores/policy_index"):
    embeddings = get_embeddings()
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    store.persist()
    return store

def load_policy_store(persist_dir="vectorstores/policy_index"):
    embeddings = get_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )