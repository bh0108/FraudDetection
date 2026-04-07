from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def load_documents(folder: str):
    docs = []
    for file in Path(folder).glob("*.pdf"):
        loader = PyPDFLoader(str(file))
        docs.extend(loader.load())
    return docs