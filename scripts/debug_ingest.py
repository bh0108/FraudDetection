import os
from pathlib import Path
from src.rag.loaders import load_documents
from src.rag.splitters import get_splitter

print("=== DEBUG: Folder Check ===")
print("Current working directory:", os.getcwd())

policy_path = Path("documents/policies")
case_path = Path("documents/cases")

print("\nPolicies folder exists:", policy_path.exists())
print("Cases folder exists:", case_path.exists())

print("\nFiles in policies folder:", os.listdir(policy_path) if policy_path.exists() else "NOT FOUND")
print("Files in cases folder:", os.listdir(case_path) if case_path.exists() else "NOT FOUND")

print("\n=== DEBUG: Loader Check ===")
docs = load_documents("documents/policies")
print("Documents loaded:", len(docs))

splitter = get_splitter()
chunks = splitter.split_documents(docs)
print("Chunks created:", len(chunks))