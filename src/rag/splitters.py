from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )