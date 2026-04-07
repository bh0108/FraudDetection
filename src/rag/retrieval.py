def retrieve_top_k(store, query, k=3):
    results = store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]