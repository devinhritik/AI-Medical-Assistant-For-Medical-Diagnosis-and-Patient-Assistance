from config import get_qdrant_client, gemini_client, QDRANT_COLLECTION

def get_query_embedding(query: str) -> list[float]:
    """
    Generates embedding vector for the search query using gemini-embedding-2.
    """
    if not gemini_client:
        raise ValueError("Gemini client is not initialized.")
    response = gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=query
    )
    return response.embeddings[0].values

def retrieve_relevant_documents(query: str, top_k: int = 4) -> dict:
    """
    Finds the most semantically similar chunks in the Qdrant database.
    Returns:
      - 'retrieved_chunks': List of matching text snippets with metadata.
      - 'agent': Name of the agent ('RAG Agent').
    """
    print(f"[RAG Agent] Retrieving relevant documents for: '{query}'")
    client = get_qdrant_client()
    
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(col.name == QDRANT_COLLECTION for col in collections)
        
        if not exists:
            print(f"[RAG Agent] Collection '{QDRANT_COLLECTION}' does not exist yet. No documents ingested.")
            return {
                "retrieved_chunks": [],
                "agent": "RAG Agent"
            }
            
        # 1. Generate query embedding
        query_vector = get_query_embedding(query)
        
        # 2. Query Qdrant
        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k
        ).points
        
        retrieved_chunks = []
        for res in results:
            retrieved_chunks.append({
                "text": res.payload.get("text", ""),
                "source_file": res.payload.get("source_file", "Unknown Document"),
                "chunk_index": res.payload.get("chunk_index", 0),
                "score": res.score
            })
            
        print(f"[RAG Agent] Retrieved {len(retrieved_chunks)} chunks from Qdrant.")
        return {
            "retrieved_chunks": retrieved_chunks,
            "agent": "RAG Agent"
        }
    except Exception as e:
        print(f"[RAG Agent] Error during document retrieval: {e}")
        return {
            "retrieved_chunks": [],
            "agent": "RAG Agent"
        }
