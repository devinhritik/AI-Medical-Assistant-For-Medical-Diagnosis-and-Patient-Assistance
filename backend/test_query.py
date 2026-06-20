from config import get_qdrant_client, QDRANT_COLLECTION
from agents.rag import get_query_embedding

def test():
    client = get_qdrant_client()
    try:
        # Check collections
        cols = client.get_collections().collections
        print("Available collections:")
        for c in cols:
            print(f"- {c.name}")
            
        # Get count
        info = client.get_collection(QDRANT_COLLECTION)
        print(f"\nCollection {QDRANT_COLLECTION} status:")
        print(f"Points count: {info.points_count}")
        print(f"Vector configuration: {info.config.params.vectors}")
        
        # Test Query
        query = "What is the classification of brain tumors according to the WHO clinical guidelines?"
        query_vector = get_query_embedding(query)
        print(f"\nQuery embedding size: {len(query_vector)}")
        
        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=4
        ).points
        
        print(f"\nSearch results (top 4):")
        for idx, res in enumerate(results):
            print(f"Match {idx+1}: Score={res.score}, File={res.payload.get('source_file')}, Chunk={res.payload.get('chunk_index')}")
            print(f"Text snippet: {res.payload.get('text')[:100]}...\n")
            
    except Exception as e:
        print(f"Error testing query: {e}")

if __name__ == "__main__":
    test()
