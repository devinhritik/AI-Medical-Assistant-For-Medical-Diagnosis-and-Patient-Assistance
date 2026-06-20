import os
import uuid
import argparse
from pathlib import Path
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import get_qdrant_client, gemini_client, QDRANT_COLLECTION

def init_qdrant_collection(vector_size: int = 3072):
    """
    Ensures that the Qdrant collection exists. 
    Gemini's 'gemini-embedding-2' produces vectors of size 3072.
    """
    client = get_qdrant_client()
    try:
        # Check if collection already exists
        collections = client.get_collections().collections
        exists = any(col.name == QDRANT_COLLECTION for col in collections)
        
        if exists:
            # Check vector size of existing collection
            info = client.get_collection(QDRANT_COLLECTION)
            current_size = info.config.params.vectors.size
            if current_size != vector_size:
                print(f"Existing collection vector size ({current_size}) doesn't match target ({vector_size}). Recreating...")
                client.delete_collection(QDRANT_COLLECTION)
                exists = False
                
        if not exists:
            print(f"Creating Qdrant collection: '{QDRANT_COLLECTION}' with vector size {vector_size}")
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        else:
            print(f"Qdrant collection '{QDRANT_COLLECTION}' already exists with correct vector size.")
    except Exception as e:
        print(f"Error initializing Qdrant collection: {e}")
        raise e

def parse_document_with_docling(file_path: Path) -> str:
    """
    Uses Docling (IBM's layout-aware document parser) to extract
    the PDF's contents into beautifully structured Markdown format.
    """
    print(f"Parsing document with Docling: {file_path}")
    from docling.document_converter import DocumentConverter
    
    converter = DocumentConverter()
    result = converter.convert(file_path)
    # Export parsed document to markdown representation
    markdown_text = result.document.export_to_markdown()
    return markdown_text

def chunk_markdown_text(text: str, max_chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """
    Chunks markdown text by splitting it at reasonable boundaries (e.g. paragraphs),
    ensuring each chunk is within max_chunk_size characters.
    """
    # Simple semantic paragraph splitter
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If adding paragraph keeps chunk within limits
        if len(current_chunk) + len(para) + 2 <= max_chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        else:
            # Save the old chunk if it contains text
            if current_chunk:
                chunks.append(current_chunk)
            
            # Start new chunk
            # If a single paragraph is longer than max_chunk_size, split it by characters
            if len(para) > max_chunk_size:
                sub_start = 0
                while sub_start < len(para):
                    chunks.append(para[sub_start : sub_start + max_chunk_size])
                    sub_start += max_chunk_size - overlap
                current_chunk = ""
            else:
                current_chunk = para
                
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def embed_text(text: str) -> list[float]:
    """
    Calls Google Gemini's 'gemini-embedding-2' model to generate
    a dense vector representation (3072 dimensions) of the text.
    """
    if not gemini_client:
        raise ValueError("Gemini client is not initialized.")
        
    response = gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    # The API returns a list of embeddings. Get values of the first one.
    return response.embeddings[0].values

def ingest_file(file_path: str):
    """
    Runs the full ingestion pipeline for a single file.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found at '{file_path}'")
        return
        
    # 1. Parse using Docling
    try:
        markdown_text = parse_document_with_docling(path)
    except Exception as e:
        print(f"Docling failed to parse {file_path}: {e}")
        return
        
    # 2. Chunk text
    chunks = chunk_markdown_text(markdown_text)
    print(f"Split document into {len(chunks)} text chunks.")
    
    # 3. Connect to Qdrant & ensure collection exists
    init_qdrant_collection(vector_size=3072)
    q_client = get_qdrant_client()
    
    points = []
    for idx, chunk in enumerate(chunks):
        print(f"Embedding chunk {idx + 1}/{len(chunks)}...", end="\r")
        try:
            vector = embed_text(chunk)
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk,
                "source_file": path.name,
                "chunk_index": idx
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        except Exception as e:
            print(f"\nFailed to embed chunk {idx}: {e}")
            
    # 4. Upsert vectors to Qdrant
    if points:
        print(f"\nUpserting {len(points)} points into Qdrant collection '{QDRANT_COLLECTION}'...")
        q_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print("Ingestion completed successfully!")
    else:
        print("No chunks embedded. Ingestion failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF documents into Qdrant using Docling & Gemini.")
    parser.add_argument("--file", type=str, required=True, help="Path to the PDF file to ingest")
    
    args = parser.parse_args()
    ingest_file(args.file)
