import json
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

# Import agent functions
from agents.router import route_query
from agents.vision import analyze_medical_image
from agents.rag import retrieve_relevant_documents
from agents.chat import handle_conversational_chat
from agents.synthesizer import synthesize_final_response
from ingest import ingest_file

app = FastAPI(title="AI Medical Assistant API", version="1.0")

# Setup CORS middleware
# Enables React frontend running on localhost:5173 or similar to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp directory for document ingestion
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/chat")
async def chat(
    query: str = Form(...),
    history: str = Form(default=None),
    file: UploadFile = File(default=None)
):
    """
    Main chat endpoint. Receives query, conversation history (optional JSON string),
    and an optional image file. Executes multi-agent routing, individual agent processing,
    and returns a synthesized final answer with safety checks, citations, and confidence scores.
    """
    try:
        # 1. Parse history JSON if provided
        parsed_history = []
        if history:
            try:
                parsed_history = json.loads(history)
            except Exception:
                print("Failed to parse history JSON, defaulting to empty list.")
                
        # 2. Determine if an image was uploaded
        has_image = file is not None
        pil_image = None
        
        if has_image:
            # Verify file is an image
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
            # Read image into memory
            image_data = await file.read()
            pil_image = Image.open(io.BytesIO(image_data))
            
        # 3. Router Agent decides path
        route = route_query(query, has_image=has_image)
        
        # 4. Invoke the selected Agent
        agent_output = {}
        if route == "VISION":
            agent_output = analyze_medical_image(pil_image, query)
        elif route == "RAG":
            agent_output = retrieve_relevant_documents(query)
        else: # CHAT
            agent_output = handle_conversational_chat(query, parsed_history)
            
        # 5. Synthesize final response
        synthesis_result = synthesize_final_response(route, query, agent_output)
        synthesis_result["route"] = route
        
        return synthesis_result
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingests reference medical documents (PDFs) into the Qdrant database.
    It parses them with Docling, embeds the chunks using Gemini embeddings, and upserts them.
    """
    # Verify file is a PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported for ingestion.")
        
    # Save the PDF file temporarily
    temp_path = UPLOAD_DIR / file.filename
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run ingestion script function
        ingest_file(str(temp_path))
        
        return {
            "success": True,
            "message": f"Successfully ingested '{file.filename}' into vector database."
        }
    except Exception as e:
        print(f"Error during ingestion endpoint execution: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        # Clean up temp file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

# Serve static files for frontend production build if available
frontend_static = Path("frontend/dist")
if frontend_static.exists():
    app.mount("/", StaticFiles(directory=str(frontend_static), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
