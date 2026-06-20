# AI Medical Assistant For Medical Diagnosis and Patient Assistance

AI Medical Assistant is an end-to-end, LLM-powered Multi-Agent system for medical imaging research and patient assistance. It routes user inputs dynamically to custom-built sub-agents for multimodal scan analysis, Retrieval-Augmented Generation (RAG) over clinical reference documents, or general medical dialogue.

##  Architecture

```
              User
                  │
                  ▼
          Frontend (React)
                  │
                  ▼
        Backend (FastAPI)
                  │
                  ▼
        ┌─────────────────────────┐
        │      Router Agent        │
        └───────────┬──────────────┘
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
Gemini Vision   RAG Retrieval   Chat Agent
(Image Scan)  (Docling + Qdrant) (Triage/Care)
     │              │              │
     └──────────────┴──────────────┘
                    ▼
        Response Synthesizer
                    │
      + citations + confidence + safety checks
                    │
                    ▼
                 Final Answer
```

## 🛠️ Tech Stack
*   **Frontend:** ReactJS, Vite, Tailwind CSS, Lucide icons, Markdown-to-HTML rendering
*   **Backend:** FastAPI, Uvicorn, Python 3.11
*   **Orchestration & LLM:** Google Gemini API (`gemini-1.5-flash`), Google GenAI SDK
*   **Document Parsing:** Docling (IBM layout-aware PDF parser)
*   **Vector Database:** Qdrant DB running via Docker
*   **CI/CD Pipeline:** GitHub Actions

---

##  Step-by-Step Installation & Run Guide

### Step 1: Start Qdrant Vector Database (Docker)
Ensure Docker Desktop is running on your machine, then spin up the Qdrant container:
```bash
docker run -d --name qdrant-medical -p 6333:6333 -p 6334:6334 qdrant/qdrant
```
You can view the Qdrant web interface directly at `http://localhost:6333/dashboard`.

### Step 2: Configure Backend Environment
1. Open the project folder in your terminal and navigate to `backend/`.
2. Ensure your `.env` contains your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   QDRANT_COLLECTION=medical_knowledge
   ```

### Step 3: Run the Backend Server
Navigate to the `backend/` directory, activate the Python virtual environment, and start the FastAPI server:
```bash
# Navigate
cd backend

# Activate Virtual Environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Run the FastAPI server
python main.py
```
The backend API is now running at `http://localhost:8000`. You can inspect the Swagger interactive documentation at `http://localhost:8000/docs`.

### Step 4: Run the Frontend App
Navigate to the `frontend/` directory, install packages, and start the Vite development server:
```bash
# Navigate
cd ../frontend

# Install dependencies
npm install

# Start Vite React server
npm run dev
```
Open `http://localhost:5173` in your web browser to access the interactive MedAssist dashboard!

---

##  How to Ingest PDF Guidelines (RAG)
1. Using the **Reference Literature Ingest** panel on the left side of the dashboard, upload any clinical guidelines PDF (e.g. Brain Tumor Guidelines, Chest X-ray Manuals).
2. The UI will show a progress indicator while **Docling** parses the PDF layout, extracts text tables, embeds them via Gemini, and upserts them to the Qdrant vector database.
3. Once ingested, ask the assistant technical or factual questions. The **Router** will send the query to the **RAG Agent**, which queries Qdrant and returns cited references.

---

