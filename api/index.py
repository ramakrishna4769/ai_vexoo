from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add parent directory to path to import the ingestion code from the existing module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local Python module
from part1_ingestion import DocumentIngestionSystem

app = FastAPI(title="Knowledge Pyramid API")

# Setup CORS for Vercel edge
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngestRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    query: str

# Instantiate a global mock instance for serverless runtime
sys_instance = DocumentIngestionSystem(page_char_limit=200)

@app.post("/api/ingest")
def ingest_text(req: IngestRequest):
    global sys_instance
    # Reinitialize to clear state on each new API ingestion
    sys_instance = DocumentIngestionSystem(page_char_limit=200)
    sys_instance.ingest_document(req.text)
    
    nodes = []
    for node in sys_instance.pyramid_nodes:
        nodes.append({
            "raw_text": node.raw_text,
            "summary": node.summary,
            "category": node.category,
            "distilled": node.distilled
        })
    return {"message": "Success", "nodes": nodes}

@app.post("/api/query")
def query_text(req: QueryRequest):
    if not sys_instance.pyramid_nodes:
        return {"error": "No document ingested yet. Please ingest a document first."}
    
    result = sys_instance.retrieve(req.query)
    return {"result": result}
