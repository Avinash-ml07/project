from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from .rag_system import HackRxRAGSystem
import tempfile
import os

app = FastAPI()

# Initialize RAG system
rag_system = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: list
    retrieved_chunks: int

@app.on_event("startup")
async def startup_event():
    global rag_system
    rag_system = HackRxRAGSystem(pinecone_api_key=os.getenv("PINECONE_API_KEY"))

@app.post("/upload-documents/")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Upload and process documents"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    temp_paths = []
    try:
        # Save uploaded files temporarily
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
                content = await file.read()
                tmp.write(content)
                temp_paths.append(tmp.name)
        
        # Process documents
        rag_system.process_documents(temp_paths)
        
        return {"message": f"Successfully processed {len(files)} documents"}
    
    finally:
        # Clean up temporary files
        for path in temp_paths:
            if os.path.exists(path):
                os.unlink(path)

@app.post("/query/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document collection"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        response = rag_system.answer_query(request.query)
        return QueryResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
