import os
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from ai_engine import AIEngine

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("book_ai")

app = FastAPI(
    title="Book AI API",
    description="Backend service for textbook question answering",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine = None

class Query(BaseModel):
    question: str
    subject: str
    exam: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    subjects_loaded: int
    uptime: float

@app.on_event("startup")
async def startup_event():
    global engine
    try:
        data_path = Path(os.getenv("DATA_PATH", "data"))
        logger.info(f"Initializing AI Engine with data from: {data_path}")
        
        start_time = time.time()
        engine = AIEngine(data_path)
        load_time = time.time() - start_time
        
        logger.info(f"AI Engine initialized in {load_time:.2f}s")
        logger.info(f"Loaded subjects: {engine.list_available_subjects()}")
    except Exception as e:
        logger.critical(f"Startup failed: {e}")
        raise

@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse(
        content={
            "service": "Book AI Backend",
            "status": "running",
            "docs": "/docs"
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if not engine:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    return {
        "status": "healthy",
        "message": "Service operational",
        "subjects_loaded": len(engine.list_available_subjects()),
        "uptime": time.time() - startup_time
    }

@app.get("/subjects")
async def list_subjects():
    if not engine:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    return {
        "subjects": engine.list_available_subjects(),
        "count": len(engine.list_available_subjects())
    }

@app.post("/ask")
async def ask_question(query: Query):
    if not engine:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    try:
        start_time = time.time()
        answer = engine.get_answer(
            question=query.question,
            subject=query.subject,
            exam=query.exam
        )
        response_time = time.time() - start_time
        
        return {
            "answer": answer,
            "metadata": {
                "response_time": f"{response_time:.2f}s",
                "subject": query.subject
            }
        }
    except Exception as e:
        logger.error(f"Question processing failed: {e}")
        raise HTTPException(status_code=500, detail="Question processing error")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info")
    )