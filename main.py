import os
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import time
from ai_engine import AIEngine  # Make sure this import matches your file structure

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("book_ai")

# Initialize FastAPI with metadata
app = FastAPI(
    title="Book AI API",
    description="Backend service for textbook question answering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# Global variable for AI Engine with health status
class EngineStatus:
    instance = None
    last_init_time = 0
    is_healthy = False

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"]
)

# Request models
class Query(BaseModel):
    question: str
    subject: str
    exam: Optional[str] = None
    context: Optional[str] = None  # Additional context for the question

class HealthResponse(BaseModel):
    status: str
    message: str
    available_subjects: int
    uptime: float
    load_time: Optional[float]
    memory_usage: Optional[float]

# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.on_event("startup")
async def startup_event():
    """Initialize AI Engine with retry logic"""
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Initializing AI Engine (Attempt {attempt + 1}/{max_retries})")
            
            # Use absolute path for data directory
            data_path = Path(__file__).parent / "data"
            logger.info(f"Looking for data in: {data_path}")
            
            start_time = time.time()
            EngineStatus.instance = AIEngine(data_path)
            load_time = time.time() - start_time
            
            EngineStatus.is_healthy = True
            EngineStatus.last_init_time = time.time()
            EngineStatus.load_time = load_time
            
            logger.info(f"AI Engine initialized successfully in {load_time:.2f}s")
            logger.info(f"Loaded subjects: {EngineStatus.instance.list_available_subjects()}")
            return
            
        except Exception as e:
            logger.error(f"Initialization failed (Attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.critical("Failed to initialize AI Engine after multiple attempts")
                EngineStatus.is_healthy = False
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
    """Comprehensive health check endpoint"""
    if not EngineStatus.is_healthy:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    subjects = []
    subject_count = 0
    if EngineStatus.instance:
        subjects = EngineStatus.instance.list_available_subjects()
        subject_count = len(subjects)
    
    return {
        "status": "healthy",
        "message": "Service operational",
        "available_subjects": subject_count,
        "uptime": time.time() - EngineStatus.last_init_time,
        "load_time": getattr(EngineStatus, 'load_time', None),
        "memory_usage": os.getpid().memory_info().rss / (1024 * 1024)  # MB
    }

@app.get("/subjects")
async def get_subjects():
    """Get available subjects with statistics"""
    if not EngineStatus.is_healthy:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    try:
        subjects = EngineStatus.instance.list_available_subjects()
        stats = EngineStatus.instance.get_subject_stats()
        
        return {
            "subjects": subjects,
            "count": len(subjects),
            "details": [
                {
                    "subject": subj,
                    "documents": stats[f"jee_{subj}"][0] if f"jee_{subj}" in stats else 0,
                    "pages": stats[f"jee_{subj}"][1] if f"jee_{subj}" in stats else 0
                }
                for subj in subjects
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching subjects: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving subject list")

@app.post("/ask")
async def ask_question(query: Query):
    """
    Submit a question and get an answer
    - question: The question text (required)
    - subject: The subject area (required)
    - exam: Optional exam type (e.g., 'jee', 'neet')
    - context: Additional context to help with answering
    """
    if not EngineStatus.is_healthy:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    # Input validation
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if not query.subject.strip():
        raise HTTPException(status_code=400, detail="Subject cannot be empty")
    
    try:
        logger.info(f"Processing question: {query.question[:100]}... (subject: {query.subject})")
        
        start_time = time.time()
        answer = EngineStatus.instance.get_answer(
            question=query.question,
            subject=query.subject,
            exam=query.exam
        )
        process_time = time.time() - start_time
        
        logger.info(f"Question processed in {process_time:.2f}s")
        
        return {
            "answer": answer,
            "metadata": {
                "subject": query.subject,
                "exam": query.exam,
                "processing_time": process_time,
                "confidence": "high" if "high confidence" in answer else "medium"
            }
        }
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question"
        )

# Production-ready server configuration
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 10000))
    workers = int(os.getenv("WEB_CONCURRENCY", 1))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_config=None,
        timeout_keep_alive=60
    )