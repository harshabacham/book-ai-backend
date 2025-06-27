# main.py
import os
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from book_ai_backend.ai_engine import AIEngine 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()
ai_engine = AIEngine()

@app.get("/")
async def root():
    return {"message": "Book AI Backend Running"}
# CORS setup for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for AI Engine
engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize with minimal memory footprint"""
    global engine
    try:
        logger.info("Initializing lightweight AI Engine...")
        engine = AIEngine()  # This will now use TF-IDF
        logger.info("AI Engine ready (TF-IDF mode)")
    except Exception as e:
        logger.error(f"Failed to initialize AI Engine: {e}")
        raise


class Query(BaseModel):
    question: str
    subject: str
    exam: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    message: str
    available_subjects: Optional[int] = None


@app.get("/", response_model=HealthResponse)
async def root():
    subjects_count = len(engine.list_available_subjects()) if engine else 0
    return {
        "status": "healthy",
        "message": "Book AI Backend is running!",
        "available_subjects": subjects_count,
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if engine is None:
        raise HTTPException(status_code=503, detail="AI Engine not initialized")

    subjects_count = len(engine.list_available_subjects())
    return {
        "status": "healthy",
        "message": "Service is up and running",
        "available_subjects": subjects_count,
    }


@app.get("/subjects")
async def get_subjects():
    """Get list of available subjects"""
    if engine is None:
        raise HTTPException(status_code=503, detail="AI Engine not initialized")

    subjects = engine.list_available_subjects()
    return {"subjects": subjects, "count": len(subjects)}


@app.post("/ask")
async def ask_question(query: Query):
    if engine is None:
        raise HTTPException(status_code=503, detail="AI Engine not initialized")

    try:
        # Validate input
        if not query.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        if not query.subject.strip():
            raise HTTPException(status_code=400, detail="Subject cannot be empty")

        # Log the query (optional, for debugging)
        logger.info(f"Processing question for subject: {query.subject}")

        # Get answer from AI engine
        answer = engine.get_answer(
            question=query.question, exam=query.exam, subject=query.subject
        )

        return {
            "answer": answer,
            "question": query.question,
            "subject": query.subject,
            "exam": query.exam,
            "status": "success",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Environment-specific settings
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
