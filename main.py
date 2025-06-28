from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from ai_engine import AIEngine
import os
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    subject: str
    exam: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    subjects_loaded: int

@app.on_event("startup")
async def startup_event():
    try:
        app.state.engine = AIEngine()
        logger.info("AI Engine initialized successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.get("/", response_model=HealthResponse)
async def health_check():
    engine = app.state.engine
    return {
        "status": "healthy",
        "message": "Book AI Backend is running",
        "subjects_loaded": len(engine.vectorizers)
    }

@app.post("/ask")
async def ask_question(query: Query):
    engine = app.state.engine
    try:
        answer = engine.get_answer(
            question=query.question,
            subject=query.subject,
            exam=query.exam
        )
        return {
            "question": query.question,
            "subject": query.subject,
            "answer": answer,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        raise HTTPException(status_code=500, detail=str(e))