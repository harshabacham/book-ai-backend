from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from ai_engine import AIEngine
import os
import json
import logging
from pathlib import Path

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

# ------------------- Models -------------------
class Query(BaseModel):
    question: str
    subject: str
    exam: Optional[str] = None

class QuizRequest(BaseModel):
    exam: str
    subject: str
    level: int

class HealthResponse(BaseModel):
    status: str
    message: str
    subjects_loaded: int

# ------------------- Startup -------------------
@app.on_event("startup")
async def startup_event():
    try:
        app.state.engine = AIEngine()
        logger.info("AI Engine initialized successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

# ------------------- Routes -------------------

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
@app.post("/get_questions")
async def get_questions(request: QuizRequest):
    subject = request.subject.lower()
    level = str(request.level)

    file_path = Path(f"exams/{subject}.json")  # 🔁 No exam name needed here

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Subject file '{file_path}' not found.")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if subject not in data:
            raise HTTPException(status_code=404, detail=f"Subject '{subject}' not found in file.")

        if level not in data[subject]:
            raise HTTPException(status_code=404, detail=f"Level {level} not found in subject '{subject}'.")

        return {
            "status": "success",
            "exam": request.exam,
            "subject": subject,
            "level": level,
            "questions": data[subject][level]
        }

    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        raise HTTPException(status_code=500, detail="Failed to load questions.")
