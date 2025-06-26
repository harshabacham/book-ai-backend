# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from ai_engine import AIEngine

app = FastAPI(title="Book AI Backend", version="1.0.0")

# CORS setup for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Engine
print("Initializing AI Engine...")
engine = AIEngine()
print("AI Engine initialized successfully!")

class Query(BaseModel):
    question: str
    subject: str
    exam: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/", response_model=HealthResponse)
async def root():
    return {"status": "healthy", "message": "Book AI Backend is running!"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "healthy", "message": "Service is up and running"}

@app.get("/subjects")
async def get_subjects():
    """Get list of available subjects"""
    subjects = engine.list_available_subjects()
    return {"subjects": subjects}

@app.post("/ask")
async def ask_question(query: Query):
    try:
        if not query.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if not query.subject.strip():
            raise HTTPException(status_code=400, detail="Subject cannot be empty")
        
        answer = engine.get_answer(
            question=query.question, 
            exam=query.exam, 
            subject=query.subject
        )
        
        return {
            "answer": answer,
            "question": query.question,
            "subject": query.subject,
            "exam": query.exam
        }
    except Exception as e:
        print(f"Error in ask_question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)