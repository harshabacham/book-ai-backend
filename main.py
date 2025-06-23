from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_engine import AIEngine

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    subject: str  # 👈 Accept subject

@app.post("/ask")
async def ask_question(query: Query):
    engine = AIEngine(data_folder=f"data/{query.subject.lower()}")
    answer = engine.get_answer(query.question)
    return {"answer": answer}
