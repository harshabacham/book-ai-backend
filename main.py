from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_engine import AIEngine

app = FastAPI()

# Enable CORS (access from mobile/Flutter app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI engine
engine = AIEngine(data_folder="data")

# Input model
class Query(BaseModel):
    question: str
    subject: str

# Ask endpoint
@app.post("/ask")
async def ask_question(query: Query):
    answer = engine.get_answer(query.question, query.subject)
    return {"answer": answer}
