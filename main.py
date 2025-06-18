from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ai_engine import AIEngine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_engine = AIEngine(data_folder="data")

@app.get("/")
def read_root():
    return {"message": "✅ AI Exam PDF Backend is Running!"}

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    answer = ai_engine.get_answer(question)
    return {"answer": answer}
