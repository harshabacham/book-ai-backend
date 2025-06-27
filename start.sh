#!/bin/bash
# Add src/ to Python path
export PYTHONPATH="${PYTHONPATH}:./src"
# Start FastAPI from the correct location
uvicorn src.book_ai_backend.main:app --host 0.0.0.0 --port ${PORT:-10000}cd src && uvicorn book_ai_backend.main:app --host 0.0.0.0 --port $PORT