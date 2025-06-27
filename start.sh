#!/bin/bash
uvicorn book_ai_backend.main:app --host 0.0.0.0 --port $PORT