#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
uvicorn book_ai_backend.main:app --host 0.0.0.0 --port 10000
