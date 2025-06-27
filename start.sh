#!/bin/bash
# Set Python path to include your source directory
export PYTHONPATH="${PYTHONPATH}:/opt/render/project/src"

# Default to port 8000 if $PORT not set
UVICORN_PORT=${PORT:-8000}

# Start the application with auto-reload for development
uvicorn book_ai_backend.main:app \
    --host 0.0.0.0 \
    --port $UVICORN_PORT \
    --reload