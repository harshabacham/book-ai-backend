#!/bin/bash
# Add src/ to Python path
#!/bin/bash
cd /opt/render/project/src
python -m book_ai_backend.main

export PYTHONPATH="${PYTHONPATH}:./src"
# Start FastAPI from the correct location
#!/bin/bash
cd src/book_ai_backend
uvicorn main:app --host 0.0.0.0 --port 10000