#!/bin/bash
# Add src/ to Python path
export PYTHONPATH="${PYTHONPATH}:./src"
# Start FastAPI from the correct location
#!/bin/bash
uvicorn main:app --host 0.0.0.0 --port 10000