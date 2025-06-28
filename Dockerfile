# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy everything to /app
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1000"]
