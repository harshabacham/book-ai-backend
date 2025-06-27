# Use official Python image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy files
COPY . /app

# Upgrade pip and install deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
