# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install necessary OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy all files and folders from current context to /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    tensorflow-cpu \
    pillow \
    python-multipart \
    keras

# Expose the port where FastAPI runs
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
