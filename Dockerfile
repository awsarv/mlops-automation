# Use a minimal Python image
FROM python:3.10-slim

# Install SQLite CLI (handy for debugging in container; optional)
RUN apt-get update && apt-get install -y --no-install-recommends sqlite3 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ ./src/
COPY models/ ./models/

# Ensure runtime dirs exist
RUN mkdir -p /app/logs /app/data

# Expose port for Uvicorn
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
