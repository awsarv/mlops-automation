# Use a minimal Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only requirements first for efficient Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ ./src/
COPY models/ ./models/

# Expose port 8000 for Uvicorn
EXPOSE 8000

# Start the API with Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
