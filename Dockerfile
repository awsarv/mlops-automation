# DevSecOps ML Pipeline - Secure Dockerfile
# Following security best practices

# Use specific version, not latest
FROM python:3.10-slim-bookworm AS builder

# Security: Don't run as root during build
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim-bookworm AS production

# Security labels
LABEL maintainer="Arvind Kumar <2023ac05606@wilp.bits-pilani.ac.in>"
LABEL description="DevSecOps ML Pipeline - Housing Price Prediction API"
LABEL version="1.0"
LABEL security.scan="enabled"

# Security: Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# Security: Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

# Create necessary directories
RUN mkdir -p /app/logs /app/data \
    && chown -R appuser:appuser /app

# Security: Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/best_model.pkl
ENV LOG_PATH=/app/logs/api.log
ENV DB_PATH=/app/logs/api_requests.db
ENV FEATURE_ORDER_PATH=/app/models/feature_order.json

# Security: Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
