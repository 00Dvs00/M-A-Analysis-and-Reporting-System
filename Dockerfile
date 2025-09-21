# Use Python 3.11 slim base image for smaller size
FROM python:3.11-slim as builder

# Set build environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=0

# Create and set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start a new stage for the final image
FROM python:3.11-slim

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create application user
RUN useradd -m -U appuser && \
    mkdir -p /app /app/cache /app/config && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Create cache directory for API responses
RUN mkdir -p /app/cache/api

# Expose port if needed (e.g., for API or web interface)
# EXPOSE 8000

# Default command
ENTRYPOINT ["python"]
CMD ["main.py", "--config", "config/default_config.yaml"]