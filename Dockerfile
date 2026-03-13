# AIIA — AI Information Architecture
# Local Brain runtime container
#
# Build:  docker build -t aiia .
# Run:    docker run -p 8100:8100 -p 8200:8200 --env-file .env aiia

FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY local_brain/ ./local_brain/

# Create data directories
RUN mkdir -p /data/eq_data /data/logs

# Environment defaults
ENV EQ_BRAIN_DATA_DIR=/data/eq_data
ENV LOCAL_BRAIN_HOST=0.0.0.0
ENV LOCAL_BRAIN_PORT=8100
ENV COMMAND_CENTER_PORT=8200
ENV PYTHONUNBUFFERED=1

# Expose Local Brain API + Command Center
EXPOSE 8100 8200

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

# Start Local Brain API
CMD ["python", "-m", "local_brain.local_api"]
