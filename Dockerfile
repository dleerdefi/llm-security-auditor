# Multi-stage Dockerfile for LLM Security Auditor
# Stage 1: Builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 auditor

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/auditor/.local

# Copy application code
COPY --chown=auditor:auditor . .

# Create necessary directories
RUN mkdir -p results prompts configs data && \
    chown -R auditor:auditor /app

# Switch to non-root user
USER auditor

# Add local bin to PATH
ENV PATH=/home/auditor/.local/bin:$PATH

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default environment variables
ENV PYTHONUNBUFFERED=1

# Make the CLI tool executable
RUN chmod +x audit_prompt.py

# Default entrypoint
ENTRYPOINT ["python", "audit_prompt.py"]

# Default command shows examples
CMD ["examples"] 