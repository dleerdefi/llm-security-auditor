# Multi-stage Dockerfile for DSPy-Powered LLM Security Gateway
# Stage 1: Builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
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
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 gateway

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/gateway/.local

# Copy application code
COPY --chown=gateway:gateway . .

# Create necessary directories
RUN mkdir -p results prompts configs data tests .dspy_cache && \
    chown -R gateway:gateway /app

# Switch to non-root user
USER gateway

# Add local bin to PATH
ENV PATH=/home/gateway/.local/bin:$PATH

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default environment variables
ENV PYTHONUNBUFFERED=1
ENV DSPY_CACHE_DIR=/app/.dspy_cache

# Make scripts executable
RUN chmod +x demo_real_llm.py audit_prompt.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import dspy; print('DSPy Gateway OK')" || exit 1

# Default entrypoint for DSPy demo
ENTRYPOINT ["python"]

# Default command runs the DSPy demo
CMD ["demo_real_llm.py"] 