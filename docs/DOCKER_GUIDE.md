# Docker Guide - DSPy Self-Fortifying LLM Gateway

This guide covers how to use Docker Compose to run the DSPy-powered LLM Security Gateway.

## Quick Start

### 1. Setup Environment Variables

Copy the example environment file and add your API keys:

```bash
cp env.example .env
# Edit .env with your API keys
```

### 2. Basic Usage

Run the main DSPy-powered security gateway:

```bash
# Start the main gateway service
docker-compose up gateway

# Or run in detached mode
docker-compose up -d gateway
```

### 3. Available Services

The docker-compose setup includes several services:

- **gateway**: Main DSPy-powered LLM security gateway
- **validator**: DSPy validation and testing service
- **mlflow**: MLflow tracking server for optimization results
- **ollama**: Local LLM server for cost-effective testing

## Service Profiles

### Default Profile (gateway only)
```bash
docker-compose up gateway
```

### Test Profile (includes validator)
```bash
docker-compose --profile test up
```

### Full Profile (all services)
```bash
docker-compose --profile full up
```

## Common Commands

### Run DSPy Demo
```bash
# Interactive demo with real LLMs
docker-compose run --rm gateway python demo_real_llm.py

# Run with specific provider
docker-compose run --rm gateway python demo_real_llm.py --provider openai
```

### Run DSPy Validation Tests
```bash
# Run all DSPy validation tests
docker-compose --profile test up validator

# Or run specific tests
docker-compose run --rm validator python -m pytest tests/test_dspy_integration.py -v
```

### Run Legacy Audit Tool
```bash
# Run the original audit tool
docker-compose run --rm gateway python audit_prompt.py examples
```

### Access MLflow UI
```bash
# Start MLflow service
docker-compose --profile full up mlflow

# Access at http://localhost:5000
```

### Use Local Ollama
```bash
# Start Ollama service
docker-compose --profile full up ollama

# Use in demo
docker-compose run --rm gateway python demo_real_llm.py --provider ollama
```

## Development Workflow

### 1. Development with Live Code Changes

Mount your local code for development:

```bash
# Create override file for development
cat > docker-compose.override.yml << EOF
services:
  gateway:
    volumes:
      - .:/app
    command: ["python", "demo_real_llm.py", "--provider", "openai"]
EOF

# Run with live code changes
docker-compose up gateway
```

### 2. Running Tests

```bash
# Run all tests
docker-compose --profile test run --rm validator

# Run specific test files
docker-compose run --rm validator python -m pytest tests/test_dspy_evaluation_framework.py -v

# Run with coverage
docker-compose run --rm validator python -m pytest tests/ --cov=. --cov-report=html
```

### 3. Interactive Shell

```bash
# Get shell access to container
docker-compose run --rm gateway bash

# Or run Python interactively
docker-compose run --rm gateway python -i
```

## Configuration

### Environment Variables

Set these in your `.env` file:

```bash
# Required: LLM API Keys
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Optional: DSPy Configuration
DSPY_LOG_LEVEL=INFO
DSPY_CACHE_DIR=/app/.dspy_cache

# Optional: MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
```

### Volume Mounts

The following directories are mounted as volumes:

- `./configs` → `/app/configs` - Configuration files
- `./prompts` → `/app/prompts` - Prompt templates
- `./data` → `/app/data` - Input data files
- `./results` → `/app/results` - Output results
- `./tests` → `/app/tests` - Test files
- `dspy-cache` → `/app/.dspy_cache` - DSPy cache (persistent)

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```bash
   # Check environment variables
   docker-compose run --rm gateway env | grep API_KEY
   ```

2. **DSPy Import Errors**
   ```bash
   # Verify DSPy installation
   docker-compose run --rm gateway python -c "import dspy; print(dspy.__version__)"
   ```

3. **Permission Issues**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER results/ data/
   ```

4. **Container Won't Start**
   ```bash
   # Check logs
   docker-compose logs gateway
   
   # Rebuild container
   docker-compose build --no-cache gateway
   ```

### Health Checks

All services include health checks:

```bash
# Check service health
docker-compose ps

# View health check logs
docker inspect dspy-llm-security-gateway --format='{{.State.Health.Status}}'
```

## Performance Optimization

### 1. Build Optimization

```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker-compose build

# Build with specific target
docker-compose build --target builder
```

### 2. Cache Management

```bash
# Clear DSPy cache
docker-compose run --rm gateway rm -rf .dspy_cache/*

# Clear Docker build cache
docker builder prune
```

### 3. Resource Limits

Add resource limits in `docker-compose.override.yml`:

```yaml
services:
  gateway:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

## Production Deployment

### 1. Security Considerations

```bash
# Use secrets for API keys
echo "your-api-key" | docker secret create openai_api_key -

# Run with read-only filesystem
docker-compose run --rm --read-only gateway
```

### 2. Monitoring

```bash
# Monitor resource usage
docker stats dspy-llm-security-gateway

# Export metrics
docker-compose --profile full up mlflow
```

### 3. Backup

```bash
# Backup volumes
docker run --rm -v dspy-cache:/data -v $(pwd):/backup alpine tar czf /backup/dspy-cache.tar.gz -C /data .
```

## Advanced Usage

### Custom Commands

```bash
# Run optimization with specific parameters
docker-compose run --rm gateway python -c "
from defender import DefenderProgram
defender = DefenderProgram()
defender.tune(num_trials=10)
"

# Run batch processing
docker-compose run --rm gateway python scripts/batch_audit.py
```

### Multi-Stage Builds

The Dockerfile uses multi-stage builds for optimization:

- **builder**: Installs dependencies
- **runtime**: Minimal runtime environment

### Network Configuration

Services communicate via the `gateway-network`:

```bash
# Inspect network
docker network inspect dspy-self-fortifying-llm-gateway_gateway-network
```

## Support

For issues with Docker setup:

1. Check the logs: `docker-compose logs`
2. Verify environment: `docker-compose config`
3. Test connectivity: `docker-compose run --rm gateway ping mlflow`
4. Review health checks: `docker-compose ps`

For DSPy-specific issues, see the main README.md and test files. 