FROM python:3.11-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    UV_SYSTEM_PYTHON=1

# Copy project files for dependency installation
COPY pyproject.toml ./

# Generate requirements.txt from pyproject.toml and install dependencies
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install -r requirements.txt uvicorn

# Copy application code
COPY ai_exercise ./ai_exercise
COPY evals.py ./
COPY evals ./evals

# Expose port
EXPOSE 80

# Start FastAPI with hot reload
CMD ["uvicorn", "ai_exercise.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl --fail http://localhost/health || exit 1
