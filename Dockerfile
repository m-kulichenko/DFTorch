FROM python:3.11-slim

# Python is unbuffered
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv into the image ( reproducibly install deps)
RUN python -m pip install --no-cache-dir uv

# Copy dependency metadata first to maximize Docker layer caching.
# If only the source changes, Docker can reuse the dependency install layer.
COPY pyproject.toml uv.lock README.md ./

# Copy the source + tests
COPY src ./src
COPY tests ./tests

# Install dependencies (dev for pytest/ruff)
RUN uv sync --extra dev

# run tests
CMD ["uv", "run", "pytest"]