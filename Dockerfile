FROM python:3.11-slim

# Build-time variables (passed in by CI)
ARG VCS_REF="unknown"
ARG BUILD_DATE="unknown"
ARG VERSION="dev"

# OCI standard-ish labels (metadata baked into the image)
LABEL org.opencontainers.image.source="https://github.com/m-kulichenko/DFTorch" \
      org.opencontainers.image.revision="$VCS_REF" \
      org.opencontainers.image.created="$BUILD_DATE" \
      org.opencontainers.image.version="$VERSION"

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