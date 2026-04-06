# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

# Stage 1: install Python dependencies in an isolated directory.
FROM python:3.9-slim AS builder

WORKDIR /install

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    true

# Stage 2: runtime image
FROM python:3.9-slim AS runtime

# Prevent Python from writing .pyc files and buffer stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy the pre-installed packages from the builder stage.
COPY --from=builder /install /usr/local

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

# Copy only the source code needed at runtime.
# data/ is provided via a Docker volume.
COPY --chown=appuser:appuser capitolwatch/ ./capitolwatch/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser pytest.ini .

# Switch to the non-root user before starting the process.
USER appuser

EXPOSE 8501

# Streamlit configuration passed as CLI flags:
#   --server.address=0.0.0.0  : listen on all interfaces (required in Docker)
#   --server.headless=true    : disable browser auto-open (no GUI in container)
CMD ["python", "-m", "streamlit", "run", "capitolwatch/web/app.py", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--server.port=8501"]
