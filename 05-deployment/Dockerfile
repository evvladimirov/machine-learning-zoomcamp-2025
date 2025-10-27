FROM agrigorev/zoomcamp-model:2025

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
# COPY . /app

# Install the application dependencies.
# WORKDIR /app
# RUN touch __init__.py
COPY main.py .
# COPY pipeline_v1.bin .
COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --frozen --no-cache

# Run the application.
CMD ["/code/.venv/bin/fastapi", "run", "main.py", "--port", "80", "--host", "0.0.0.0"]
