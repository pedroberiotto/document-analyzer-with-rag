FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Source needed before install (setuptools discovers the app/ and eval/ packages).
COPY pyproject.toml README.md ./
COPY app ./app
COPY eval ./eval
COPY config ./config
COPY streamlit_app.py ./

RUN pip install --upgrade pip && pip install .

EXPOSE 8000 8501

# Default to the API; docker-compose overrides the command for the UI service.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
