FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-heb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY src ./src
COPY text ./text
COPY text_type.csv ./text_type.csv

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "gunicorn --chdir src --workers 1 --timeout 240 --graceful-timeout 30 --bind 0.0.0.0:${PORT} web_app:app"]
