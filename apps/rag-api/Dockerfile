FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# + tesseract (system OCR engine)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pinned deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY src ./src
ENV PYTHONPATH=/app/src

# Set DATA_DIR for Docker environment
ENV DATA_DIR=/app/data

# writable data dir + non-root user
RUN mkdir -p /app/data && useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]