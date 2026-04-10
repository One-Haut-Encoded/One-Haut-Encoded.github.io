# Hugging Face Space / Docker: API only (backend FastAPI).
# Connect Space alexoh2020/onehaut_backend to this repo; SDK: Docker.
FROM python:3.11-slim
WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY scripts ./scripts

ENV PYTHONPATH=/app

EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
