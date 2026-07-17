FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-docling.txt ./
RUN pip install --no-cache-dir -r requirements-docling.txt
COPY requirements-entity-resolution.txt ./
RUN pip install --no-cache-dir -r requirements-entity-resolution.txt
COPY knowledge_forge ./knowledge_forge
COPY server.py ./
RUN mkdir -p /app/KnowledgeFiles/docs /app/KnowledgeFiles/chunks /app/KnowledgeFiles/media /app/KnowledgeFiles/.trash

EXPOSE 8010 8011
CMD ["python", "server.py"]
