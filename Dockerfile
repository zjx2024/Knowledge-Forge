FROM python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

ARG CONSTRAINTS_FILE=constraints.txt
ARG DOCLING_REQUIREMENTS=requirements-docling.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN apt-get -o Acquire::ForceIPv4=true update \
    && apt-get -o Acquire::ForceIPv4=true install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-docling.txt requirements-docling-gpu.txt requirements-entity-resolution.txt constraints.txt constraints-gpu.txt ./
RUN pip install --no-cache-dir -c "${CONSTRAINTS_FILE}" \
        -r requirements.txt \
        -r "${DOCLING_REQUIREMENTS}" \
        -r requirements-entity-resolution.txt \
    && pip check
COPY knowledge_forge ./knowledge_forge
COPY server.py reranker_server.py ./
RUN groupadd --system --gid 10001 knowledgeforge \
    && useradd --system --uid 10001 --gid knowledgeforge --home-dir /home/knowledgeforge --create-home knowledgeforge \
    && mkdir -p /app/KnowledgeFiles/docs /app/KnowledgeFiles/chunks /app/KnowledgeFiles/media /app/KnowledgeFiles/.trash /home/knowledgeforge/.cache \
    && chown -R knowledgeforge:knowledgeforge /app /home/knowledgeforge

EXPOSE 8010 8011 8012
USER knowledgeforge
CMD ["python", "server.py"]
