# Multi-stage: the builder installs the package (editable install, so the
# source stays the code that runs), the runtime image only carries the
# installed dependencies plus the source and datasets it needs.
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml ./
COPY evalkit ./evalkit/
RUN pip install --no-cache-dir -e .

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY pyproject.toml ./
COPY evalkit ./evalkit/
COPY datasets ./datasets/
COPY scripts ./scripts/
# Render injects PORT; 8000 is only the local default (see evalkit/config.py).
ENV PORT=8000
EXPOSE 8000
# $PORT must stay unexpanded here: Render sets it at run time, and a
# hard-coded 8000 would make the service answer 502 behind its router.
CMD uvicorn evalkit.api:app --host 0.0.0.0 --port $PORT
