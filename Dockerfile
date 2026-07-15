# --- build stage: install dependencies ---
ARG PYTHON_IMAGE=public.ecr.aws/docker/library/python:3.11.14-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.11

FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS builder

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
COPY pyproject.toml uv.lock ./
RUN uv venv --python 3.11 && \
    uv sync --locked --no-dev --extra server --no-install-project
COPY model_library/ model_library/
COPY model_gateway/ model_gateway/
RUN uv sync --locked --no-dev --extra server --no-editable

# --- runtime stage ---
FROM ${PYTHON_IMAGE}

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY model_library/ model_library/
COPY model_gateway/ model_gateway/

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn model_gateway.main:create_app --factory --host 0.0.0.0 --port 8000 --workers ${GATEWAY_UVICORN_WORKERS:-2} --loop ${GATEWAY_UVICORN_LOOP:-auto} --http ${GATEWAY_UVICORN_HTTP_PROTOCOL:-auto} --timeout-keep-alive ${GATEWAY_UVICORN_KEEPALIVE_TIMEOUT_SECONDS:-75} --timeout-graceful-shutdown ${GATEWAY_UVICORN_GRACEFUL_SHUTDOWN_SECONDS:-120}"]
