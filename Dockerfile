# =============================================================================
# Imagen de la API de predicción (multi-stage para que quede liviana)
#
#   docker build -t informalidad-api .
#   docker run -p 8000:8000 informalidad-api
# =============================================================================

# ---------- Etapa 1: builder (instala dependencias con uv) ----------
FROM python:3.11-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN pip install --no-cache-dir uv

WORKDIR /app

# 1) Solo los archivos de dependencias: si el código cambia pero las dependencias no,
#    Docker reutiliza esta capa (builds mucho más rápidos).
COPY pyproject.toml uv.lock README.md ./
# --no-default-groups: solo dependencias de SERVING (sin MLflow, Prefect, Optuna, pytest...)
RUN uv sync --frozen --no-default-groups --no-install-project

# 2) Ahora el código del paquete, y se instala el proyecto (no editable)
COPY src/ ./src/
RUN uv sync --frozen --no-default-groups --no-editable


# ---------- Etapa 2: runtime (solo lo necesario para correr) ----------
FROM python:3.11-slim AS runtime

# Usuario sin privilegios: buena práctica de seguridad (no correr como root)
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser models/champion/ ./models/champion/
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs

ENV PATH="/app/.venv/bin:$PATH" \
    PROJECT_ROOT=/app \
    PYTHONUNBUFFERED=1 \
    LOG_PREDICTIONS=true

USER appuser
EXPOSE 8000

# Docker marca el contenedor como "unhealthy" si /health deja de responder
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "informalidad.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
