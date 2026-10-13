# Guía de despliegue

## 1. Local, sin Docker (desarrollo)

```bash
uv sync
uv run uvicorn informalidad.api.app:app --reload --port 8000
```

Abrir `http://localhost:8000` (formulario) o `http://localhost:8000/docs` (Swagger).

## 2. Docker (recomendado)

Requisitos: Docker Desktop corriendo y el modelo exportado en `models/champion/` (ya viene en el
repositorio; si reentrenas, el pipeline lo actualiza solo).

```bash
docker compose up --build        # construye la imagen y levanta la API en el puerto 8000
docker compose ps                # STATUS debe decir "healthy" después de ~20 s
docker compose logs -f api       # ver logs
docker compose down              # detener
```

Sin compose:

```bash
docker build -t informalidad-api .
docker run -d -p 8000:8000 --name informalidad-api informalidad-api
```

Características de la imagen:

- **Multi-stage**: la etapa `builder` instala dependencias con `uv`; la etapa final solo copia el
  entorno virtual, la configuración y el modelo.
- Solo dependencias de serving (sin MLflow, Prefect ni Optuna) y sin NCCL → ~350 MB.
- Corre con un **usuario sin privilegios** (`appuser`), no como root.
- **HEALTHCHECK** contra `/health`.
- `logs/` se monta como volumen para que el log de predicciones quede en tu máquina.
  En Linux, si aparece *Permission denied* al escribir el log: `chmod 777 logs`.

## 3. Actualizar el modelo en producción

```bash
uv run python pipelines/training_flow.py      # entrena; si mejora, exporta el nuevo champion
docker compose up --build -d                  # reconstruye la imagen con el modelo nuevo
curl http://localhost:8000/model/info         # confirmar la versión en servicio
```

Para volver a una versión anterior: en la UI de MLflow, mover el alias `champion` a esa versión,
luego `uv run python scripts/export_champion.py` y reconstruir la imagen.

## 4. CI/CD (GitHub Actions)

En cada push a `main` o pull request, `.github/workflows/ci.yml`:

1. Instala el entorno exacto con `uv sync --frozen`.
2. Corre `ruff check` y `ruff format --check`.
3. Corre las pruebas (unitarias + integración con MLflow temporal).
4. Construye la imagen Docker, la levanta y prueba `/health` y `/predict` (smoke test).

## 5. Nube (opcional, siguiente paso)

La misma imagen sirve en cualquier servicio de contenedores. La ruta más corta, con lo visto en el
curso (`04-Deployment/deploy/web-service-aws/GUIA_AWS_EC2.md`):

1. Crear una instancia EC2 (Amazon Linux, t3.small) y abrir el puerto 8000 en el Security Group.
2. Instalar Docker y git en la instancia, clonar este repositorio.
3. `docker compose up --build -d`.
4. Probar con `http://<IP-publica>:8000/docs`.
5. Al terminar, **detener la instancia** para no generar costos.

Alternativas administradas: AWS App Runner, Google Cloud Run o Azure Container Apps (se publica la
imagen en un registro y el servicio la ejecuta con HTTPS y autoescalado).
