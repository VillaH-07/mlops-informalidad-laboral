# Predicción de Informalidad Laboral en Colombia · Proyecto MLOps end-to-end

[![CI](https://github.com/VillaH-07/mlops-informalidad-laboral/actions/workflows/ci.yml/badge.svg)](https://github.com/VillaH-07/mlops-informalidad-laboral/actions/workflows/ci.yml)

Proyecto final del curso **MLOps – Universidad de Medellín**. Toma un modelo que predice si un
trabajador colombiano es informal (no cotiza a pensión), a partir de la **GEIH del DANE**, y lo lleva
a un sistema de ML de producción: tracking de experimentos, pipeline orquestado, model registry,
API en Docker, monitoreo de drift, pruebas y CI.

> Base: [proyecto anterior de informalidad laboral](https://villah-07.github.io/ju-informalidad-laboral/)
> (EDA + modelos en scripts). Este repositorio le agrega toda la capa MLOps.

---

## 1. Problema de negocio (hipotético)

Una entidad pública que diseña programas de **formalización laboral y protección a la vejez**
necesita saber qué trabajadores tienen alta probabilidad de **no cotizar a pensión**, para focalizar
campañas, subsidios o vinculación a programas como BEPS. Hoy esa focalización se hace con reglas
simples (p. ej. "sin contrato = informal"), que dejan por fuera a muchos informales.

**El modelo** recibe características observables de un trabajador y devuelve la probabilidad de que
sea informal. Se consume por API, individualmente o en lotes.

## 2. Métricas de éxito y resultados

Evaluado sobre un **test fuera de tiempo** (enero–febrero 2026, meses que el modelo nunca vio):

| Criterio | Meta | Resultado |
|---|---|---|
| ROC-AUC | ≥ 0.95 | **0.972** ✅ |
| F1 clase informal | ≥ 0.90 | **0.919** ✅ |
| Mejora sobre regresión logística (AUC) | ≥ +0.01 | **+0.020** ✅ |
| Recall informales | ≥ 0.90 | **0.904** ✅ |
| Latencia API (1 trabajador) | < 100 ms | ~12–17 ms ✅ |
| Pruebas y lint en CI | 100 % en verde | 37 pruebas ✅ |

| Modelo | CV AUC (5 folds) | Test AUC | Test F1 |
|---|---|---|---|
| Dummy (proporción) | 0.500 | 0.500 | 0.714 |
| Regla "sin contrato o cuenta propia" | – | 0.754 | 0.751 |
| Regresión logística | 0.953 | 0.952 | 0.890 |
| Random Forest | 0.967 | 0.962 | 0.899 |
| XGBoost (default) | 0.973 | 0.971 | 0.916 |
| **XGBoost + Optuna (champion)** | – | **0.972** | **0.919** |

Detalle en [`notebooks/02_baseline.ipynb`](notebooks/02_baseline.ipynb) y
[`notebooks/03_experiments.ipynb`](notebooks/03_experiments.ipynb).

## 3. Datos

- **Fuente:** Gran Encuesta Integrada de Hogares (GEIH), DANE. Módulos *Características generales* y
  *Ocupados*, 10 meses: ene–abr 2024, ene–abr 2025, ene–feb 2026.
- **Registros:** 287.900 trabajadores ocupados · **Tasa de informalidad:** 56,9 %.
- **Target:** `INFORMAL = 1` si no cotiza a pensión (`P6920 = 2`).
- **Features (10):** edad, horas/semana, ingreso laboral, sexo, nivel educativo (P3042), estado civil,
  afiliación a salud, posición ocupacional, contrato y rama de actividad (CIIU 2 dígitos).
- **Construcción:** `scripts/build_dataset.py` une los módulos *Características generales* y *Ocupados*
  de cada mes a partir de los CSV crudos del DANE, así el dataset es reproducible desde la fuente.
- **Corrección frente al proyecto anterior:** allí se usó P6083 como nivel educativo, pero en el
  diccionario del DANE P6083 es *"¿La madre de ... reside en este hogar?"*. Aquí se usa la variable
  correcta, P3042 (*mayor nivel educativo alcanzado*). El resto de columnas coincide fila a fila.
- **Versionado:** el dataset completo (17 MB, generado desde ~38 GB de microdatos) no va en git. Va una
  **muestra estratificada por mes** (`data/sample/`, 10.000 filas) para que cualquiera pueda correr el
  proyecto completo. Cada corrida registra en MLflow el **checksum** del archivo usado (`data_version`).

## 4. Arquitectura

```
GEIH ─► validar ─► split temporal ─► comparar algoritmos ─► Optuna ─► modelo final ─► MLflow Registry
                                    (CV, MLflow)         (nested runs)               candidate/champion
                                                                                           │ export
   monitoreo (PSI, AUC, equidad) ◄── logs/predictions.jsonl ◄── FastAPI en Docker ◄────────┘
            │ alerta
            └──────────► reentrenamiento (Prefect, mensual)
```

Diagrama completo y decisiones de diseño: [`docs/arquitectura.md`](docs/arquitectura.md).

## 5. Cómo ejecutarlo

### Requisitos

- [uv](https://docs.astral.sh/uv/) (instala Python 3.11 automáticamente)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (para el contenedor)
- Git

### Paso a paso (los comandos funcionan igual en PowerShell, Mac y Linux)

```bash
# 1. Clonar e instalar el entorno exacto (uv.lock)
git clone https://github.com/VillaH-07/mlops-informalidad-laboral.git
cd mlops-informalidad-laboral
uv sync

# 2. (Opcional) Dataset completo desde los CSV crudos de la GEIH (una carpeta por mes)
uv run python scripts/build_dataset.py --raw-dir "ruta/a/GEIH/raw"
#    Si no está, todo corre automáticamente con la muestra de data/sample/.

# 3. Pruebas y calidad de código
uv run pytest
uv run ruff check .

# 4. Pipeline de entrenamiento (Prefect + MLflow + Optuna) ~2-3 min con datos completos
uv run python pipelines/training_flow.py
#    Más rápido: uv run python pipelines/training_flow.py --n-trials 5

# 5. Ver experimentos y el Model Registry en MLflow  -> http://127.0.0.1:5000
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# 6. Levantar la API en Docker  -> http://localhost:8000  y  http://localhost:8000/docs
docker compose up --build

# 7. Simular tráfico y monitorear (en otra terminal, con la API arriba)
uv run python scripts/simulate_traffic.py --n 1000
uv run python pipelines/monitoring_flow.py --source api-log
uv run python pipelines/monitoring_flow.py                    # último mes con etiquetas
uv run python pipelines/monitoring_flow.py --simulate-drift   # demo de alerta
```

### Ejecución programada con Prefect

```bash
uv run prefect server start          # terminal 1 -> UI en http://127.0.0.1:4200
uv run python pipelines/deploy.py    # terminal 2: reentreno mensual + monitoreo semanal
uv run prefect deployment run "informalidad-training-pipeline/reentrenamiento-mensual"
```

### Problemas comunes en Windows

| Síntoma | Causa y solución |
|---|---|
| `ModuleNotFoundError: No module named 'informalidad'` | Pasaba cuando la ruta del proyecto tiene tildes (p. ej. `C:\Users\José\...`): Python 3.11 lee los `.pth` con la codificación de Windows. Ya está resuelto con `dev-mode-exact` en `pyproject.toml`; si aparece, correr `uv sync` de nuevo. |
| `[Errno 10048]` o la UI no abre en el puerto 5000 | Otro proceso (p. ej. otro `mlflow ui`) usa el puerto. Usar otro: `--port 5001`. |
| `failed to connect to the docker API` | Docker Desktop no está abierto. Abrirlo y esperar a que diga *Engine running*. |

Más detalles: [`docs/despliegue.md`](docs/despliegue.md) · API: [`docs/api.md`](docs/api.md) ·
Monitoreo: [`docs/monitoreo.md`](docs/monitoreo.md).

## 6. Estructura del repositorio

```
mlops-informalidad-laboral/
├── configs/config.yaml          # configuración central (rutas, features, MLflow, umbrales)
├── data/
│   ├── raw/                     # dataset completo (no se sube a git)
│   └── sample/                  # muestra versionada (10.000 filas)
├── src/informalidad/            # paquete instalable
│   ├── data/                    # carga, validación (contrato de datos), split temporal
│   ├── features/                # feature engineering dentro del sklearn Pipeline
│   ├── models/                  # candidatos, métricas, entrenamiento, Optuna, registry
│   ├── api/                     # FastAPI, esquemas Pydantic, carga del modelo, formulario web
│   └── monitoring/              # PSI, reporte de drift/desempeño/equidad
├── pipelines/                   # flujos de Prefect: entrenamiento, monitoreo, deployments
├── scripts/                     # construir dataset, exportar champion, muestra, simular tráfico
├── notebooks/                   # 01_eda, 02_baseline, 03_experiments
├── models/champion/             # modelo en servicio (lo empaqueta Docker)
├── tests/                       # unit/ + integration/ (37 pruebas)
├── docs/                        # arquitectura, API, despliegue, monitoreo, reportes de ejemplo
├── Dockerfile · docker-compose.yml · .dockerignore
├── .github/workflows/ci.yml     # CI: lint + tests + build y smoke test de Docker
├── .pre-commit-config.yaml
└── pyproject.toml · uv.lock
```

## 7. Cumplimiento de la guía del proyecto

| Fase | Requisito | Dónde |
|---|---|---|
| 1.1 Selección | Problema de negocio, métricas de éxito, alcance, timeline | Secciones 1, 2 y 8 de este README |
| 1.2 Setup | Repo con estructura clara, entorno con uv, dependencias | `pyproject.toml`, `uv.lock` |
| 1.3 Dataset | EDA, estrategia de preprocesamiento, baseline | `notebooks/01_eda.ipynb`, `02_baseline.ipynb` |
| 2.1 MLflow | Experimento, params, métricas, artifacts, firma del modelo | `src/informalidad/models/train.py` |
| 2.2 Experimentos | RF, XGBoost, LogReg, HPO, **cross-validation** | `candidates.py`, `optimize.py`, `train.py` |
| 2.3 Registry | Registro, tags, aliases candidate/champion, descripción | `src/informalidad/models/registry.py` |
| 3.1 Prefect | Flows, tasks con retries, **scheduling** | `pipelines/training_flow.py`, `deploy.py` |
| 3.2 Data pipeline | ETL, validación de datos, logging | `src/informalidad/data/` |
| 3.3 Model pipeline | Entrenamiento automático, evaluación, **reentrenamiento automático** ⭐ | `training_flow.py`, `monitoring_flow.py --auto-retrain` |
| 4.1 Docker | Dockerfile, **imagen optimizada (multi-stage, no root, ~350 MB)** ⭐ | `Dockerfile` |
| 4.2 API | FastAPI, `/predict`, `/predict/batch`, validación de inputs | `src/informalidad/api/` |
| 4.3 Cloud/CI-CD | **CI con GitHub Actions** ⭐ (nube: guía opcional) | `.github/workflows/ci.yml`, `docs/despliegue.md` |
| 5 Monitoreo | Diseño **e implementación**: PSI, desempeño, equidad, alertas | `docs/monitoreo.md`, `src/informalidad/monitoring/` |
| 6.1 Testing | Unit tests (+ integración) | `tests/` |
| 6.2 Code quality | Linter y formatter (ruff = flake8 + black), **pre-commit** ⭐ | `pyproject.toml`, `.pre-commit-config.yaml` |
| 6.3 Docs | README, API, guía de despliegue | `README.md`, `docs/` |

⭐ = ítems marcados como *nice to have* en la guía.

## Autor

**Juan José Villa** — Especialización en Ciencia de datos e Inteligencia Artificial, Universidad de Medellín.
