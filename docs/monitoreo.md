# Diseño de monitoreo

## ¿Qué puede salir mal en producción?

| Riesgo | Ejemplo en este problema | Cómo se detecta |
|---|---|---|
| **Data drift** (cambian las entradas) | Sube el salario mínimo, se reduce la jornada laboral (Ley 2101 de 2021) | PSI por variable |
| **Prediction drift** | El modelo empieza a predecir muchos más informales | PSI de la probabilidad predicha |
| **Concept drift** (cambia la relación X→y) | Una reforma pensional cambia quién cotiza | AUC/F1 con etiquetas nuevas de la GEIH |
| **Sesgo / equidad** | El modelo detecta peor la informalidad de las mujeres | Recall y precisión por sexo |
| **Datos inválidos** | Edad 500, código de sexo 3 | Validación Pydantic (API) y contrato de datos (pipeline) |
| **Salud del servicio** | La API se cae o se vuelve lenta | `/health` + healthcheck de Docker + latencia en el log |

## Lo que está implementado

1. **Log de predicciones** (`logs/predictions.jsonl`): la API guarda cada entrada, la probabilidad,
   la versión del modelo y la latencia. Es la materia prima del monitoreo.
2. **Perfil de referencia** (`models/champion/reference_profile.json`): al entrenar se guardan los
   cortes por cuantiles y las proporciones de cada variable, la distribución de la predicción y la
   tasa de informalidad de entrenamiento.
3. **Flujo de monitoreo** (`pipelines/monitoring_flow.py`, programado cada lunes):
   - PSI de cada variable y de la predicción frente al perfil de referencia.
   - Si hay etiquetas (mes nuevo de la GEIH): AUC, F1, precisión y recall, comparados con el test.
   - Equidad: tasa real vs predicha, recall y precisión por sexo.
   - Reporte en Markdown y JSON (`reports/monitoring/`) y como artifact en la UI de Prefect.
   - Opción `--auto-retrain`: si hay alertas, dispara el pipeline de entrenamiento (que solo
     reemplaza al champion si el nuevo modelo es mejor).

## Reglas de decisión

| Nivel | Condición | Acción |
|---|---|---|
| 🟢 Estable | Todas las variables con PSI < 0.2 | Ninguna |
| 🟡 Advertencia | Alguna **variable de entrada** con PSI ≥ 0.2 | Investigar la causa; no se reentrena solo por esto |
| 🔴 Alerta | PSI de la **predicción** ≥ 0.2, **o** el AUC cae más de 0.03 frente al test | Reentrenar |
| — | Menos de 200 registros | No se calculan alertas de drift (el PSI sería ruido) |

¿Por qué separar advertencias de alertas? Porque **no todo data drift daña el modelo**. El caso
real de febrero de 2026 lo muestra (ver [`ejemplo_reporte_monitoreo.md`](ejemplo_reporte_monitoreo.md)):

- `INGRESO_LABORAL` (PSI 0.55) y `HORAS_SEMANA` (PSI 0.31) cambiaron mucho: el ingreso nominal sube
  cada año y la jornada típica bajó de 48 a 44 horas.
- Pero la predicción casi no cambió (PSI 0.019) y el **AUC se mantuvo (0.9708 vs 0.9717)**.
- Conclusión: hay que vigilar, pero no hace falta reentrenar todavía.

En cambio, con un escenario simulado de choque económico (ingresos −45 % y 25 % de trabajadores
pasando a cuenta propia sin contrato), el AUC cae a 0.932 y la predicción se desplaza: el flujo
recomienda reentrenar ([`ejemplo_reporte_drift_simulado.md`](ejemplo_reporte_drift_simulado.md)).

## Cómo correrlo

```bash
# Último mes disponible (con etiquetas)
uv run python pipelines/monitoring_flow.py

# Sobre el tráfico real de la API (sin etiquetas)
uv run python scripts/simulate_traffic.py --n 1000      # con la API arriba
uv run python pipelines/monitoring_flow.py --source api-log

# Demostración de una alerta
uv run python pipelines/monitoring_flow.py --simulate-drift
```

## Siguiente etapa (propuesta)

- **Prometheus + Grafana**: exponer `/metrics` en la API (conteo de requests, latencia p95,
  tasa de informalidad predicha por hora) y montar un dashboard con alertas a Slack/correo.
- **Evidently AI** para reportes HTML interactivos de drift.
- **Ingreso en términos reales**: expresar el ingreso como múltiplos del salario mínimo de cada año.
  Eliminaría el drift "nominal" del ingreso que hoy genera advertencias todos los años.
- **Monitoreo por segmentos**: departamento (ya está en los datos) y posición ocupacional.
- **Retraso de las etiquetas**: la verdad (si cotiza o no) llega con la siguiente GEIH; mientras tanto
  solo se monitorea drift. Documentar ese retraso en el SLA del modelo.
