# Reporte de monitoreo del modelo

- Generado: 2026-10-12T23:13:54.454077+00:00
- Fuente: `GEIH Febrero 2026 + drift simulado` (29,349 registros)
- Estado: **🔴 Reentrenamiento recomendado**

## Alertas
- Drift en PREDICCION (PSI=0.215): cambió la distribución de las predicciones
- Caída de desempeño: AUC 0.9324 vs 0.9717 en referencia

## Advertencias
- Drift en INGRESO_LABORAL (PSI=0.676)
- Drift en HORAS_SEMANA (PSI=0.309)

## Desempeño
| Métrica | Actual | Referencia (test) |
|---|---|---|
| roc_auc | 0.9324 | 0.9717 |
| f1 | 0.8226 | 0.9185 |
| precision | 0.7074 | 0.9338 |
| recall | 0.9826 | 0.9036 |

## Data drift (PSI)
| variable             |    psi | nivel   |
|:---------------------|-------:|:--------|
| INGRESO_LABORAL      | 0.6757 | alerta  |
| HORAS_SEMANA         | 0.3091 | alerta  |
| PREDICCION           | 0.2149 | alerta  |
| TIENE_CONTRATO       | 0.0811 | estable |
| POSICION_OCUPACIONAL | 0.0772 | estable |
| RAMA_ACTIVIDAD       | 0.0109 | estable |
| NIVEL_EDUCATIVO      | 0.0094 | estable |
| AFILIADO_SALUD       | 0.0015 | estable |
| EDAD                 | 0.0006 | estable |
| ESTADO_CIVIL         | 0.0002 | estable |
| SEXO                 | 0.0001 | estable |

## Equidad por sexo
| grupo   |     n |   tasa_real |   tasa_predicha |   recall |   precision |
|:--------|------:|------------:|----------------:|---------:|------------:|
| Hombre  | 16222 |      0.5768 |          0.7968 |   0.9808 |      0.71   |
| Mujer   | 13127 |      0.5289 |          0.7402 |   0.985  |      0.7038 |
