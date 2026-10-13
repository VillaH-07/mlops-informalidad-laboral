# Reporte de monitoreo del modelo

- Generado: 2026-10-12T23:13:38.515786+00:00
- Fuente: `GEIH Febrero 2026` (29,349 registros)
- Estado: **🟡 Estable, con advertencias de drift**

## Advertencias
- Drift en INGRESO_LABORAL (PSI=0.551)
- Drift en HORAS_SEMANA (PSI=0.309)

## Desempeño
| Métrica | Actual | Referencia (test) |
|---|---|---|
| roc_auc | 0.9708 | 0.9717 |
| f1 | 0.9169 | 0.9185 |
| precision | 0.9318 | 0.9338 |
| recall | 0.9025 | 0.9036 |

## Data drift (PSI)
| variable             |    psi | nivel   |
|:---------------------|-------:|:--------|
| INGRESO_LABORAL      | 0.5513 | alerta  |
| HORAS_SEMANA         | 0.3091 | alerta  |
| PREDICCION           | 0.019  | estable |
| RAMA_ACTIVIDAD       | 0.0109 | estable |
| NIVEL_EDUCATIVO      | 0.0094 | estable |
| AFILIADO_SALUD       | 0.0015 | estable |
| POSICION_OCUPACIONAL | 0.0006 | estable |
| EDAD                 | 0.0006 | estable |
| TIENE_CONTRATO       | 0.0004 | estable |
| ESTADO_CIVIL         | 0.0002 | estable |
| SEXO                 | 0.0001 | estable |

## Equidad por sexo
| grupo   |     n |   tasa_real |   tasa_predicha |   recall |   precision |
|:--------|------:|------------:|----------------:|---------:|------------:|
| Hombre  | 16222 |      0.5768 |          0.5588 |   0.8957 |      0.9245 |
| Mujer   | 13127 |      0.5289 |          0.5122 |   0.9117 |      0.9415 |
