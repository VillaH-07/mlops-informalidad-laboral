# API de predicción

Base local: `http://localhost:8000` · Documentación interactiva (Swagger): `http://localhost:8000/docs`

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Formulario web para probar el modelo |
| GET | `/health` | Estado del servicio (lo usa el healthcheck de Docker) |
| GET | `/model/info` | Versión, métricas, umbral y fecha de entrenamiento del modelo en servicio |
| POST | `/predict` | Predicción para un trabajador |
| POST | `/predict/batch` | Predicción para 1 a 1000 trabajadores |

## Entrada: un trabajador

| Campo | Tipo | Valores válidos |
|---|---|---|
| `EDAD` | int | 10 – 110 |
| `HORAS_SEMANA` | int | 0 – 168 |
| `INGRESO_LABORAL` | float | 0 – 500.000.000 (pesos/mes) |
| `SEXO` | int | 1 = Hombre, 2 = Mujer |
| `NIVEL_EDUCATIVO` | int | 1 – 13 o 99: mayor nivel alcanzado, P3042 de la GEIH (1 Ninguno, 3 Primaria, 5 Media, 10 Universitaria, 13 Doctorado; ver `/docs`) |
| `ESTADO_CIVIL` | int | 1 – 6 (diccionario GEIH, ver `/docs`) |
| `AFILIADO_SALUD` | int | 1 = Sí, 2 = No, 9 = No sabe |
| `POSICION_OCUPACIONAL` | int | 1 – 8 (diccionario GEIH, ver `/docs`) |
| `TIENE_CONTRATO` | int | 1 = Sí, 2 = No |
| `RAMA_ACTIVIDAD` | int | 0 – 99 (CIIU Rev. 4, 2 dígitos) |

Si un campo falta o está fuera de rango, la API responde **422** con el detalle del error.

## Ejemplos

**PowerShell (Windows)**

```powershell
$body = @{
  EDAD = 34; HORAS_SEMANA = 48; INGRESO_LABORAL = 1300000; SEXO = 1; NIVEL_EDUCATIVO = 5
  ESTADO_CIVIL = 2; AFILIADO_SALUD = 1; POSICION_OCUPACIONAL = 4; TIENE_CONTRATO = 2; RAMA_ACTIVIDAD = 47
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Body $body -ContentType "application/json"
```

**curl (Mac / Linux / Git Bash)**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"EDAD":34,"HORAS_SEMANA":48,"INGRESO_LABORAL":1300000,"SEXO":1,"NIVEL_EDUCATIVO":5,
       "ESTADO_CIVIL":2,"AFILIADO_SALUD":1,"POSICION_OCUPACIONAL":4,"TIENE_CONTRATO":2,"RAMA_ACTIVIDAD":47}'
```

**Respuesta**

```json
{
  "probabilidad_informal": 0.9359,
  "prediccion": 1,
  "etiqueta": "Informal",
  "umbral": 0.4949,
  "version_modelo": "1"
}
```

`umbral` es el punto de corte que maximiza el F1 en validación (no 0.5 por defecto). Si la
probabilidad es mayor o igual al umbral, la predicción es "Informal".

**Lote**

```bash
curl -X POST http://localhost:8000/predict/batch -H "Content-Type: application/json" \
  -d '{"trabajadores": [ {...}, {...} ]}'
```

Devuelve `predicciones`, `total` y `tasa_informalidad_predicha`.

## Registro para monitoreo

Cada predicción se agrega a `logs/predictions.jsonl` (entrada, probabilidad, versión del modelo,
latencia). Se puede desactivar con la variable de entorno `LOG_PREDICTIONS=false`.
