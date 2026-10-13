"""Simula tráfico de producción: envía trabajadores reales de la muestra a la API.

Sirve para demostrar el ciclo completo: API -> logs/predictions.jsonl -> flujo de monitoreo.

Uso (con la API corriendo en otra terminal, local o en Docker):
    uv run python scripts/simulate_traffic.py --n 1000
    uv run python pipelines/monitoring_flow.py --source api-log
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from informalidad.config import get_config, raw_feature_columns, resolve_path
from informalidad.data import load_dataset


def main(url: str, n: int, period: str | None, batch_size: int) -> None:
    cfg = get_config()["data"]
    df = load_dataset(resolve_path(cfg["sample_path"]))
    if period:
        df = df[df[cfg["period_column"]] == period]
    rows = df.sample(n=min(n, len(df)), random_state=1)[raw_feature_columns()]
    records = json.loads(rows.to_json(orient="records"))

    sent, informales = 0, 0
    for start in range(0, len(records), batch_size):
        payload = json.dumps({"trabajadores": records[start : start + batch_size]}).encode()
        req = urllib.request.Request(
            f"{url}/predict/batch", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        sent += body["total"]
        informales += sum(p["prediccion"] for p in body["predicciones"])
    print(f"Enviados {sent} trabajadores a {url}. Predichos informales: {informales / sent:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--period", default=None, help='Ej: "Febrero 2026"')
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    main(args.url, args.n, args.period, args.batch_size)
