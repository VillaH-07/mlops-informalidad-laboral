"""Genera la muestra versionada en git (data/sample/) a partir del dataset completo.

La muestra permite que cualquier persona (evaluadores, CI) corra el proyecto completo sin
descargar los ~38 GB de microdatos de la GEIH. Se estratifica por mes para conservar los
10 periodos (y así el split temporal funciona igual que con los datos completos).

Uso:
    uv run python scripts/make_sample.py --rows-per-period 1000
"""

from __future__ import annotations

import argparse

from informalidad.config import get_config, resolve_path
from informalidad.data import load_dataset


def main(rows_per_period: int) -> None:
    cfg = get_config()
    data_cfg = cfg["data"]
    df = load_dataset(resolve_path(data_cfg["raw_path"]))
    sample = df.groupby(data_cfg["period_column"], group_keys=False).sample(
        n=rows_per_period, random_state=cfg["project"]["random_state"]
    )
    out = resolve_path(data_cfg["sample_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(f"Muestra guardada en {out} ({len(sample):,} filas)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-period", type=int, default=1000)
    main(parser.parse_args().rows_per_period)
