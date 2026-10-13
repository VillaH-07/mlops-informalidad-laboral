"""Construye el dataset del modelo a partir de los microdatos crudos de la GEIH (DANE).

Une, para cada mes, el módulo "Características generales, seguridad social en salud y
educación" con el módulo "Ocupados" (llaves DIRECTORIO, SECUENCIA_P, ORDEN), selecciona las
variables del modelo y crea la variable objetivo:

    INFORMAL = 1 si el ocupado NO cotiza a pensión (P6920 = 2), 0 si cotiza (P6920 = 1).
    Los ya pensionados (P6920 = 3) se excluyen.

Variables (códigos según el diccionario de datos de la GEIH, marco 2018):

    P3271 -> SEXO                  P6430     -> POSICION_OCUPACIONAL
    P6040 -> EDAD                  P6440     -> TIENE_CONTRATO
    P3042 -> NIVEL_EDUCATIVO       P6500     -> INGRESO_MONETARIO
    P6070 -> ESTADO_CIVIL          P6800     -> HORAS_SEMANA
    DPTO  -> DEPARTAMENTO          P6920     -> COTIZA_PENSION
    P6090 -> AFILIADO_SALUD        INGLABO   -> INGRESO_LABORAL
                                   RAMA2D_R4 -> RAMA_ACTIVIDAD

Nota: el proyecto anterior usaba P6083 como nivel educativo, pero P6083 es
"¿La madre de ... reside en este hogar?". El nivel educativo es P3042.

Uso (solo necesita pandas):
    uv run python scripts/build_dataset.py --raw-dir "ruta/a/data/raw"

La carpeta --raw-dir debe tener una subcarpeta por mes ("Enero 2024", ...) con los CSV del
DANE en cualquier nivel de profundidad (el DANE cambia la estructura de carpetas entre meses).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

MESES = [
    "Enero 2024",
    "Febrero 2024",
    "Marzo 2024",
    "Abril 2024",
    "Enero 2025",
    "Febrero 2025",
    "Marzo 2025",
    "Abril 2025",
    "Enero 2026",
    "Febrero 2026",
]

LLAVES = ["DIRECTORIO", "SECUENCIA_P", "ORDEN"]

# código DANE -> nombre en el proyecto, separado por el módulo del que se toma
VARIABLES_CARACTERISTICAS = {
    "P3271": "SEXO",
    "P6040": "EDAD",
    "P3042": "NIVEL_EDUCATIVO",
    "P6070": "ESTADO_CIVIL",
    "DPTO": "DEPARTAMENTO",
    "P6090": "AFILIADO_SALUD",
}
VARIABLES_OCUPADOS = {
    "P6430": "POSICION_OCUPACIONAL",
    "P6440": "TIENE_CONTRATO",
    "P6500": "INGRESO_MONETARIO",
    "P6800": "HORAS_SEMANA",
    "P6920": "COTIZA_PENSION",
    "INGLABO": "INGRESO_LABORAL",
    "RAMA2D_R4": "RAMA_ACTIVIDAD",
}

CODIGOS_ENTEROS = [
    "SEXO",
    "EDAD",
    "NIVEL_EDUCATIVO",
    "ESTADO_CIVIL",
    "DEPARTAMENTO",
    "AFILIADO_SALUD",
    "POSICION_OCUPACIONAL",
    "TIENE_CONTRATO",
    "HORAS_SEMANA",
    "COTIZA_PENSION",
    "RAMA_ACTIVIDAD",
]

COLUMNAS_SALIDA = [
    "SEXO",
    "EDAD",
    "NIVEL_EDUCATIVO",
    "ESTADO_CIVIL",
    "DEPARTAMENTO",
    "AFILIADO_SALUD",
    "POSICION_OCUPACIONAL",
    "TIENE_CONTRATO",
    "INGRESO_MONETARIO",
    "HORAS_SEMANA",
    "COTIZA_PENSION",
    "INGRESO_LABORAL",
    "RAMA_ACTIVIDAD",
    "MES_PERIODO",
    "INFORMAL",
]


def buscar_csv(carpeta_mes: Path, modulo: str) -> Path:
    """Busca el CSV de un módulo ('caracteristicas' u 'ocupados') dentro de la carpeta del mes."""
    candidatos = []
    for raiz, _, archivos in os.walk(carpeta_mes):
        for nombre in archivos:
            n = nombre.lower()
            if not n.endswith(".csv"):
                continue
            if modulo == "caracteristicas" and n.startswith("caracter"):
                candidatos.append(Path(raiz) / nombre)
            # "No ocupados.CSV" también contiene "ocupados": por eso se exige que empiece así
            if modulo == "ocupados" and n.startswith("ocupados"):
                candidatos.append(Path(raiz) / nombre)
    if len(candidatos) != 1:
        raise FileNotFoundError(
            f"Se esperaba 1 archivo de '{modulo}' en {carpeta_mes}, se encontraron: {candidatos}"
        )
    return candidatos[0]


def leer_modulo(ruta: Path, variables: dict[str, str]) -> pd.DataFrame:
    """Lee solo las llaves y las columnas necesarias (los CSV del DANE usan ';' y latin-1)."""
    encabezado = pd.read_csv(ruta, sep=";", encoding="latin-1", nrows=0).columns
    faltantes = [c for c in [*LLAVES, *variables] if c not in encabezado]
    if faltantes:
        raise KeyError(f"{ruta.name}: faltan las columnas {faltantes}")
    df = pd.read_csv(
        ruta, sep=";", encoding="latin-1", usecols=[*LLAVES, *variables], low_memory=False
    )
    return df.rename(columns=variables)


def construir_mes(carpeta_mes: Path, mes: str) -> pd.DataFrame:
    caract = leer_modulo(buscar_csv(carpeta_mes, "caracteristicas"), VARIABLES_CARACTERISTICAS)
    ocup = leer_modulo(buscar_csv(carpeta_mes, "ocupados"), VARIABLES_OCUPADOS)
    datos = caract.merge(ocup, on=LLAVES, how="inner", validate="one_to_one")
    datos["MES_PERIODO"] = mes
    print(
        f"  {mes:<13} caracteristicas={len(caract):>7,}  ocupados={len(ocup):>6,}  "
        f"unidos={len(datos):>6,}"
    )
    return datos


def construir_dataset(raw_dir: Path) -> pd.DataFrame:
    partes = [construir_mes(raw_dir / mes, mes) for mes in MESES]
    df = pd.concat(partes, ignore_index=True)

    # Variable objetivo (se excluyen los pensionados, P6920 = 3)
    df = df[df["COTIZA_PENSION"].isin([1, 2])].copy()
    df["INFORMAL"] = (df["COTIZA_PENSION"] == 2).astype(int)

    # Limpieza mínima, igual a la del proyecto anterior
    df["INGRESO_MONETARIO"] = df["INGRESO_MONETARIO"].fillna(0)
    # Ingreso faltante: mediana del MISMO mes (no se mezcla información de meses futuros)
    df["INGRESO_LABORAL"] = df["INGRESO_LABORAL"].fillna(
        df.groupby("MES_PERIODO")["INGRESO_LABORAL"].transform("median")
    )
    for col in ["ESTADO_CIVIL", "AFILIADO_SALUD"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Los códigos del DANE son enteros (pandas los lee como float cuando la columna tiene vacíos)
    for col in CODIGOS_ENTEROS:
        df[col] = df[col].astype("Int64")

    return df[COLUMNAS_SALIDA].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir", required=True, type=Path, help="Carpeta con una subcarpeta por mes de la GEIH"
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw/geih_informalidad_10meses.csv"))
    args = parser.parse_args()

    print(f"Leyendo microdatos de {args.raw_dir}")
    df = construir_dataset(args.raw_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nDataset guardado en {args.out}: {len(df):,} filas x {df.shape[1]} columnas")
    print(f"Tasa de informalidad: {df['INFORMAL'].mean():.1%}")
    print("Nulos por columna:", {c: int(n) for c, n in df.isna().sum().items() if n})
    conteo = df["NIVEL_EDUCATIVO"].value_counts().sort_index()
    print("NIVEL_EDUCATIVO (P3042):", {int(k): int(v) for k, v in conteo.items()})


if __name__ == "__main__":
    main()
