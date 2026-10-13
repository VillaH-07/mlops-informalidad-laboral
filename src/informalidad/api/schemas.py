"""Contratos de entrada/salida de la API (validación con Pydantic).

Los códigos categóricos son los del diccionario de datos de la GEIH (DANE). Si llega un valor
fuera del dominio, la API responde 422 con un mensaje claro en vez de predecir basura.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EJEMPLO_TRABAJADOR = {
    "EDAD": 34,
    "HORAS_SEMANA": 48,
    "INGRESO_LABORAL": 1300000,
    "SEXO": 1,
    "NIVEL_EDUCATIVO": 5,
    "ESTADO_CIVIL": 2,
    "AFILIADO_SALUD": 1,
    "POSICION_OCUPACIONAL": 4,
    "TIENE_CONTRATO": 2,
    "RAMA_ACTIVIDAD": 47,
}


class Trabajador(BaseModel):
    """Características de un trabajador ocupado (una fila de la GEIH)."""

    model_config = ConfigDict(json_schema_extra={"example": EJEMPLO_TRABAJADOR})

    EDAD: int = Field(..., ge=10, le=110, description="Edad en años")
    HORAS_SEMANA: int = Field(..., ge=0, le=168, description="Horas trabajadas a la semana")
    INGRESO_LABORAL: float = Field(
        ..., ge=0, le=500_000_000, description="Ingreso laboral mensual en pesos (INGLABO)"
    )
    SEXO: Literal[1, 2] = Field(..., description="1=Hombre, 2=Mujer")
    NIVEL_EDUCATIVO: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 99] = Field(
        ...,
        description=(
            "Mayor nivel educativo alcanzado (P3042): 1=Ninguno, 2=Preescolar, 3=Básica primaria, "
            "4=Básica secundaria, 5=Media académica, 6=Media técnica, 7=Normalista, "
            "8=Técnica profesional, 9=Tecnológica, 10=Universitaria, 11=Especialización, "
            "12=Maestría, 13=Doctorado, 99=No sabe"
        ),
    )
    ESTADO_CIVIL: Literal[1, 2, 3, 4, 5, 6] = Field(
        ...,
        description=(
            "1=Unión libre <2 años, 2=Unión libre >=2 años, 3=Casado(a), "
            "4=Separado(a)/divorciado(a), 5=Viudo(a), 6=Soltero(a)"
        ),
    )
    AFILIADO_SALUD: Literal[1, 2, 9] = Field(
        ..., description="Afiliado a seguridad social en salud: 1=Sí, 2=No, 9=No sabe"
    )
    POSICION_OCUPACIONAL: Literal[1, 2, 3, 4, 5, 6, 7, 8] = Field(
        ...,
        description=(
            "1=Empleado particular, 2=Empleado del gobierno, 3=Empleado doméstico, "
            "4=Cuenta propia, 5=Patrón o empleador, 6=Trabajador familiar sin remuneración, "
            "7=Trabajador sin remuneración en empresas de otros hogares, 8=Jornalero o peón"
        ),
    )
    TIENE_CONTRATO: Literal[1, 2] = Field(..., description="Tiene contrato: 1=Sí, 2=No")
    RAMA_ACTIVIDAD: int = Field(
        ..., ge=0, le=99, description="Rama de actividad económica CIIU Rev. 4 a 2 dígitos"
    )


class LoteTrabajadores(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"trabajadores": [EJEMPLO_TRABAJADOR]}})

    trabajadores: list[Trabajador] = Field(..., min_length=1, max_length=1000)


class Prediccion(BaseModel):
    probabilidad_informal: float = Field(..., description="Probabilidad de NO cotizar a pensión")
    prediccion: int = Field(..., description="1=Informal, 0=Formal")
    etiqueta: Literal["Informal", "Formal"]
    umbral: float
    version_modelo: str


class PrediccionLote(BaseModel):
    predicciones: list[Prediccion]
    total: int
    tasa_informalidad_predicha: float


class Salud(BaseModel):
    status: Literal["ok"]
    modelo_cargado: bool
    version_modelo: str


class InfoModelo(BaseModel):
    model_name: str
    version: str
    alias: str
    run_id: str
    threshold: float
    metrics: dict[str, float]
    features: dict[str, list[str]]
    data_version: str | None = None
    test_periods: str | None = None
    trained_at: str
    library_versions: dict[str, str]
