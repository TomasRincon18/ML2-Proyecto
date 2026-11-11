"""Carga, validación y partición temporal del dataset consolidado del INS."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    DATA_PATH,
    FEATURES,
    TARGET,
    WAVE_CALIBRATION,
    WAVE_TEST,
    WAVE_TRAIN,
)


def load_dataset(path: str | Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Carga el dataset consolidado y normaliza nombres de columnas."""
    path = Path(path) if path is not None else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {path}. "
            "Ejecuta desde Entrega3 o define ML2_DATA_PATH."
        )

    df = pd.read_csv(path, nrows=nrows)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Verifica columnas, nulos y etiquetas de ola antes de modelar."""
    required = {TARGET, *FEATURES, "ola"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    nulls = df[list(required)].isna().sum()
    if int(nulls.sum()) > 0:
        raise ValueError(f"Hay valores nulos en columnas clave:\n{nulls[nulls > 0]}")

    expected = {WAVE_TRAIN, WAVE_CALIBRATION, WAVE_TEST}
    unknown = set(df["ola"].unique()) - expected
    if unknown:
        raise ValueError(f"Etiquetas de ola desconocidas: {sorted(unknown)}")


def wave(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Devuelve una ola con índice reiniciado."""
    subset = df.loc[df["ola"] == label].reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"La ola {label!r} no tiene registros.")
    return subset


def split_waves(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Devuelve las tres cohortes temporales del proyecto."""
    return {
        "2020": wave(df, WAVE_TRAIN),
        "2021": wave(df, WAVE_CALIBRATION),
        "2022": wave(df, WAVE_TEST),
    }


def X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa predictores y variable objetivo con el orden congelado."""
    return df[FEATURES].copy(), df[TARGET].astype(int).copy()
