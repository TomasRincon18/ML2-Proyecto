"""Configuración central del proyecto — Entrega 3.

Congela rutas, semilla, variables y el pipeline de la fase 2 para que
entrenamiento, calibración, interpretabilidad y despliegue usen exactamente
las mismas definiciones.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]          # .../Proyecto/Entrega3
PROJECT_DIR = BASE_DIR.parent                           # .../Proyecto

DEFAULT_DATA_PATH = (
    PROJECT_DIR / "Entrega2" / "data" / "processed" / "dataset_reducido.csv"
)
DATA_PATH = Path(os.environ.get("ML2_DATA_PATH", DEFAULT_DATA_PATH))

ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = BASE_DIR / "figures"
INFORME_DIR = BASE_DIR / "informe"
DOCS_DIR = BASE_DIR / "docs"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Problema predictivo
# ---------------------------------------------------------------------------
TARGET = "grave"

# Variables seleccionadas en la fase 2 (sin fuga de información).
FEATURES = [
    "edad_anios",
    "sexo",
    "ciudad_top",
    "fuente_tipo_contagio",
]
NUMERIC_FEATURES = ["edad_anios"]
CATEGORICAL_FEATURES = ["sexo", "ciudad_top", "fuente_tipo_contagio"]

# Partición temporal: 2020 entrena, 2021 calibra, 2022 test final.
WAVE_TRAIN = "2020 (original)"
WAVE_CALIBRATION = "2021 (Delta)"
WAVE_TEST = "2022 (Omicron)"
WAVE_LABELS = {
    "2020": WAVE_TRAIN,
    "2021": WAVE_CALIBRATION,
    "2022": WAVE_TEST,
}

# ---------------------------------------------------------------------------
# Modelo congelado (fase 2)
# ---------------------------------------------------------------------------
LR_C = 0.1
LR_MAX_ITER = 1000
CLASS_WEIGHT = "balanced"

# Rejilla de búsqueda para la validación cruzada de B2.
C_GRID = [0.01, 0.1, 1.0, 10.0]

# Umbrales candidatos para la tabla de decisión (rango completo).
THRESHOLD_GRID = [round(i / 100, 2) for i in range(5, 96)]

# Sensibilidad clínica mínima para el umbral alternativo.
MIN_RECALL = 0.80

# Costos relativos para el umbral clínico: cuántos falsos positivos
# equivale un falso negativo. Se declara 10 como escenario base.
COST_RATIOS = [1, 2, 5, 10, 20, 50, 100]
DEFAULT_COST_RATIO = 10
