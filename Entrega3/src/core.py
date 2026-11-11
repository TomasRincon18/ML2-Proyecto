"""Lógica compartida entre la app y el entrenamiento (serving).

Contiene las funciones de predicción y de explicación local (SHAP en log-odds
y what-if en probabilidad). Se apoya en el artefacto `model_final.joblib`, que
ya incluye el background SHAP, los coeficientes, las referencias y las
categorías, de modo que la app no necesita cargar el dataset.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, FEATURES

GROUP_LABELS = {
    "edad_anios": "Edad",
    "sexo": "Sexo",
    "ciudad_top": "Ciudad",
    "fuente_tipo_contagio": "Fuente de contagio",
}

FEATURE_LABELS = {
    "edad_anios": "Edad (años)",
    "sexo": "Sexo",
    "ciudad_top": "Ciudad",
    "fuente_tipo_contagio": "Fuente de contagio",
}


def load_artifact(path: str | Path | None = None) -> dict:
    path = Path(path) if path is not None else ARTIFACTS_DIR / "model_final.joblib"
    return joblib.load(path)


def _frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame([values], columns=FEATURES)


def predict_case(artifact: dict, values: dict) -> dict:
    """Predicción calibrada para un caso."""
    final_model = artifact["final_model"]
    X = _frame(values)
    proba = final_model.predict_proba(X)[0]
    p_cal = float(proba[1])
    return {
        "p_calibrada": p_cal,
        "prediccion": 1 if p_cal >= final_model.threshold else 0,
        "threshold": float(final_model.threshold),
    }


def predict_frame(artifact: dict, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Predicción vectorizada para lotes."""
    final_model = artifact["final_model"]
    thr = float(final_model.threshold if threshold is None else threshold)
    p = final_model.predict_proba(df[FEATURES])[:, 1]
    out = df.copy()
    out["riesgo_calibrado"] = p
    out["prediccion"] = (p >= thr).astype(int)
    return out


def explain_log_odds(artifact: dict, values: dict) -> dict:
    """Contribuciones SHAP exactas del modelo lineal, en log-odds."""
    final_model = artifact["final_model"]
    prep = final_model.model.named_steps["prep"]
    shap_meta = artifact["shap"]

    coef = np.asarray(shap_meta["coef"], dtype=float)
    background = np.asarray(shap_meta["background_mean"], dtype=float)
    intercept = float(shap_meta["intercept"])
    groups = shap_meta["groups"]

    x = np.asarray(prep.transform(_frame(values)), dtype=float)[0]
    delta = x - background
    contrib = delta * coef

    base = float(intercept + np.dot(background, coef))
    logit = float(intercept + np.dot(x, coef))
    contributions = {
        g: float(contrib[idx].sum()) for g, idx in groups.items()
    }
    return {"contribuciones": contributions, "base": base, "logit": logit}


def what_if(artifact: dict, values: dict) -> dict:
    """Cambio de riesgo calibrado si cada variable tomara su valor de referencia."""
    final_model = artifact["final_model"]
    references = artifact["shap"]["references"]

    p_actual = float(final_model.predict_proba(_frame(values))[0][1])
    cambios = {}
    for feat in FEATURES:
        values_ref = dict(values)
        values_ref[feat] = references[feat]
        p_ref = float(final_model.predict_proba(_frame(values_ref))[0][1])
        cambios[feat] = {
            "actual": values[feat],
            "referencia": references[feat],
            "delta_prob": float(p_ref - p_actual),
        }
    return {"p_actual": p_actual, "cambios": cambios}
