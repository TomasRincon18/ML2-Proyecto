"""Modelo final desplegable: pipeline + calibrador Platt + umbral.

El artefacto guarda un objeto `FinalModel` que aplica, en orden:
1. El pipeline de scikit-learn (preprocesamiento + regresión logística).
2. El calibrador Platt ajustado con el holdout de 2021.
3. El umbral de decisión bayesiano para el costo FN/FP declarado.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class PlattCalibrator:
    """Mapeo Platt 1D sobre el logit de la probabilidad cruda."""

    def __init__(self, coef: float, intercept: float):
        self.coef = float(coef)
        self.intercept = float(intercept)

    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        return sigmoid(self.coef * logit(p) + self.intercept)

    def __repr__(self) -> str:
        return f"PlattCalibrator(coef={self.coef:.4f}, intercept={self.intercept:.4f})"


class BetaCalibrator:
    """Calibración beta (Kull et al., 2017) sobre probabilidades crudas.

    logit(p_cal) = a * log(p) - b * log(1 - p) + c
    """

    def __init__(self, a: float, b: float, c: float):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, EPS, 1 - EPS)
        z = self.a * np.log(p) - self.b * np.log(1 - p) + self.c
        return sigmoid(z)

    def __repr__(self) -> str:
        return f"BetaCalibrator(a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f})"


class FinalModel:
    """Wrapper picklable para serving y evaluación."""

    def __init__(self, model, calibrator=None, threshold: float = 0.5, features=None):
        self.model = model
        self.calibrator = calibrator
        self.threshold = float(threshold)
        self.features = list(features or [])

    def raw_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        p = self.raw_proba(X)
        if self.calibrator is not None:
            p = self.calibrator.predict_proba(p)
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def __repr__(self) -> str:
        return (
            f"FinalModel(threshold={self.threshold:.4f}, "
            f"calibrator={self.calibrator!r})"
        )
