"""Calibración de probabilidades y corrección de prevalencia.

Incluye:
- Platt scaling de origen, ajustado con predicciones out-of-fold de 2020.
- Corrección de prevalencia por EM (Saerens-Latinne-Decaestecker).
- Corrección oracle con la prevalencia real (solo diagnóstico).
- Curvas de fiabilidad y ECE.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import SEED

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_source_platt(model, X, y, cv: int = 5):
    """Ajusta un mapeo Platt con predicciones out-of-fold del conjunto fuente."""
    oof = cross_val_predict(
        model,
        X,
        y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    platt = LogisticRegression()
    platt.fit(logit(oof).reshape(-1, 1), y)
    return platt, oof


def apply_platt(platt, p: np.ndarray) -> np.ndarray:
    return platt.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def prior_correct(p: np.ndarray, pi_source: float, pi_target: float) -> np.ndarray:
    """Corrección exacta de probabilidades bajo cambio de prevalencia."""
    w1 = p / pi_source
    w0 = (1 - p) / (1 - pi_source)
    num = w1 * pi_target
    den = num + w0 * (1 - pi_target)
    return num / den


def em_prior_shift(
    p: np.ndarray,
    pi_source: float,
    init: str = "mean",
    max_iter: int = 2000,
    tol: float = 1e-10,
):
    """EM de Saerens-Latinne-Decaestecker para estimar la prevalencia objetivo.

    Devuelve (pi_estimada, probabilidades_corregidas, iteraciones, convergio).
    """
    pi_target = float(np.mean(p)) if init == "mean" else float(pi_source)
    corrected = p
    for i in range(max_iter):
        w1 = p / pi_source
        w0 = (1 - p) / (1 - pi_source)
        num = w1 * pi_target
        den = num + w0 * (1 - pi_target)
        corrected = num / den
        new_pi = float(np.mean(corrected))
        if abs(new_pi - pi_target) < tol:
            return new_pi, corrected, i + 1, True
        pi_target = new_pi
    return pi_target, corrected, max_iter, False


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Curva de fiabilidad por cuantiles de la probabilidad predicha."""
    quantiles = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    idx = np.digitize(p, quantiles[1:-1])
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": b,
                "n": int(mask.sum()),
                "pred": float(np.mean(p[mask])),
                "obs": float(np.mean(y[mask])),
            }
        )
    return rows


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    rows = reliability_curve(y, p, n_bins=n_bins)
    total = len(y)
    return float(sum((r["n"] / total) * abs(r["obs"] - r["pred"]) for r in rows))


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    prevalence = float(np.mean(y))
    brier = float(brier_score_loss(y, p))
    baseline = prevalence * (1 - prevalence)
    return {
        "n": int(len(y)),
        "prevalencia": prevalence,
        "mean_p": float(np.mean(p)),
        "Brier": brier,
        "Brier_skill": float(1 - brier / baseline) if baseline > 0 else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ECE": expected_calibration_error(y, p),
        "AUC_PR": float(average_precision_score(y, p)),
        "ROC_AUC": float(roc_auc_score(y, p)),
    }
