"""Entrenamiento y evaluación del modelo final — Entrega 3.

Uso:
    python src/train_final.py --smoke     # verificación rápida (B1)
    python src/train_final.py             # entrenamiento completo (B2)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

# Permite `python src/train_final.py` además de `python -m src.train_final`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    ARTIFACTS_DIR,
    C_GRID,
    COST_RATIOS,
    DEFAULT_COST_RATIO,
    FEATURES,
    METRICS_PATH,
    MIN_RECALL,
    MODEL_PATH,
    SEED,
    TARGET,
    THRESHOLD_GRID,
)
from src.data import load_dataset, split_waves, validate_dataset, X_y  # noqa: E402
from src.pipeline import build_model, transformed_feature_names  # noqa: E402


def _metrics(y_true, probs, threshold: float) -> dict:
    pred = (probs >= threshold).astype(int)
    prevalence = float(np.mean(y_true))
    ap = float(average_precision_score(y_true, probs))
    return {
        "n": int(len(y_true)),
        "prevalencia": prevalence,
        "ROC_AUC": float(roc_auc_score(y_true, probs)),
        "AUC_PR": ap,
        "AUC_PR_ratio_prevalencia": float(ap / prevalence) if prevalence else None,
        "Brier": float(brier_score_loss(y_true, probs)),
        "threshold": float(threshold),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def threshold_table(y_true, probs) -> list[dict]:
    """Tabla de F1/precisión/recall para umbrales candidatos."""
    rows = []
    for t in THRESHOLD_GRID:
        pred = (probs >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "positivos": int(pred.sum()),
            }
        )
    return rows


def _best_threshold(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: (r["f1"], r["recall"]))


def _recall_threshold(rows: list[dict], min_recall: float = MIN_RECALL) -> dict:
    candidates = [r for r in rows if r["recall"] >= min_recall]
    if not candidates:
        return max(rows, key=lambda r: r["recall"])
    return max(candidates, key=lambda r: (r["precision"], r["f1"]))


def cost_table(y_true, probs) -> list[dict]:
    """Umbral que minimiza FN * ratio + FP para distintos costos relativos."""
    rows = []
    for ratio in COST_RATIOS:
        best = None
        for t in THRESHOLD_GRID:
            pred = (probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            cost = float(fn * ratio + fp)
            candidate = {
                "ratio_fn_fp": ratio,
                "threshold": float(t),
                "cost": cost,
                "FN": int(fn),
                "FP": int(fp),
                "TP": int(tp),
                "TN": int(tn),
                "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
                "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
                "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
                "positivos": int(pred.sum()),
            }
            if best is None or candidate["cost"] < best["cost"]:
                best = candidate
        rows.append(best)
    return rows


def _cost_choice(rows: list[dict], ratio: float = DEFAULT_COST_RATIO) -> dict:
    for row in rows:
        if row["ratio_fn_fp"] == ratio:
            return row
    return min(rows, key=lambda r: abs(r["ratio_fn_fp"] - ratio))


def smoke(data_path: str | None) -> None:
    """Verificación rápida: carga, pipeline, predicción y métricas."""
    df = load_dataset(data_path)
    validate_dataset(df)
    waves = split_waves(df)

    train = waves["2020"].sample(n=min(5000, len(waves["2020"])), random_state=SEED)
    X, y = X_y(train)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    model = build_model()
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_va)[:, 1]

    rows = threshold_table(y_va, probs)
    best = _best_threshold(rows)
    metrics = _metrics(y_va, probs, best["threshold"])

    print("=== SMOKE TEST — ENTREGA 3 ===")
    print(f"Dataset: {len(df):,} filas | 2020: {len(waves['2020']):,}")
    print(f"Train smoke: {len(X_tr):,} | Validación smoke: {len(X_va):,}")
    print(f"Columnas transformadas: {len(transformed_feature_names(model))}")
    print(f"AUC-PR: {metrics['AUC_PR']:.4f} | ROC-AUC: {metrics['ROC_AUC']:.4f}")
    print(f"Umbral F1-óptimo: {best['threshold']:.2f} | F1: {best['f1']:.4f}")
    print(f"Prevalencia validación: {metrics['prevalencia']:.4f}")
    print("OK: pipeline reproducible y sin errores de compatibilidad.")


def full(data_path: str | None) -> None:
    """Entrenamiento completo: CV 2020, ajuste, umbral y evaluación 2021/2022."""
    df = load_dataset(data_path)
    validate_dataset(df)
    waves = split_waves(df)

    X_train, y_train = X_y(waves["2020"])
    X_cal, y_cal = X_y(waves["2021"])
    X_test, y_test = X_y(waves["2022"])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        build_model(),
        param_grid={"clf__C": C_GRID},
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    best_c = float(search.best_params_["clf__C"])

    probs_cal = model.predict_proba(X_cal)[:, 1]
    probs_test = model.predict_proba(X_test)[:, 1]

    rows = threshold_table(y_cal, probs_cal)
    best_f1 = _best_threshold(rows)
    best_recall = _recall_threshold(rows)
    cost_rows = cost_table(y_cal, probs_cal)
    cost_choice = _cost_choice(cost_rows)

    threshold = float(cost_choice["threshold"])
    threshold_f1 = float(best_f1["threshold"])
    threshold_recall = float(best_recall["threshold"])

    metrics_cal = _metrics(y_cal, probs_cal, threshold)
    metrics_test = _metrics(y_test, probs_test, threshold)
    metrics_cal_f1 = _metrics(y_cal, probs_cal, threshold_f1)
    metrics_cal_recall = _metrics(y_cal, probs_cal, threshold_recall)
    metrics_test_f1 = _metrics(y_test, probs_test, threshold_f1)
    metrics_test_recall = _metrics(y_test, probs_test, threshold_recall)

    operating_points = {
        "costo": {
            "threshold": threshold,
            "costo_fn_fp": DEFAULT_COST_RATIO,
            "calibration": metrics_cal,
            "test": metrics_test,
        },
        "f1": {
            "threshold": threshold_f1,
            "calibration": metrics_cal_f1,
            "test": metrics_test_f1,
        },
        "recall80": {
            "threshold": threshold_recall,
            "calibration": metrics_cal_recall,
            "test": metrics_test_recall,
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "threshold": threshold,
        "threshold_f1": threshold_f1,
        "threshold_recall80": threshold_recall,
        "cost_ratio": DEFAULT_COST_RATIO,
        "best_c": best_c,
        "operating_points": operating_points,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": {
            "input": X_test.iloc[0].to_dict(),
            "proba": model.predict_proba(X_test.iloc[[0]])[0].tolist(),
        },
    }
    joblib.dump(artifact, MODEL_PATH)

    results = {
        "best_c": best_c,
        "cv_best_auc_pr": float(search.best_score_),
        "threshold_cost": threshold,
        "threshold_f1": threshold_f1,
        "threshold_recall80": threshold_recall,
        "cost_ratio": DEFAULT_COST_RATIO,
        "cost_table": cost_rows,
        "threshold_table": rows,
        "calibration": metrics_cal,
        "test": metrics_test,
        "operating_points": operating_points,
    }
    METRICS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("=== MODELO FINAL — ENTREGA 3 ===")
    print(f"Mejor C: {best_c} | CV AUC-PR: {search.best_score_:.4f}")
    print(
        f"Umbral costo (FN/FP={DEFAULT_COST_RATIO}): {threshold:.2f} | "
        f"Recall={metrics_cal['Recall']:.4f} | Precision={metrics_cal['Precision']:.4f} | "
        f"F1={metrics_cal['F1']:.4f}"
    )
    print(
        f"Umbral F1: {threshold_f1:.2f} | F1={metrics_cal_f1['F1']:.4f} | "
        f"Recall={metrics_cal_f1['Recall']:.4f} | Precision={metrics_cal_f1['Precision']:.4f}"
    )
    print(
        f"Umbral recall>=80%: {threshold_recall:.2f} | Recall={metrics_cal_recall['Recall']:.4f} | "
        f"Precision={metrics_cal_recall['Precision']:.4f} | F1={metrics_cal_recall['F1']:.4f}"
    )
    print(
        f"2021 AUC-PR: {metrics_cal['AUC_PR']:.4f} | ROC-AUC: {metrics_cal['ROC_AUC']:.4f}"
    )
    print(
        f"2022 AUC-PR: {metrics_test['AUC_PR']:.4f} | ROC-AUC: {metrics_test['ROC_AUC']:.4f} | "
        f"F1@{threshold:.2f}: {metrics_test['F1']:.4f} | Recall: {metrics_test['Recall']:.4f}"
    )
    print(f"Artefacto: {MODEL_PATH}")
    print(f"Métricas: {METRICS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo final de la Entrega 3")
    parser.add_argument("--smoke", action="store_true", help="Verificación rápida")
    parser.add_argument("--data", default=None, help="Ruta alternativa al CSV")
    args = parser.parse_args()

    if args.smoke:
        smoke(args.data)
    else:
        full(args.data)


if __name__ == "__main__":
    main()
