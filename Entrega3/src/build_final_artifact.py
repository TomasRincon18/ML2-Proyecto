"""Construye el artefacto final elegido en B4: reentreno 2020+2021 + Platt.

Uso:
    python src/build_final_artifact.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import apply_platt  # noqa: E402
from src.config import (  # noqa: E402
    ARTIFACTS_DIR,
    DEFAULT_COST_RATIO,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    SEED,
    TARGET,
)
from src.data import load_dataset, split_waves, X_y  # noqa: E402
from src.final_model import FinalModel, PlattCalibrator  # noqa: E402
from src.pipeline import build_model  # noqa: E402

EPS = 1e-6
BAYES_THRESHOLD = 1.0 / (1.0 + DEFAULT_COST_RATIO)
FINAL_PATH = ARTIFACTS_DIR / "model_final.joblib"


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    waves = split_waves(load_dataset())
    X20, y20 = X_y(waves["2020"])
    X21, y21 = X_y(waves["2021"])
    X22, y22 = X_y(waves["2022"])

    # Misma partición de B4: 80 % entrena, 20 % calibra.
    X21_tr, X21_cal, y21_tr, y21_cal = train_test_split(
        X21, y21, test_size=0.20, stratify=y21, random_state=SEED
    )

    model = build_model()
    model.fit(
        pd.concat([X20, X21_tr], ignore_index=True),
        pd.concat([y20, y21_tr], ignore_index=True),
    )

    p_cal_raw = model.predict_proba(X21_cal)[:, 1]
    platt = LogisticRegression().fit(
        np.log(np.clip(p_cal_raw, EPS, 1 - EPS) / (1 - np.clip(p_cal_raw, EPS, 1 - EPS))).reshape(-1, 1),
        y21_cal,
    )
    calibrator = PlattCalibrator(
        coef=float(platt.coef_[0, 0]),
        intercept=float(platt.intercept_[0]),
    )

    final_model = FinalModel(
        model=model,
        calibrator=calibrator,
        threshold=BAYES_THRESHOLD,
        features=FEATURES,
    )

    # Inspección para la app: background SHAP, coeficientes, referencias y
    # categorías. Así la app explica sin cargar el dataset completo.
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    train_df = pd.concat([X20, X21_tr], ignore_index=True)
    X_trans = np.asarray(prep.transform(train_df), dtype=float)
    background_mean = X_trans.mean(axis=0)
    feature_names = list(prep.get_feature_names_out())

    def group_indices(names):
        groups = {g: [] for g in FEATURES}
        for j, n in enumerate(names):
            if n.startswith("num__edad_anios"):
                groups["edad_anios"].append(j)
            elif n.startswith("cat__sexo_"):
                groups["sexo"].append(j)
            elif n.startswith("cat__ciudad_top_"):
                groups["ciudad_top"].append(j)
            elif n.startswith("cat__fuente_tipo_contagio_"):
                groups["fuente_tipo_contagio"].append(j)
        return groups

    references = {
        "edad_anios": float(np.median(train_df["edad_anios"])),
        "sexo": "F",
        "ciudad_top": "BOGOTA",
        "fuente_tipo_contagio": "COMUNITARIA",
    }
    categories = {
        "sexo": sorted(train_df["sexo"].unique().tolist()),
        "ciudad_top": sorted(train_df["ciudad_top"].unique().tolist()),
        "fuente_tipo_contagio": sorted(train_df["fuente_tipo_contagio"].unique().tolist()),
        "edad_min": int(np.floor(train_df["edad_anios"].min())),
        "edad_max": int(np.ceil(train_df["edad_anios"].max())),
        "edad_median": references["edad_anios"],
    }
    shap_meta = {
        "background_mean": background_mean.tolist(),
        "feature_names": feature_names,
        "groups": group_indices(feature_names),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "references": references,
    }
    reference_summary = {
        "n_train": int(len(train_df)),
        "mean_age": float(train_df["edad_anios"].mean()),
        "age_quantiles": {
            str(q): float(np.percentile(train_df["edad_anios"], q))
            for q in [5, 25, 50, 75, 95]
        },
        "sexo_props": train_df["sexo"].value_counts(normalize=True).to_dict(),
        "ciudad_props": train_df["ciudad_top"].value_counts(normalize=True).to_dict(),
        "fuente_props": train_df["fuente_tipo_contagio"].value_counts(normalize=True).to_dict(),
    }

    # Métricas de 2022 y verificación de paridad.
    p22 = final_model.predict_proba(X22)[:, 1]
    y22_np = y22.to_numpy()

    b4 = json.loads((ARTIFACTS_DIR / "calibration_b4.json").read_text())
    expected = b4["strategies"]["retrain_platt"]
    metrics_2022 = dict(expected["calibration_2022"])
    metrics_2022["threshold"] = float(BAYES_THRESHOLD)
    metrics_2022["cost_ratio_fn_fp"] = DEFAULT_COST_RATIO
    metrics_2022["decision"] = expected["decision_2022"]

    assert abs(float(average_precision_score(y22_np, p22)) - expected["calibration_2022"]["AUC_PR"]) < 1e-9, "AUC-PR no coincide con B4"
    assert abs(float(brier_score_loss(y22_np, np.clip(p22, EPS, 1 - EPS))) - expected["calibration_2022"]["Brier"]) < 1e-9, "Brier no coincide con B4"

    ref_input = X22.iloc[0]
    ref_proba = final_model.predict_proba(ref_input.to_frame().T)[0].tolist()

    artifact = {
        "final_model": final_model,
        "strategy": "retrain_2020_2021_platt",
        "features": FEATURES,
        "target": TARGET,
        "threshold": BAYES_THRESHOLD,
        "cost_ratio_fn_fp": DEFAULT_COST_RATIO,
        "metrics_2022": metrics_2022,
        "calibrator": {
            "kind": "platt",
            "coef": calibrator.coef,
            "intercept": calibrator.intercept,
        },
        "shap": shap_meta,
        "categories": categories,
        "reference_summary": reference_summary,
        "training": {
            "n_train": int(len(X20) + len(X21_tr)),
            "n_calibration": int(len(X21_cal)),
            "waves_train": ["2020 (original)", "2021 (Delta, 80 %)"],
            "wave_calibration": "2021 (Delta, 20 %)",
            "wave_test": "2022 (Omicron)",
        },
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": {
            "input": ref_input.to_dict(),
            "proba": ref_proba,
        },
    }
    joblib.dump(artifact, FINAL_PATH)

    # Paridad tras recargar.
    loaded = joblib.load(FINAL_PATH)
    fm = loaded["final_model"]
    X_ref = pd.DataFrame([loaded["reference"]["input"]], columns=loaded["features"])
    p_ref = fm.predict_proba(X_ref)[0]
    diff = float(np.max(np.abs(p_ref - np.array(loaded["reference"]["proba"]))))
    assert diff <= 1e-12, f"Fallo de paridad: {diff}"

    print("=== ARTEFACTO FINAL — B4 ===")
    print(f"Estrategia: {artifact['strategy']}")
    print(f"Umbral bayesiano: {BAYES_THRESHOLD:.4f} (FN/FP={DEFAULT_COST_RATIO})")
    print(f"Train: {len(X20):,} (2020) + {len(X21_tr):,} (2021) = {len(X20)+len(X21_tr):,}")
    print(f"Calibración: {len(X21_cal):,} casos de 2021")
    print(
        f"2022 AUC-PR: {metrics_2022['AUC_PR']:.4f} | ROC-AUC: {metrics_2022['ROC_AUC']:.4f} | "
        f"Brier: {metrics_2022['Brier']:.5f} | ECE: {metrics_2022['ECE']:.5f}"
    )
    print(
        f"Decisión @ {BAYES_THRESHOLD:.3f}: Recall={metrics_2022['decision']['Recall']:.4f} | "
        f"Precision={metrics_2022['decision']['Precision']:.4f} | F1={metrics_2022['decision']['F1']:.4f}"
    )
    print(f"Paridad referencia: {diff:.2e}")
    print(f"Artefacto: {FINAL_PATH}")


if __name__ == "__main__":
    main()
