"""Compara Platt vs calibración beta sobre el modelo final de B4.

Uso:
    python src/experiments_beta_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import (  # noqa: E402
    apply_platt,
    calibration_metrics,
    reliability_curve,
)
from src.config import ARTIFACTS_DIR, FIGURES_DIR, SEED  # noqa: E402
from src.data import load_dataset, split_waves, X_y  # noqa: E402
from src.final_model import BetaCalibrator  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

EPS = 1e-6
BAYES_THRESHOLD = 1.0 / 11.0
FINAL_PATH = ARTIFACTS_DIR / "model_final.joblib"


def decision_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Recall": float(recall),
        "Precision": float(precision),
        "F1": float(f1),
        "positivos": int(pred.sum()),
    }


def plot_curve(curves: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), dpi=300)
    colors = {"platt": "#1d3557", "beta": "#e63946"}

    for ax, lim in zip(axes, [1.0, 0.20]):
        ax.plot([0, lim], [0, lim], "--", color="#adb5bd", lw=1.2, label="Perfecta")
        for name, rows in curves.items():
            ax.plot(
                [r["pred"] for r in rows],
                [r["obs"] for r in rows],
                marker="o",
                ms=4,
                lw=1.8,
                color=colors[name],
                label=name.capitalize(),
            )
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Probabilidad predicha", fontsize=9.5)
        ax.set_ylabel("Frecuencia observada", fontsize=9.5)
        ax.set_title(
            "Rango completo — 2022" if lim == 1.0 else "Zoom 0–0.20 — 2022",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.legend(frameon=True, fontsize=8.5, loc="upper left")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    artifact = joblib.load(FINAL_PATH)
    final_model = artifact["final_model"]
    retrained = final_model.model

    waves = split_waves(load_dataset())
    X21, y21 = X_y(waves["2021"])
    X22, y22 = X_y(waves["2022"])
    y21_cal = None
    _, X21_cal, _, y21_cal = train_test_split(
        X21, y21, test_size=0.20, stratify=y21, random_state=SEED
    )

    p_cal_raw = retrained.predict_proba(X21_cal)[:, 1]
    p22_raw = retrained.predict_proba(X22)[:, 1]
    y22_np = y22.to_numpy()

    # Platt sobre el mismo holdout.
    z_cal = np.log(np.clip(p_cal_raw, EPS, 1 - EPS) / (1 - np.clip(p_cal_raw, EPS, 1 - EPS)))
    platt = LogisticRegression().fit(z_cal.reshape(-1, 1), y21_cal)
    p22_platt = apply_platt(platt, p22_raw)

    # Beta calibration.
    z_beta = np.column_stack(
        [
            np.log(np.clip(p_cal_raw, EPS, 1 - EPS)),
            -np.log(1 - np.clip(p_cal_raw, EPS, 1 - EPS)),
        ]
    )
    beta = LogisticRegression(C=np.inf, max_iter=1000).fit(z_beta, y21_cal)
    a, b = float(beta.coef_[0, 0]), float(beta.coef_[0, 1])
    c = float(beta.intercept_[0])
    beta_cal = BetaCalibrator(a=a, b=b, c=c)
    p22_beta = beta_cal.predict_proba(p22_raw)

    metrics = {
        "platt": calibration_metrics(y22_np, p22_platt),
        "beta": calibration_metrics(y22_np, p22_beta),
    }
    decisions = {
        "platt": decision_metrics(y22_np, p22_platt, BAYES_THRESHOLD),
        "beta": decision_metrics(y22_np, p22_beta, BAYES_THRESHOLD),
    }
    curves = {
        "platt": reliability_curve(y22_np, p22_platt),
        "beta": reliability_curve(y22_np, p22_beta),
    }

    better = (
        a > 0
        and b > 0
        and metrics["beta"]["Brier"] < metrics["platt"]["Brier"]
        and metrics["beta"]["ECE"] < metrics["platt"]["ECE"]
        and decisions["beta"]["F1"] >= decisions["platt"]["F1"] - 1e-4
    )

    out = {
        "beta_params": {"a": a, "b": b, "c": c},
        "monotonic": bool(a > 0 and b > 0),
        "metrics_2022": metrics,
        "decision_2022": decisions,
        "platt_params": {
            "coef": float(platt.coef_[0, 0]),
            "intercept": float(platt.intercept_[0]),
        },
        "top_decile": {
            "platt": curves["platt"][-1],
            "beta": curves["beta"][-1],
        },
        "adopt_beta": bool(better),
    }
    (ARTIFACTS_DIR / "beta_calibration.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    plot_curve(curves, FIGURES_DIR / "calibracion_beta_2022.png")

    print("=== BETA CALIBRATION vs PLATT ===")
    print(f"Beta: a={a:.4f} b={b:.4f} c={c:.4f} | monotónica={out['monotonic']}")
    for name in ["platt", "beta"]:
        m = metrics[name]
        d = decisions[name]
        print(
            f"{name:5s}: Brier={m['Brier']:.5f} ECE={m['ECE']:.5f} "
            f"LogLoss={m['log_loss']:.5f} AUC-PR={m['AUC_PR']:.4f} "
            f"| R={d['Recall']:.4f} P={d['Precision']:.4f} F1={d['F1']:.4f}"
        )
    print("Decil superior (predicho / observado):")
    print(f"  Platt: {curves['platt'][-1]['pred']:.4f} / {curves['platt'][-1]['obs']:.4f}")
    print(f"  Beta : {curves['beta'][-1]['pred']:.4f} / {curves['beta'][-1]['obs']:.4f}")
    print(f"\n¿Adoptar beta?: {better}")
    print(f"Resultados: {ARTIFACTS_DIR / 'beta_calibration.json'}")
    print(f"Figura: {FIGURES_DIR / 'calibracion_beta_2022.png'}")


if __name__ == "__main__":
    main()
