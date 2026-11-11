"""B4 — Calibración II y selección de la estrategia final.

Compara:
1. Estático (modelo 2020 crudo).
2. Platt de origen (2020 OOF, sin etiquetas de despliegue).
3. Platt ajustado con 2021.
4. Isotónica ajustada con 2021.
5. Reentrenamiento 2020 + 80 % de 2021, calibrado con el 20 % holdout de 2021.

La evaluación final es 2022. El umbral de decisión para probabilidades
calibradas es el bayesiano t* = 1 / (1 + FN/FP).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import (  # noqa: E402
    apply_platt,
    calibration_metrics,
    reliability_curve,
)
from src.config import (  # noqa: E402
    ARTIFACTS_DIR,
    DEFAULT_COST_RATIO,
    FIGURES_DIR,
    MODEL_PATH,
    SEED,
)
from src.data import load_dataset, split_waves, X_y  # noqa: E402
from src.pipeline import build_model  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

BAYES_THRESHOLD = 1.0 / (1.0 + DEFAULT_COST_RATIO)
EPS = 1e-6

COLORS = {
    "static": "#6c757d",
    "platt_source": "#1d3557",
    "platt_target": "#457b9d",
    "isotonic_target": "#2a9d8f",
    "retrain_platt": "#e63946",
    "retrain_isotonic": "#f4a261",
}
LABELS = {
    "static": "Estático",
    "platt_source": "Platt origen (2020)",
    "platt_target": "Platt 2021",
    "isotonic_target": "Isotónica 2021",
    "retrain_platt": "Reentreno + Platt",
    "retrain_isotonic": "Reentreno + isotónica",
}


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


def cost_threshold(y: np.ndarray, p: np.ndarray, ratio: int = DEFAULT_COST_RATIO) -> float:
    best_t, best_cost = 0.5, None
    for t in np.arange(0.01, 1.0, 0.01):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        cost = fn * ratio + fp
        if best_cost is None or cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t


def plot_reliability_2022(curves: dict, metrics: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=300)

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="#adb5bd", lw=1.2, label="Perfecta")
    for name, rows in curves.items():
        ax.plot(
            [r["pred"] for r in rows],
            [r["obs"] for r in rows],
            marker="o",
            ms=3.5,
            lw=1.6,
            color=COLORS[name],
            label=f"{LABELS[name]} (ECE={metrics[name]['ECE']:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probabilidad predicha", fontsize=9.5)
    ax.set_ylabel("Frecuencia observada", fontsize=9.5)
    ax.set_title("Rango completo — 2022", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=True, fontsize=6.8, loc="upper left")

    ax = axes[1]
    lim = 0.20
    ax.plot([0, lim], [0, lim], "--", color="#adb5bd", lw=1.2, label="Perfecta")
    for name, rows in curves.items():
        if name == "static":
            continue
        ax.plot(
            [r["pred"] for r in rows],
            [r["obs"] for r in rows],
            marker="o",
            ms=3.5,
            lw=1.6,
            color=COLORS[name],
            label=f"{LABELS[name]} (ECE={metrics[name]['ECE']:.3f})",
        )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Probabilidad predicha", fontsize=9.5)
    ax.set_ylabel("Frecuencia observada", fontsize=9.5)
    ax.set_title(f"Zoom 0–{lim:.2f} — 2022", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=True, fontsize=6.8, loc="upper left")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_comparison(results: dict, path: Path) -> None:
    names = list(results.keys())
    brier = [results[n]["calibration_2022"]["Brier"] for n in names]
    f1 = [results[n]["decision_2022"]["F1"] for n in names]
    recall = [results[n]["decision_2022"]["Recall"] for n in names]
    precision = [results[n]["decision_2022"]["Precision"] for n in names]

    x = np.arange(len(names))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=300)

    ax = axes[0]
    ax.bar(x, brier, color=[COLORS[n] for n in names])
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in names], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Brier score (menor es mejor)", fontsize=9)
    ax.set_title("Calibración en 2022", fontsize=10, fontweight="bold")
    for i, v in enumerate(brier):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax = axes[1]
    ax.bar(x - width, recall, width, label="Recall", color="#1d3557")
    ax.bar(x, precision, width, label="Precision", color="#2a9d8f")
    ax.bar(x + width, f1, width, label="F1", color="#e63946")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in names], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Métrica", fontsize=9)
    ax.set_title(f"Decisión en 2022 (t*={BAYES_THRESHOLD:.3f})", fontsize=10, fontweight="bold")
    ax.legend(frameon=True, fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(MODEL_PATH)
    static_model = artifact["model"]
    waves = split_waves(load_dataset())
    X20, y20 = X_y(waves["2020"])
    X21, y21 = X_y(waves["2021"])
    X22, y22 = X_y(waves["2022"])
    y22_np = y22.to_numpy()

    b3 = json.loads((ARTIFACTS_DIR / "calibration_b3.json").read_text())

    def source_platt(p: np.ndarray) -> np.ndarray:
        z = b3["platt_coef"] * np.log(
            np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))
        ) + b3["platt_intercept"]
        return 1.0 / (1.0 + np.exp(-z))

    p21_raw = static_model.predict_proba(X21)[:, 1]
    p22_raw = static_model.predict_proba(X22)[:, 1]

    # Platt e isotónica ajustadas con 2021.
    platt_target = LogisticRegression().fit(
        np.log(np.clip(p21_raw, EPS, 1 - EPS) / (1 - np.clip(p21_raw, EPS, 1 - EPS))).reshape(-1, 1),
        y21,
    )
    p21_platt = apply_platt(platt_target, p21_raw)
    p22_platt = apply_platt(platt_target, p22_raw)

    iso_target = IsotonicRegression(out_of_bounds="clip").fit(p21_raw, y21)
    p21_iso = iso_target.predict(p21_raw)
    p22_iso = iso_target.predict(p22_raw)

    # Reentrenamiento: 2020 + 80 % de 2021; calibración con el 20 % holdout.
    X21_tr, X21_cal, y21_tr, y21_cal = train_test_split(
        X21, y21, test_size=0.20, stratify=y21, random_state=SEED
    )
    retrain_model = build_model()
    retrain_model.fit(
        pd.concat([X20, X21_tr], ignore_index=True),
        pd.concat([y20, y21_tr], ignore_index=True),
    )
    p21_cal_raw = retrain_model.predict_proba(X21_cal)[:, 1]
    p22_retr_raw = retrain_model.predict_proba(X22)[:, 1]

    platt_retr = LogisticRegression().fit(
        np.log(np.clip(p21_cal_raw, EPS, 1 - EPS) / (1 - np.clip(p21_cal_raw, EPS, 1 - EPS))).reshape(-1, 1),
        y21_cal,
    )
    p22_retr_platt = apply_platt(platt_retr, p22_retr_raw)

    iso_retr = IsotonicRegression(out_of_bounds="clip").fit(p21_cal_raw, y21_cal)
    p22_retr_iso = iso_retr.predict(p22_retr_raw)

    # Estrategias y umbrales.
    strategies = {
        "static": {
            "p22": p22_raw,
            "threshold": float(artifact["threshold"]),
            "calibrated": False,
        },
        "platt_source": {
            "p22": source_platt(p22_raw),
            "threshold": BAYES_THRESHOLD,
            "calibrated": True,
        },
        "platt_target": {
            "p22": p22_platt,
            "threshold": BAYES_THRESHOLD,
            "calibrated": True,
        },
        "isotonic_target": {
            "p22": p22_iso,
            "threshold": BAYES_THRESHOLD,
            "calibrated": True,
        },
        "retrain_platt": {
            "p22": p22_retr_platt,
            "threshold": BAYES_THRESHOLD,
            "calibrated": True,
        },
        "retrain_isotonic": {
            "p22": p22_retr_iso,
            "threshold": BAYES_THRESHOLD,
            "calibrated": True,
        },
    }

    results = {}
    curves = {}
    for name, cfg in strategies.items():
        p22 = cfg["p22"]
        t = cfg["threshold"]
        results[name] = {
            "threshold": t,
            "calibrated": cfg["calibrated"],
            "calibration_2022": calibration_metrics(y22_np, p22),
            "decision_2022": decision_metrics(y22_np, p22, t),
            "curve_2022": reliability_curve(y22_np, p22),
        }
        curves[name] = results[name]["curve_2022"]

    # Diagnóstico: umbral empírico de costo en 2021 (solo para estrategias con 2021).
    diagnostics = {}
    p21_map = {
        "static": p21_raw,
        "platt_source": source_platt(p21_raw),
        "platt_target": p21_platt,
        "isotonic_target": p21_iso,
    }
    for name, p21 in p21_map.items():
        t21 = cost_threshold(y21.to_numpy(), p21)
        p22 = strategies[name]["p22"]
        diagnostics[name] = {
            "threshold_2021": t21,
            "decision_2022_at_threshold_2021": decision_metrics(y22_np, p22, t21),
        }

    out = {
        "bayes_threshold": BAYES_THRESHOLD,
        "cost_ratio": DEFAULT_COST_RATIO,
        "true_prevalence_2022": float(y22_np.mean()),
        "strategies": results,
        "diagnostics": diagnostics,
        "retrain_n_train": int(len(X20) + len(X21_tr)),
        "retrain_n_cal": int(len(X21_cal)),
    }
    (ARTIFACTS_DIR / "calibration_b4.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )

    plot_reliability_2022(curves, {n: r["calibration_2022"] for n, r in results.items()}, FIGURES_DIR / "calibracion_final_2022.png")
    plot_comparison(results, FIGURES_DIR / "comparacion_b4.png")

    print("=== B4 — CALIBRACIÓN II ===")
    print(f"Umbral bayesiano FN/FP={DEFAULT_COST_RATIO}: {BAYES_THRESHOLD:.4f}")
    print(f"Prevalencia 2022: {y22_np.mean():.5f}\n")
    for name, r in results.items():
        c = r["calibration_2022"]
        d = r["decision_2022"]
        print(
            f"{name:18s} Brier={c['Brier']:.5f} ECE={c['ECE']:.5f} "
            f"LogLoss={c['log_loss']:.5f} AUC-PR={c['AUC_PR']:.4f} "
            f"| t={d['threshold']:.3f} R={d['Recall']:.4f} P={d['Precision']:.4f} F1={d['F1']:.4f}"
        )
    print("\nDiagnóstico con umbral empírico de 2021:")
    for name, d in diagnostics.items():
        m = d["decision_2022_at_threshold_2021"]
        print(
            f"{name:18s} t21={d['threshold_2021']:.2f} "
            f"R={m['Recall']:.4f} P={m['Precision']:.4f} F1={m['F1']:.4f}"
        )
    print(f"\nResultados: {ARTIFACTS_DIR / 'calibration_b4.json'}")
    print(f"Figuras: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
