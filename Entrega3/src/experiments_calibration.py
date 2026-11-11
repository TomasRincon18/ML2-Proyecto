"""B3 — Calibración I: estático, Platt de origen y EM de prevalencia.

Uso:
    python src/experiments_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import (  # noqa: E402
    apply_platt,
    calibration_metrics,
    em_prior_shift,
    fit_source_platt,
    prior_correct,
    reliability_curve,
)
from src.config import ARTIFACTS_DIR, FIGURES_DIR, MODEL_PATH, SEED  # noqa: E402
from src.data import load_dataset, split_waves, X_y  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

COLORS = {
    "static": "#6c757d",
    "platt": "#1d3557",
    "em": "#e63946",
    "oracle": "#2a9d8f",
}
LABELS = {
    "static": "Estático (modelo crudo)",
    "platt": "Platt de origen (2020)",
    "em": "EM de prevalencia",
    "oracle": "Oracle (prevalencia real)",
}


def plot_reliability(rows_by_method: dict, metrics: dict, cohort: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), dpi=300)

    # Panel izquierdo: rango completo, para mostrar la descalibración del crudo.
    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle="--", color="#adb5bd", lw=1.2, label="Perfecta")
    for method, rows in rows_by_method.items():
        pred = [r["pred"] for r in rows]
        obs = [r["obs"] for r in rows]
        label = f"{LABELS[method]} (ECE={metrics[method]['ECE']:.3f})"
        ax.plot(pred, obs, marker="o", ms=4, lw=1.8, color=COLORS[method], label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probabilidad predicha", fontsize=9.5)
    ax.set_ylabel("Frecuencia observada", fontsize=9.5)
    ax.set_title(f"Rango completo — {cohort}", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=True, fontsize=7.2, loc="upper left")

    # Panel derecho: zoom en la zona clínicamente relevante.
    ax = axes[1]
    lim = 0.20
    ax.plot([0, lim], [0, lim], linestyle="--", color="#adb5bd", lw=1.2, label="Perfecta")
    for method in ["platt", "em", "oracle"]:
        rows = rows_by_method[method]
        pred = [r["pred"] for r in rows]
        obs = [r["obs"] for r in rows]
        label = f"{LABELS[method]} (ECE={metrics[method]['ECE']:.3f})"
        ax.plot(pred, obs, marker="o", ms=4, lw=1.8, color=COLORS[method], label=label)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Probabilidad predicha", fontsize=9.5)
    ax.set_ylabel("Frecuencia observada", fontsize=9.5)
    ax.set_title(f"Zoom 0–{lim:.2f} — {cohort}", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=True, fontsize=7.2, loc="upper left")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_summary(results: dict, path: Path) -> None:
    cohorts = ["2021", "2022"]
    methods = ["static", "platt", "em", "oracle"]
    x = np.arange(len(cohorts))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)

    ax = axes[0]
    for i, method in enumerate(methods):
        vals = [results[c][method]["Brier"] for c in cohorts]
        ax.bar(x + (i - 1.5) * width, vals, width, color=COLORS[method], label=LABELS[method])
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("Brier score (menor es mejor)", fontsize=9)
    ax.set_title("Brier por método y cohorte", fontsize=10, fontweight="bold")
    ax.legend(frameon=True, fontsize=7.5)
    ax.set_ylim(0, max(results[c]["static"]["Brier"] for c in cohorts) * 1.2)

    ax = axes[1]
    true_vals = [results[c]["true_prevalence"] for c in cohorts]
    platt_vals = [results[c]["platt"]["mean_p"] for c in cohorts]
    em_vals = [results[c]["em"]["pi_hat"] for c in cohorts]
    ax.bar(x - width, true_vals, width, color="#495057", label="Prevalencia real")
    ax.bar(x, platt_vals, width, color=COLORS["platt"], label="Media Platt")
    ax.bar(x + width, em_vals, width, color=COLORS["em"], label="Prevalencia EM")
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("Proporción", fontsize=9)
    ax.set_title("Prevalencia real vs estimada", fontsize=10, fontweight="bold")
    ax.legend(frameon=True, fontsize=7.5)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    waves = split_waves(load_dataset())

    X20, y20 = X_y(waves["2020"])
    pi_source = float(y20.mean())
    platt, oof = fit_source_platt(model, X20, y20)

    results = {
        "source_prevalence": pi_source,
        "platt_coef": float(platt.coef_[0, 0]),
        "platt_intercept": float(platt.intercept_[0]),
        "methods": {},
    }

    reliability_rows = {}
    for cohort in ["2021", "2022"]:
        X, y = X_y(waves[cohort])
        y_np = y.to_numpy()
        p_raw = model.predict_proba(X)[:, 1]
        p_platt = apply_platt(platt, p_raw)
        pi_hat, p_em, iterations, converged = em_prior_shift(p_platt, pi_source)
        p_oracle = prior_correct(p_platt, pi_source, float(y_np.mean()))

        methods = {
            "static": p_raw,
            "platt": p_platt,
            "em": p_em,
            "oracle": p_oracle,
        }

        cohort_results = {"true_prevalence": float(y_np.mean())}
        for name, probs in methods.items():
            cohort_results[name] = calibration_metrics(y_np, probs)
            reliability_rows.setdefault(cohort, {})[name] = reliability_curve(y_np, probs)

        cohort_results["em"]["pi_hat"] = float(pi_hat)
        cohort_results["em"]["iterations"] = int(iterations)
        cohort_results["em"]["converged"] = bool(converged)
        cohort_results["platt"]["oof_brier"] = float(
            calibration_metrics(y20.to_numpy(), oof)["Brier"]
        )
        results["methods"][cohort] = cohort_results

        plot_reliability(
            reliability_rows[cohort],
            cohort_results,
            cohort,
            FIGURES_DIR / f"calibracion_{cohort}.png",
        )

    plot_summary(results["methods"], FIGURES_DIR / "brier_prevalencia.png")

    out = ARTIFACTS_DIR / "calibration_b3.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("=== B3 — CALIBRACIÓN I ===")
    print(f"Prevalencia fuente 2020: {pi_source:.5f}")
    print(f"Platt: coef={platt.coef_[0,0]:.4f} intercept={platt.intercept_[0]:.4f}")
    for cohort in ["2021", "2022"]:
        r = results["methods"][cohort]
        print(f"\n{cohort} — prevalencia real {r['true_prevalence']:.5f}")
        for name in ["static", "platt", "em", "oracle"]:
            m = r[name]
            extra = f" | pi_hat={m['pi_hat']:.5f} iter={m['iterations']}" if name == "em" else ""
            print(
                f"  {name:7s}: media={m['mean_p']:.5f} Brier={m['Brier']:.5f} "
                f"logloss={m['log_loss']:.5f} ECE={m['ECE']:.5f} "
                f"AUC-PR={m['AUC_PR']:.4f}{extra}"
            )
    print(f"\nResultados: {out}")
    print(f"Figuras: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
