"""Genera figuras específicas para el deck, con tipografía y estilo de diapositiva."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.config import SEED  # noqa: E402
from src.data import load_dataset, split_waves, X_y  # noqa: E402

AZUL = "#0B1D3A"
AZUL_MEDIO = "#1D3557"
VERDE = "#2A9D8F"
ROJO = "#E63946"
GRIS = "#6C757D"
ARENA = "#F1F5F9"
DORADO = "#F4A261"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": "#CED4DA",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#343A40",
        "xtick.color": "#495057",
        "ytick.color": "#495057",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)


def calibration_slide(b4: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=220)
    series = [
        ("static", "Modelo crudo", GRIS, "--", "o"),
        ("retrain_platt", "Modelo final", AZUL_MEDIO, "-", ""),
    ]
    for ax, lim, title in zip(axes, [1.0, 0.20], ["Rango completo", "Zoom 0 a 0,20"]):
        ax.plot([0, lim], [0, lim], color="#ADB5BD", lw=1.0, ls=":", label="Perfecta")
        for key, label, color, ls, marker in series:
            rows = b4["strategies"][key]["curve_2022"]
            ax.plot(
                [r["pred"] for r in rows],
                [r["obs"] for r in rows],
                color=color,
                lw=2.0,
                ls=ls,
                marker=marker,
                ms=4,
                label=label,
            )
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Probabilidad predicha", fontsize=10)
        ax.set_ylabel("Frecuencia observada", fontsize=10)
        ax.set_title(title, fontsize=11, color=AZUL, fontweight="bold", loc="left")
        clean_axes(ax)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def shap_slide(b6: dict, out: Path):
    keys = ["edad_anios", "sexo", "ciudad_top", "fuente_tipo_contagio"]
    labels = ["Edad", "Sexo", "Ciudad", "Fuente"]
    data = b6["shap_bootstrap"]
    vals = [data[k]["mean_abs_shap"] for k in keys]
    low = [data[k]["mean_abs_shap"] - data[k]["ci_low"] for k in keys]
    high = [data[k]["ci_high"] - data[k]["mean_abs_shap"] for k in keys]
    order = np.argsort(vals)

    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=220)
    ax.barh(
        [labels[i] for i in order],
        [vals[i] for i in order],
        xerr=[[low[i] for i in order], [high[i] for i in order]],
        color=VERDE,
        edgecolor=AZUL_MEDIO,
        height=0.55,
        capsize=3,
    )
    for y, i in enumerate(order):
        ax.text(vals[i] + high[i] + 0.04, y, f"{vals[i]:.2f}", va="center", fontsize=9, color=AZUL)
    ax.set_xlabel("Importancia media |SHAP| (log-odds)", fontsize=10)
    ax.set_xlim(0, max(vals) * 1.18)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def errors_slide(b6: dict, out: Path):
    groups = list(b6["subgroups"]["edad"].keys())
    labels = [g.replace("–", "-") for g in groups]
    metrics = ["Recall", "Precision", "F1"]
    colors = [AZUL_MEDIO, VERDE, ROJO]
    x = np.arange(len(groups))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=220)
    for i, (m, color) in enumerate(zip(metrics, colors)):
        vals = [
            (b6["subgroups"]["edad"][g][m] or 0.0) for g in groups
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=m, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{label}\n(n={b6['subgroups']['edad'][g]['n']:,})" for label, g in zip(labels, groups)],
        fontsize=8,
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Métrica en 2022", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def decision_slide(b6: dict, out: Path):
    rows = b6["decision_curve"]["rows"]
    t = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=220)
    ax.plot(t, [r["nb_model"] for r in rows], color=AZUL_MEDIO, lw=2.4, label="Modelo final")
    ax.plot(t, [r["nb_all"] for r in rows], color=ROJO, lw=1.6, ls="--", label="Tratar a todos")
    ax.axhline(0, color=GRIS, lw=1.0, ls=":", label="No tratar")
    ax.axvline(1 / 11, color=VERDE, lw=1.2, ls="--", label="Umbral 0,091")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.01, max(r["nb_model"] for r in rows) * 1.25)
    ax.set_xlabel("Umbral de probabilidad para tratar", fontsize=10)
    ax.set_ylabel("Beneficio neto", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def pdp_slide(out: Path):
    artifact = joblib.load(BASE / "artifacts" / "model_final.joblib")
    final_model = artifact["final_model"]
    waves = split_waves(load_dataset())
    X22, _ = X_y(waves["2022"])
    sample = X22.sample(n=300, random_state=SEED + 1).copy()

    ages = np.arange(0, 101, 1)
    mean, p10, p90 = [], [], []
    for age in ages:
        sample["edad_anios"] = age
        p = final_model.predict_proba(sample)[:, 1]
        mean.append(float(np.mean(p)))
        p10.append(float(np.percentile(p, 10)))
        p90.append(float(np.percentile(p, 90)))

    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=220)
    ax.plot(ages, mean, color=AZUL_MEDIO, lw=2.6, label="Riesgo medio")
    ax.fill_between(ages, p10, p90, color=AZUL_MEDIO, alpha=0.14, label="Banda P10-P90")
    ax.axhline(final_model.threshold, color=ROJO, lw=1.2, ls="--", label="Umbral 0,091")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Edad (años)", fontsize=10)
    ax.set_ylabel("Probabilidad calibrada", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def monitor_slide(b6: dict, out: Path):
    real = {r["cohorte"]: r for r in b6["monitor"]["real"]}
    sim = {r["cohorte"]: r for r in b6["monitor"]["simulaciones"]}
    labels = ["2020 test", "2021", "2022", "Sim 10 %", "Sim 30 %"]
    ocsvm = [
        real["2020 test (out-of-sample)"]["OCSVM"],
        real["2021 Delta"]["OCSVM"],
        real["2022 Ómicron"]["OCSVM"],
        sim["10 % ruido"]["OCSVM"],
        sim["30 % ruido"]["OCSVM"],
    ]
    iforest = [
        real["2020 test (out-of-sample)"]["IForest"],
        real["2021 Delta"]["IForest"],
        real["2022 Ómicron"]["IForest"],
        sim["10 % ruido"]["IForest"],
        sim["30 % ruido"]["IForest"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=220)
    ax.bar(x - width / 2, ocsvm, width, color=AZUL_MEDIO, label="OCSVM")
    ax.bar(x + width / 2, iforest, width, color=ROJO, label="Isolation Forest")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Proporción de inliers", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2)
    for i, (a, b) in enumerate(zip(ocsvm, iforest)):
        ax.text(i - width / 2, a + 0.02, f"{a*100:.0f}", ha="center", fontsize=7, color=AZUL)
        ax.text(i + width / 2, b + 0.02, f"{b*100:.0f}", ha="center", fontsize=7, color=ROJO)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def app_crop(out: Path):
    src = BASE / "docs" / "evidencias" / "streamlit_prediccion_alto.png"
    image = Image.open(src)
    w, h = image.size
    # La captura tiene 1,5x de escala; recorta la barra lateral y el pie vacío.
    left = int(w * 0.22)
    top = 0
    right = w
    bottom = int(h * 0.73)
    image.crop((left, top, right, bottom)).save(out)


def main():
    out = BASE / "slides" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    b4 = json.loads((BASE / "artifacts" / "calibration_b4.json").read_text())
    b6 = json.loads((BASE / "artifacts" / "b6_results.json").read_text())

    calibration_slide(b4, out / "calibracion_slide.png")
    shap_slide(b6, out / "shap_slide.png")
    errors_slide(b6, out / "errores_slide.png")
    decision_slide(b6, out / "decision_slide.png")
    pdp_slide(out / "pdp_slide.png")
    monitor_slide(b6, out / "monitor_slide.png")
    app_crop(out / "app_slide.png")
    print("figuras de diapositiva generadas en", out)


if __name__ == "__main__":
    main()
