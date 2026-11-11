"""B6 — Estabilidad, errores por subgrupo, curva de decisión y monitor.

Uso:
    python src/experiments_b6.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ARTIFACTS_DIR, FIGURES_DIR, PROJECT_DIR, SEED  # noqa: E402
from src.data import load_dataset, split_waves, X_y  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

EPS = 1e-6
AGE_BINS = [0, 18, 40, 60, 80, 120]
AGE_LABELS = ["0–17", "18–39", "40–59", "60–79", "80+"]


def subgroup_metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    prevalence = float(np.mean(y))
    mean_p = float(np.mean(p))
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return {
        "n": int(n),
        "prevalencia": prevalence,
        "mean_p": mean_p,
        "gap_calibracion": float(mean_p - prevalence),
        "Brier": float(brier_score_loss(y, np.clip(p, EPS, 1 - EPS))),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Recall": float(recall) if recall is not None else None,
        "Precision": float(precision) if precision is not None else None,
        "F1": float(f1) if f1 is not None else None,
        "tasa_marcado": float(np.mean(pred)),
    }


def evaluate_groups(df: pd.DataFrame) -> dict:
    y = df["grave"].to_numpy()
    p = df["p"].to_numpy()
    pred = df["pred"].to_numpy()

    out = {}
    for name, series in {
        "edad": pd.cut(df["edad_anios"], bins=AGE_BINS, labels=AGE_LABELS, right=False),
        "sexo": df["sexo"],
        "fuente_tipo_contagio": df["fuente_tipo_contagio"],
        "ciudad_top": df["ciudad_top"],
    }.items():
        rows = {}
        for group, idx in df.groupby(series, observed=True).groups.items():
            mask = np.zeros(len(df), dtype=bool)
            mask[list(idx)] = True
            rows[str(group)] = subgroup_metrics(y[mask], p[mask], pred[mask])
        out[name] = rows
    return out


def decision_curve(y: np.ndarray, p: np.ndarray) -> dict:
    n = len(y)
    prevalence = float(np.mean(y))
    thresholds = np.arange(0.01, 0.51, 0.01)
    rows = []
    for pt in thresholds:
        pred = p >= pt
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        nb_model = tp / n - (fp / n) * (pt / (1 - pt))
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append(
            {
                "threshold": float(pt),
                "nb_model": float(nb_model),
                "nb_all": float(nb_all),
                "nb_none": 0.0,
            }
        )
    best = max(rows, key=lambda r: r["nb_model"])
    default = min(rows, key=lambda r: abs(r["threshold"] - 1 / 11))
    return {
        "rows": rows,
        "best": best,
        "default_threshold": default,
        "prevalence": prevalence,
    }


def plot_errors_age(df: pd.DataFrame, group_metrics: dict, path: Path) -> None:
    labels = AGE_LABELS
    metrics = ["Recall", "Precision", "F1"]
    colors = ["#1d3557", "#2a9d8f", "#e63946"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=300)
    for i, (m, color) in enumerate(zip(metrics, colors)):
        vals = [
            group_metrics["edad"][label][m] if group_metrics["edad"][label][m] is not None else 0
            for label in labels
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=m, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{label}\n(n={group_metrics['edad'][label]['n']:,})" for label in labels],
        fontsize=8.5,
    )
    ax.set_xlabel("Grupo de edad", fontsize=9.5)
    ax.set_ylabel("Métrica en 2022", fontsize=9.5)
    ax.set_title("Errores por grupo de edad — umbral 0.091", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=True, fontsize=8)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_errors_sex_source(group_metrics: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=300)
    for ax, key, title in zip(axes, ["sexo", "fuente_tipo_contagio"], ["Sexo", "Fuente de contagio"]):
        labels = list(group_metrics[key].keys())
        x = np.arange(len(labels))
        width = 0.25
        for i, (m, color) in enumerate(zip(["Recall", "Precision", "F1"], ["#1d3557", "#2a9d8f", "#e63946"])):
            vals = [
                group_metrics[key][label][m] if group_metrics[key][label][m] is not None else 0
                for label in labels
            ]
            ax.bar(x + (i - 1) * width, vals, width, label=m, color=color)
        tick_labels = []
        for label in labels:
            n = group_metrics[key][label]["n"]
            extra = "" if group_metrics[key][label]["Recall"] is not None else "\nsin positivos"
            tick_labels.append(f"{label}\n(n={n:,}){extra}")
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=7.5)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Métrica", fontsize=9)
        ax.legend(frameon=True, fontsize=7.5)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_decision_curve(dc: dict, path: Path) -> None:
    rows = dc["rows"]
    t = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=300)
    ax.plot(t, [r["nb_model"] for r in rows], color="#1d3557", lw=2.2, label="Modelo final")
    ax.plot(t, [r["nb_all"] for r in rows], color="#e63946", lw=1.6, linestyle="--", label="Tratar a todos")
    ax.axhline(0, color="#6c757d", lw=1.2, linestyle=":", label="No tratar")
    ax.axvline(1 / 11, color="#2a9d8f", lw=1.2, linestyle="--", label="Umbral 0.091")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.01, max(r["nb_model"] for r in rows) * 1.25)
    ax.set_xlabel("Probabilidad umbral para tratar", fontsize=9.5)
    ax.set_ylabel("Beneficio neto", fontsize=9.5)
    ax.set_title("Curva de decisión — 2022", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def shap_bootstrap(path_npz: Path, n_boot: int = 1000) -> dict:
    data = np.load(path_npz, allow_pickle=True)
    sv = data["sv_agg"]
    names = [str(x) for x in data["feature_names"]]
    rng = np.random.default_rng(SEED)
    n = len(sv)
    boot = np.zeros((n_boot, sv.shape[1]), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = np.mean(np.abs(sv[idx]), axis=0)
    out = {}
    for j, name in enumerate(names):
        out[name] = {
            "mean_abs_shap": float(np.mean(np.abs(sv[:, j]))),
            "ci_low": float(np.percentile(boot[:, j], 2.5)),
            "ci_high": float(np.percentile(boot[:, j], 97.5)),
        }
    return out


def plot_shap_bootstrap(boot: dict, path: Path) -> None:
    labels = ["Edad", "Sexo", "Ciudad", "Fuente"]
    keys = ["edad_anios", "sexo", "ciudad_top", "fuente_tipo_contagio"]
    vals = [boot[k]["mean_abs_shap"] for k in keys]
    low = [boot[k]["mean_abs_shap"] - boot[k]["ci_low"] for k in keys]
    high = [boot[k]["ci_high"] - boot[k]["mean_abs_shap"] for k in keys]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
    ax.barh(
        [labels[i] for i in order],
        [vals[i] for i in order],
        xerr=[[low[i] for i in order], [high[i] for i in order]],
        color="#2a9d8f",
        edgecolor="#264653",
        height=0.55,
        capsize=3,
    )
    ax.set_xlabel("Importancia media |SHAP| (log-odds)", fontsize=9.5)
    ax.set_title("Estabilidad de la importancia SHAP — IC 95 %", fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def monitor_phase2(path_json: Path) -> dict:
    data = json.loads(path_json.read_text())
    inliers = data["ocsvm_inliers"]
    real_keys = [
        ("2020 test (out-of-sample)", "2020 Test (out-of-sample, 95k)"),
        ("2021 Delta", "2021 (Delta, 100k)"),
        ("2022 Ómicron", "2022 (Omicron, 100k)"),
    ]
    sim_keys = [
        ("5 % ruido", "Sim 5% ruido"),
        ("10 % ruido", "Sim 10% ruido"),
        ("30 % ruido", "Sim 30% ruido"),
    ]
    real = [
        {
            "cohorte": label,
            "OCSVM": float(inliers[key]["OCSVM_inliers"]),
            "IForest": float(inliers[key]["IForest_inliers"]),
        }
        for label, key in real_keys
    ]
    sim = [
        {
            "cohorte": label,
            "OCSVM": float(inliers[key]["OCSVM_inliers"]),
            "IForest": float(inliers[key]["IForest_inliers"]),
        }
        for label, key in sim_keys
    ]
    return {"real": real, "simulaciones": sim}


def plot_monitor(monitor: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), dpi=300)
    for ax, key, title in zip(axes, ["real", "simulaciones"], ["Cohortes reales", "Simulaciones de drift"]):
        rows = monitor[key]
        labels = [r["cohorte"] for r in rows]
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, [r["OCSVM"] for r in rows], width, color="#1d3557", label="OCSVM")
        ax.bar(x + width / 2, [r["IForest"] for r in rows], width, color="#e63946", label="Isolation Forest")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.16)
        ax.set_ylabel("Proporción de inliers", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        for i, row in enumerate(rows):
            ax.text(i - width / 2, row["OCSVM"] + 0.02, f"{row['OCSVM']*100:.1f}", ha="center", fontsize=6.5)
            ax.text(i + width / 2, row["IForest"] + 0.02, f"{row['IForest']*100:.1f}", ha="center", fontsize=6.5)
    handles = list(axes[0].containers[:2])
    legend_labels = ["OCSVM", "Isolation Forest"]
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(ARTIFACTS_DIR / "model_final.joblib")
    final_model = artifact["final_model"]
    threshold = final_model.threshold
    waves = split_waves(load_dataset())
    X22, y22 = X_y(waves["2022"])

    p = final_model.predict_proba(X22)[:, 1]
    df = X22.copy()
    df["grave"] = y22.to_numpy()
    df["p"] = p
    df["pred"] = (p >= threshold).astype(int)

    group_metrics = evaluate_groups(df)
    dc = decision_curve(df["grave"].to_numpy(), p)
    boot = shap_bootstrap(ARTIFACTS_DIR / "shap_values.npz")
    monitor = monitor_phase2(PROJECT_DIR / "Entrega2" / "src" / "resultados_completos.json")

    plot_errors_age(df, group_metrics, FIGURES_DIR / "errores_edad.png")
    plot_errors_sex_source(group_metrics, FIGURES_DIR / "errores_sexo_fuente.png")
    plot_decision_curve(dc, FIGURES_DIR / "decision_curve.png")
    plot_shap_bootstrap(boot, FIGURES_DIR / "shap_importance_ci.png")
    plot_monitor(monitor, FIGURES_DIR / "monitor_drift.png")

    out = {
        "threshold": float(threshold),
        "subgroups": group_metrics,
        "decision_curve": dc,
        "shap_bootstrap": boot,
        "monitor": monitor,
    }
    (ARTIFACTS_DIR / "b6_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )

    print("=== B6 — ESTABILIDAD, ERRORES Y MONITOR ===")
    print(f"Umbral: {threshold:.4f}")
    print("\nErrores por edad:")
    for label in AGE_LABELS:
        m = group_metrics["edad"][label]
        print(
            f"  {label:6s} n={m['n']:>6,} prev={m['prevalencia']:.4f} "
            f"R={m['Recall'] if m['Recall'] is not None else float('nan'):.3f} "
            f"P={m['Precision'] if m['Precision'] is not None else float('nan'):.3f} "
            f"F1={m['F1'] if m['F1'] is not None else float('nan'):.3f} "
            f"gap={m['gap_calibracion']:+.4f}"
        )
    print("\nCurva de decisión:")
    print(f"  Mejor beneficio neto: {dc['best']['nb_model']:.4f} en t={dc['best']['threshold']:.2f}")
    print(
        f"  En t=0.091: modelo={dc['default_threshold']['nb_model']:.4f} "
        f"vs tratar a todos={dc['default_threshold']['nb_all']:.4f}"
    )
    print("\nBootstrap SHAP (IC 95 %):")
    for k, v in boot.items():
        print(f"  {k:22s} {v['mean_abs_shap']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")
    print("\nMonitor fase 2:")
    for r in monitor["real"]:
        print(f"  {r['cohorte']:28s} OCSVM={r['OCSVM']*100:.2f}% IForest={r['IForest']*100:.2f}%")
    print(f"\nResultados: {ARTIFACTS_DIR / 'b6_results.json'}")
    print(f"Figuras: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
