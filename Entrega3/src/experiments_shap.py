"""B5 — Interpretabilidad: SHAP global, SHAP local y PDP de edad.

Uso:
    python src/experiments_shap.py
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
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ARTIFACTS_DIR, FEATURES, FIGURES_DIR, SEED  # noqa: E402
from src.data import load_dataset, split_waves, X_y  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

GROUPS = ["edad_anios", "sexo", "ciudad_top", "fuente_tipo_contagio"]
GROUP_LABELS = {
    "edad_anios": "Edad",
    "sexo": "Sexo",
    "ciudad_top": "Ciudad",
    "fuente_tipo_contagio": "Fuente de contagio",
}
N_SHAP = 5000
N_BACKGROUND = 1000


def group_feature_indices(feature_names: list[str]) -> dict[str, list[int]]:
    groups = {g: [] for g in GROUPS}
    for j, name in enumerate(feature_names):
        if name.startswith("num__edad_anios"):
            groups["edad_anios"].append(j)
        elif name.startswith("cat__sexo_"):
            groups["sexo"].append(j)
        elif name.startswith("cat__ciudad_top_"):
            groups["ciudad_top"].append(j)
        elif name.startswith("cat__fuente_tipo_contagio_"):
            groups["fuente_tipo_contagio"].append(j)
    return groups


def select_cases(X: pd.DataFrame, p_cal: np.ndarray, threshold: float) -> list[dict]:
    """Selecciona tres perfiles representativos: bajo, alto y borderline."""
    cases = []
    ages = X["edad_anios"].to_numpy()

    low_mask = (ages >= 20) & (ages <= 40) & (p_cal < 0.01)
    if low_mask.any():
        idx = int(np.where(low_mask)[0][np.argmin(np.abs(ages[low_mask] - 30))])
        cases.append({"tipo": "Bajo riesgo", "idx": idx})

    high_mask = (ages >= 75) & (ages <= 90) & (p_cal > 0.15)
    if high_mask.any():
        idx = int(np.where(high_mask)[0][np.argmax(p_cal[high_mask])])
        cases.append({"tipo": "Alto riesgo", "idx": idx})

    border = np.abs(p_cal - threshold)
    border_mask = (ages >= 50) & (ages <= 80)
    if border_mask.any():
        idx = int(np.where(border_mask)[0][np.argmin(border[border_mask])])
    else:
        idx = int(np.argmin(border))
    if all(c["idx"] != idx for c in cases):
        cases.append({"tipo": "Caso límite", "idx": idx})

    return cases


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(ARTIFACTS_DIR / "model_final.joblib")
    final_model = artifact["final_model"]
    pipe = final_model.model
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]

    waves = split_waves(load_dataset())
    X22, y22 = X_y(waves["2022"])
    sample = X22.sample(n=min(N_SHAP, len(X22)), random_state=SEED).reset_index(drop=True)

    X_trans = np.asarray(prep.transform(sample), dtype=float)
    background = X_trans[: min(N_BACKGROUND, len(X_trans))]

    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.LinearExplainer(clf, masker)
    sv = explainer.shap_values(X_trans)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[-1])
    base = float(base)

    feature_names = list(prep.get_feature_names_out())
    groups = group_feature_indices(feature_names)
    sv_agg = np.column_stack([sv[:, idx].sum(axis=1) for idx in groups.values()])

    raw_logit = base + sv_agg.sum(axis=1)
    assert np.allclose(raw_logit, clf.decision_function(X_trans), atol=1e-8), "SHAP no reconstruye el logit"

    # Matriz numérica para colorear el beeswarm.
    city_order = {c: i for i, c in enumerate(sorted(X22["ciudad_top"].unique()))}
    X_plot = pd.DataFrame(
        {
            "edad_anios": sample["edad_anios"].to_numpy(dtype=float),
            "sexo": sample["sexo"].map({"F": 0, "M": 1}).to_numpy(dtype=float),
            "ciudad_top": sample["ciudad_top"].map(city_order).to_numpy(dtype=float),
            "fuente_tipo_contagio": sample["fuente_tipo_contagio"]
            .map({"COMUNITARIA": 0, "IMPORTADO": 1, "RELACIONADO": 2})
            .to_numpy(dtype=float),
        }
    )

    p_cal = final_model.predict_proba(sample)[:, 1]
    p_raw = 1 / (1 + np.exp(-raw_logit))

    # Importancia global.
    mean_abs = np.mean(np.abs(sv_agg), axis=0)
    mean_signed = np.mean(sv_agg, axis=0)
    importance = {
        g: {
            "mean_abs_shap": float(mean_abs[i]),
            "mean_signed_shap": float(mean_signed[i]),
        }
        for i, g in enumerate(GROUPS)
    }

    # Barra global.
    order = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
    ax.barh(
        [GROUP_LABELS[GROUPS[i]] for i in order],
        mean_abs[order],
        color="#2a9d8f",
        edgecolor="#264653",
        height=0.55,
    )
    for y, v in enumerate(mean_abs[order]):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Importancia media |SHAP| (log-odds)", fontsize=9.5)
    ax.set_title("Importancia global de variables — SHAP", fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_global_bar.png", dpi=300)
    plt.close()

    # Beeswarm global.
    shap.summary_plot(
        sv_agg,
        X_plot,
        feature_names=[GROUP_LABELS[g] for g in GROUPS],
        plot_type="dot",
        show=False,
        plot_size=(8.2, 4.4),
    )
    plt.title("SHAP global — 2022", fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=300)
    plt.close()

    # Casos locales, con valores legibles para el waterfall.
    local_data = pd.DataFrame(
        {
            "Edad": sample["edad_anios"].astype(int).astype(str),
            "Sexo": sample["sexo"],
            "Ciudad": sample["ciudad_top"],
            "Fuente de contagio": sample["fuente_tipo_contagio"],
        }
    )
    cases = select_cases(sample, p_cal, final_model.threshold)
    local_cases = []
    for case in cases:
        i = case["idx"]
        exp = shap.Explanation(
            values=sv_agg[i],
            base_values=base,
            data=local_data.iloc[i].to_numpy(dtype=object),
            feature_names=[GROUP_LABELS[g] for g in GROUPS],
        )
        shap.plots.waterfall(exp, max_display=4, show=False)
        plt.title(
            f"{case['tipo']} — edad {int(sample.iloc[i]['edad_anios'])}, "
            f"{sample.iloc[i]['sexo']}, {sample.iloc[i]['ciudad_top']}, "
            f"riesgo calibrado {p_cal[i]:.3f}",
            fontsize=9.5,
            fontweight="bold",
            pad=10,
        )
        plt.tight_layout()
        name = f"shap_local_{case['tipo'].lower().replace(' ', '_')}.png"
        plt.savefig(FIGURES_DIR / name, dpi=300, bbox_inches="tight")
        plt.close()

        local_cases.append(
            {
                "tipo": case["tipo"],
                "edad": float(sample.iloc[i]["edad_anios"]),
                "sexo": str(sample.iloc[i]["sexo"]),
                "ciudad": str(sample.iloc[i]["ciudad_top"]),
                "fuente": str(sample.iloc[i]["fuente_tipo_contagio"]),
                "p_raw": float(p_raw[i]),
                "p_calibrada": float(p_cal[i]),
                "raw_logit": float(raw_logit[i]),
                "contribuciones": {
                    GROUP_LABELS[g]: float(sv_agg[i, j]) for j, g in enumerate(GROUPS)
                },
                "figura": name,
            }
        )

    # PDP de edad.
    pdp_sample = X22.sample(n=300, random_state=SEED + 1).copy()
    ages = np.arange(0, 101, 1)
    means, p10s, p90s = [], [], []
    for age in ages:
        pdp_sample["edad_anios"] = age
        p = final_model.predict_proba(pdp_sample)[:, 1]
        means.append(float(np.mean(p)))
        p10s.append(float(np.percentile(p, 10)))
        p90s.append(float(np.percentile(p, 90)))

    means = np.array(means)
    p10s = np.array(p10s)
    p90s = np.array(p90s)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    ax.plot(ages, means, color="#1d3557", lw=2.2, label="Riesgo medio calibrado")
    ax.fill_between(ages, p10s, p90s, color="#1d3557", alpha=0.15, label="Banda P10–P90")
    ax.axhline(final_model.threshold, color="#e63946", linestyle="--", lw=1.2, label="Umbral 0.091")
    ax.set_xlabel("Edad (años)", fontsize=9.5)
    ax.set_ylabel("Probabilidad calibrada de gravedad", fontsize=9.5)
    ax.set_title("Dependencia parcial de la edad — 2022", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(0, 100)
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pdp_edad.png", dpi=300)
    plt.close()

    pdp_points = {
        str(age): {
            "mean": float(means[age]),
            "p10": float(p10s[age]),
            "p90": float(p90s[age]),
        }
        for age in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }

    out = {
        "n_samples": int(len(sample)),
        "n_background": int(len(background)),
        "base_value_log_odds": base,
        "global_importance": importance,
        "local_cases": local_cases,
        "pdp_age": pdp_points,
        "shap_version": shap.__version__,
        "model_strategy": artifact.get("strategy"),
    }
    (ARTIFACTS_DIR / "shap_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    np.savez_compressed(
        ARTIFACTS_DIR / "shap_values.npz",
        sv_agg=sv_agg,
        X_plot=X_plot.to_numpy(dtype=float),
        base=np.array([base]),
        feature_names=np.array(GROUPS),
        p_cal=p_cal,
    )

    print("=== B5 — SHAP + PDP ===")
    print(f"Muestra: {len(sample):,} casos de 2022 | background: {len(background):,}")
    print(f"Base value (log-odds): {base:.4f}")
    print("Importancia global (media |SHAP|):")
    for g in sorted(GROUPS, key=lambda x: -importance[x]["mean_abs_shap"]):
        print(f"  {GROUP_LABELS[g]:20s} {importance[g]['mean_abs_shap']:.4f}")
    print("\nCasos locales:")
    for c in local_cases:
        print(
            f"  {c['tipo']:12s} edad={c['edad']:.0f} {c['sexo']} {c['ciudad']} "
            f"p_cal={c['p_calibrada']:.3f}"
        )
    print("\nPDP edad (media calibrada):")
    for age in ["0", "20", "40", "60", "80", "100"]:
        print(f"  {age:>3} años: {pdp_points[age]['mean']:.4f}")
    print(f"\nResultados: {ARTIFACTS_DIR / 'shap_summary.json'}")
    print(f"Figuras: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
