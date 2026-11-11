"""Aplicación Streamlit — Entrega 3.

Pestañas:
1. Predicción individual con explicación en dos capas.
2. (B8) Predicción por lotes + monitor de drift.
3. (B8) Ficha del modelo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import core  # noqa: E402
from src.config import ARTIFACTS_DIR, FEATURES, FIGURES_DIR  # noqa: E402


@st.cache_data
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())

st.set_page_config(page_title="Gravedad COVID-19 — Monitor", layout="wide")


@st.cache_resource
def load() -> dict:
    return core.load_artifact()


artifact = load()
final_model = artifact["final_model"]
categories = artifact["categories"]
shap_meta = artifact["shap"]

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Modelo")
    st.write(f"**Estrategia:** {artifact['strategy']}")
    st.write(f"**Umbral por defecto:** {final_model.threshold:.3f}")

    if "threshold" not in st.session_state:
        st.session_state["threshold"] = float(final_model.threshold)

    thr = st.slider("Umbral de decisión", 0.01, 0.50, key="threshold", step=0.01)
    st.caption(f"Costo FN/FP implícito: {(1 - thr) / thr:.1f}")

    c1, c2, c3 = st.columns(3)
    if c1.button("0.05", width="stretch"):
        st.session_state["threshold"] = 0.05
        st.rerun()
    if c2.button("0.091", width="stretch"):
        st.session_state["threshold"] = 0.0909
        st.rerun()
    if c3.button("0.20", width="stretch"):
        st.session_state["threshold"] = 0.20
        st.rerun()

    st.divider()
    st.caption("Métricas 2022")
    m = artifact["metrics_2022"]
    st.write(f"AUC-PR: {m['AUC_PR']:.3f}")
    st.write(f"ROC-AUC: {m['ROC_AUC']:.3f}")
    st.write(f"Brier: {m['Brier']:.4f}")

st.title("Predicción de gravedad clínica por COVID-19")
st.caption("Regresión logística + calibración Platt · monitor OCSVM (Jones & Farrow, 2025)")

tab_pred, tab_batch, tab_card = st.tabs(
    ["1 · Predicción individual", "2 · Lote + monitor", "3 · Ficha del modelo"]
)

# ---------------------------------------------------------------------------
# Pestaña 1: predicción individual
# ---------------------------------------------------------------------------
with tab_pred:
    with st.form("prediccion"):
        c1, c2 = st.columns(2)
        edad = c1.number_input(
            "Edad (años)",
            int(categories["edad_min"]),
            int(categories["edad_max"]),
            int(categories["edad_median"]),
            step=1,
        )
        sexo = c2.selectbox("Sexo", categories["sexo"])
        ciudad = c1.selectbox("Ciudad", categories["ciudad_top"], index=categories["ciudad_top"].index("BOGOTA"))
        fuente = c2.selectbox("Fuente de contagio", categories["fuente_tipo_contagio"])
        submitted = st.form_submit_button("Predecir", type="primary")

    if submitted:
        values = {
            "edad_anios": float(edad),
            "sexo": sexo,
            "ciudad_top": ciudad,
            "fuente_tipo_contagio": fuente,
        }
        res = core.predict_case(artifact, values)
        p = res["p_calibrada"]
        pred = 1 if p >= thr else 0

        col1, col2 = st.columns(2)
        col1.metric("Riesgo calibrado", f"{p:.1%}")
        col2.metric(
            "Decisión",
            "Grave (alerta)" if pred else "No grave",
            delta=f"umbral {thr:.3f}",
        )

        if edad < 60:
            st.warning(
                "Este modelo no detecta gravedad en menores de 60 años al umbral elegido. "
                "No lo use para descartar gravedad en ese grupo."
            )

        # Explicación simple: what-if en probabilidad.
        st.subheader("¿Qué cambiaría el riesgo?")
        wi = core.what_if(artifact, values)
        rows = []
        for feat in FEATURES:
            c = wi["cambios"][feat]
            rows.append(
                {
                    "Variable": core.FEATURE_LABELS[feat],
                    "Actual": str(c["actual"]),
                    "Referencia": str(c["referencia"]),
                    "delta_prob": c["delta_prob"],
                }
            )
        df = pd.DataFrame(rows)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("delta_prob", title="Δ riesgo (probabilidad)"),
                y=alt.Y("Variable", sort=None, title=None),
                color=alt.condition(
                    alt.datum.delta_prob > 0,
                    alt.value("#e63946"),
                    alt.value("#2a9d8f"),
                ),
                tooltip=["Variable", "Actual", "Referencia", "delta_prob"],
            )
            .properties(height=170)
        )
        st.altair_chart(chart, width="stretch")
        st.caption(
            "Cambio en el riesgo calibrado si cada variable tomara su valor de referencia "
            "(edad mediana, sexo femenino, Bogotá y contagio comunitario)."
        )

        # Explicación técnica: SHAP en log-odds.
        st.subheader("Detalle técnico (SHAP, log-odds)")
        sh = core.explain_log_odds(artifact, values)
        contrib = pd.Series(
            {core.GROUP_LABELS[g]: v for g, v in sh["contribuciones"].items()}
        ).sort_values()
        st.bar_chart(contrib, height=220)
        st.caption(
            f"Valor base {sh['base']:.3f} log-odds → logit {sh['logit']:.3f} → "
            f"probabilidad cruda {1 / (1 + 2.71828 ** (-sh['logit'])):.3f} → calibrada {p:.3f}. "
            "Los SHAP se suman en log-odds; la probabilidad calibrada se obtiene con Platt."
        )

# ---------------------------------------------------------------------------
# Pestaña 2: predicción por lotes + monitor
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Predicción por lotes")
    st.write("Suba un CSV con las columnas del modelo. La columna `grave` es opcional.")

    template = pd.DataFrame(
        [
            {
                "edad_anios": 65,
                "sexo": "F",
                "ciudad_top": "BOGOTA",
                "fuente_tipo_contagio": "COMUNITARIA",
                "grave": 0,
            }
        ]
    )
    st.download_button(
        "Descargar plantilla CSV",
        template.to_csv(index=False).encode(),
        "plantilla_lote.csv",
        "text/csv",
    )

    up = st.file_uploader("CSV del lote", type=["csv"], key="batch_csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            df.columns = [str(c).strip() for c in df.columns]
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo leer el CSV: {exc}")
        else:
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                st.error(f"Faltan columnas obligatorias: {missing}")
            else:
                for c in ["sexo", "ciudad_top", "fuente_tipo_contagio"]:
                    df[c] = df[c].astype(str).str.strip()
                df["edad_anios"] = pd.to_numeric(df["edad_anios"], errors="coerce")

                bad_age = df["edad_anios"].isna() | (df["edad_anios"] < 0) | (df["edad_anios"] > 115)
                if bad_age.any():
                    st.warning(f"{int(bad_age.sum())} filas tienen edad inválida; se imputará la mediana.")
                    df.loc[bad_age, "edad_anios"] = categories["edad_median"]

                for c in ["sexo", "ciudad_top", "fuente_tipo_contagio"]:
                    known = set(artifact["categories"][c])
                    unknown = sorted(set(df[c].unique()) - known)
                    if unknown:
                        st.warning(f"Categorías no vistas en {c}: {unknown}. El one-hot las ignora.")

                pred = core.predict_frame(artifact, df, threshold=thr)

                c1, c2, c3 = st.columns(3)
                c1.metric("Filas", f"{len(pred):,}")
                c2.metric("Riesgo medio", f"{pred['riesgo_calibrado'].mean():.1%}")
                c3.metric("Alertas", f"{int(pred['prediccion'].sum()):,}")

                st.dataframe(pred.head(100), width="stretch")

                if "grave" in df.columns:
                    y_all = pd.to_numeric(df["grave"], errors="coerce")
                    valid = y_all.isin([0, 1])
                    y = y_all[valid].astype(int)
                    p = pred.loc[valid, "riesgo_calibrado"]
                    if y.nunique() < 2:
                        st.info("El lote tiene una sola clase; no se calculan AUC ni matriz de confusión.")
                    else:
                        st.markdown("**Métricas del lote (con etiquetas)**")
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("AUC-PR", f"{average_precision_score(y, p):.3f}")
                        m2.metric("ROC-AUC", f"{roc_auc_score(y, p):.3f}")
                        m3.metric("Brier", f"{brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6)):.4f}")
                        preds = pred.loc[valid, "prediccion"]
                        m4.metric("Recall", f"{recall_score(y, preds, zero_division=0):.3f}")
                        m5.metric("F1", f"{f1_score(y, preds, zero_division=0):.3f}")
                        cm = confusion_matrix(y, preds)
                        st.dataframe(
                            pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]),
                            width="stretch",
                        )

                st.download_button(
                    "Descargar predicciones",
                    pred.to_csv(index=False).encode(),
                    "predicciones.csv",
                    "text/csv",
                )

                st.subheader("Distribución del riesgo en el lote")
                hist_risk = (
                    alt.Chart(pred)
                    .mark_bar(color="#1d3557")
                    .encode(
                        x=alt.X("riesgo_calibrado:Q", bin=alt.Bin(maxbins=30), title="Riesgo calibrado"),
                        y=alt.Y("count():Q", title="Casos"),
                    )
                )
                st.altair_chart(hist_risk, width="stretch")

                st.subheader("Comparación descriptiva con el entrenamiento")
                ref = artifact["reference_summary"]
                age = pred["edad_anios"].dropna()
                hist_age = (
                    alt.Chart(pd.DataFrame({"edad": age}))
                    .mark_bar(color="#457b9d")
                    .encode(
                        x=alt.X("edad:Q", bin=alt.Bin(maxbins=30), title="Edad (años)"),
                        y=alt.Y("count():Q", title="Casos"),
                    )
                )
                rule5 = (
                    alt.Chart(pd.DataFrame({"x": [ref["age_quantiles"]["5"]]}))
                    .mark_rule(color="#e63946", strokeDash=[4, 4])
                    .encode(x="x:Q")
                )
                rule95 = (
                    alt.Chart(pd.DataFrame({"x": [ref["age_quantiles"]["95"]]}))
                    .mark_rule(color="#e63946", strokeDash=[4, 4])
                    .encode(x="x:Q")
                )
                st.altair_chart(hist_age + rule5 + rule95, width="stretch")
                st.caption("Líneas rojas: percentiles 5 y 95 de la edad en el entrenamiento.")

                comparison = pd.DataFrame(
                    [
                        {
                            "Indicador": "Edad media",
                            "Lote": f"{age.mean():.1f}",
                            "Entrenamiento": f"{ref['mean_age']:.1f}",
                        },
                        {
                            "Indicador": "% femenino",
                            "Lote": f"{(pred['sexo'] == 'F').mean():.1%}",
                            "Entrenamiento": f"{ref['sexo_props'].get('F', 0):.1%}",
                        },
                        {
                            "Indicador": "% contagio comunitario",
                            "Lote": f"{(pred['fuente_tipo_contagio'] == 'COMUNITARIA').mean():.1%}",
                            "Entrenamiento": f"{ref['fuente_props'].get('COMUNITARIA', 0):.1%}",
                        },
                        {
                            "Indicador": "Ciudad más frecuente",
                            "Lote": str(pred["ciudad_top"].mode().iloc[0]),
                            "Entrenamiento": str(max(ref["ciudad_props"], key=ref["ciudad_props"].get)),
                        },
                    ]
                )
                st.dataframe(comparison, width="stretch")

                st.subheader("Monitor OCSVM / Isolation Forest (fase 2)")
                b6 = load_json(str(ARTIFACTS_DIR / "b6_results.json"))
                monitor_rows = [
                    {"Tipo": "Real", **row} for row in b6["monitor"]["real"]
                ] + [
                    {"Tipo": "Simulación", **row} for row in b6["monitor"]["simulaciones"]
                ]
                st.dataframe(pd.DataFrame(monitor_rows), width="stretch")
                monitor_fig = FIGURES_DIR / "monitor_drift.png"
                if monitor_fig.exists():
                    st.image(str(monitor_fig), caption="Sensibilidad del monitor de drift")
                st.caption(
                    "El monitor de la fase 2 vigila P(X). No detecta cambios de prevalencia ni de P(Y|X); "
                    "para eso están la calibración y el análisis de subgrupos."
                )

# ---------------------------------------------------------------------------
# Pestaña 3: ficha del modelo
# ---------------------------------------------------------------------------
with tab_card:
    st.subheader("Ficha del modelo final")
    m = artifact["metrics_2022"]
    d = m["decision"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC-PR", f"{m['AUC_PR']:.3f}")
    c2.metric("ROC-AUC", f"{m['ROC_AUC']:.3f}")
    c3.metric("Brier", f"{m['Brier']:.4f}")
    c4.metric("Recall @ umbral", f"{d['Recall']:.3f}")
    c5.metric("F1 @ umbral", f"{d['F1']:.3f}")

    st.markdown(
        f"""
        **Estrategia:** {artifact['strategy']}  
        **Entrenamiento:** 2020 completo + 80 % de 2021 (180.000 casos)  
        **Calibración:** Platt sobre el 20 % holdout de 2021 (20.000 casos)  
        **Test:** 2022 (100.000 casos)  
        **Umbral por defecto:** {final_model.threshold:.4f} (FN/FP = 10)  
        **Calibrador:** Platt (`a={artifact['calibrator']['coef']:.3f}`, `b={artifact['calibrator']['intercept']:.3f}`)
        """
    )

    st.markdown("### Calibración y utilidad clínica")
    col1, col2 = st.columns(2)
    cal_fig = FIGURES_DIR / "calibracion_final_2022.png"
    dc_fig = FIGURES_DIR / "decision_curve.png"
    with col1:
        if cal_fig.exists():
            st.image(str(cal_fig), caption="Curva de fiabilidad — 2022")
    with col2:
        if dc_fig.exists():
            st.image(str(dc_fig), caption="Curva de decisión — 2022")

    st.markdown("### Interpretabilidad")
    col1, col2 = st.columns(2)
    shap_fig = FIGURES_DIR / "shap_global_bar.png"
    pdp_fig = FIGURES_DIR / "pdp_edad.png"
    with col1:
        if shap_fig.exists():
            st.image(str(shap_fig), caption="Importancia global SHAP")
    with col2:
        if pdp_fig.exists():
            st.image(str(pdp_fig), caption="Dependencia parcial de la edad")

    st.markdown("### Errores por grupo de edad")
    b6 = load_json(str(ARTIFACTS_DIR / "b6_results.json"))
    age_rows = []
    for group, gm in b6["subgroups"]["edad"].items():
        age_rows.append(
            {
                "Grupo": group,
                "n": gm["n"],
                "Prevalencia": gm["prevalencia"],
                "Recall": gm["Recall"],
                "Precision": gm["Precision"],
                "F1": gm["F1"],
                "Gap calibración": gm["gap_calibracion"],
            }
        )
    st.dataframe(pd.DataFrame(age_rows), width="stretch")
    st.caption(
        "Al umbral por defecto, el modelo no detecta casos graves menores de 60 años. "
        "Es una herramienta de triage para mayores de 60, no un tamizaje poblacional."
    )

    st.markdown("### Limitaciones")
    st.markdown(
        """
        - **Modelo etario:** la edad domina la predicción; no detecta graves en menores de 60 al umbral 0,091.
        - **Brier skill negativo:** el valor está en el ranking y el triage, no en batir la tasa base.
        - **Sobreestimación en el decil superior:** 0,217 predicho vs 0,085 observado en 2022.
        - **2022 no es un test ciego:** ya se observó en la fase 2.
        - **Monitor P(X):** el OCSVM certifica estabilidad de covariables, no cambios de prevalencia ni de P(Y|X).
        """
    )
