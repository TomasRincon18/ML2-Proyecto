# Entrega 2 — Resumen para sustentar (2 min)

**Problema:** ¿cómo detectar si la población de pacientes cambia con el tiempo (*population drift*) y si eso rompe un modelo de predicción de gravedad de COVID-19 en Colombia?

## Qué se hizo

1. **Datos:** 6.390.844 registros crudos del INS → depuración (nulos, solo 2020–2022) → 6.311.405 casos → muestra estratificada de **300.000** (100k por ola). Prevalencia de gravedad: **2,82% (2020) → 2,33% (2021) → 1,10% (2022)**.
2. **Simulación de drift:** se desplazó la edad hacia el percentil 99 (≈92 años) con ruido al 5%, 10% y 30% de σ. Se justifica porque la letalidad crece de 0,1% en jóvenes a >30% en ancianos.
3. **Monitor de drift:** réplica del método de **Jones & Farrow (2025)** con **One-Class SVM** (RBF, ν=0,01) sobre las covariables X. Se validó primero en el dataset Wisconsin y luego en Colombia.
4. **Modelado supervisado:** Regresión Logística, Random Forest, XGBoost, LightGBM y SVM-RBF, con **AUC-PR** y ROC-AUC por el desbalance severo, más **Validación Cruzada Anidada (5×3)**.
5. **Evaluación temporal:** entrenar con 2020 y evaluar en 2021 y 2022.

**Con qué se hizo:** Python 3.11/3.12, `scikit-learn` (Pipeline, StandardScaler, OneHotEncoder, OCSVM, Nested CV), XGBoost, LightGBM, pandas/NumPy, Matplotlib/Seaborn. Todo en notebooks + scripts reproducibles (`src/ejecutar_experimentos.py`, `resultados_completos.json`).

## Resultados clave

- Todos los modelos **empatan estadísticamente**: AUC-PR ≈ 0,25–0,27 (±0,015–0,02). Nested CV de la Logística: **0,2662 ± 0,0143** → no hay sobreajuste de hiperparámetros.
- Se elige la **Regresión Logística** por parsimonia, interpretabilidad y estabilidad (mismo desempeño que los modelos avanzados).
- El **OCSVM está bien calibrado**: 1,04% de falsas alarmas *out-of-sample* (ν=0,01). Detecta el drift simulado (inliers caen a 72% con ruido 30%), mientras **Isolation Forest es insensible** (100%).
- En 2021–2022 el OCSVM da ~98–99% inliers: **P(X) no cambió** (KS D ≤ 0,06).

## Hallazgo principal (el que hay que defender)

En 2022 el F1 cae, pero ROC-AUC *sube* a 0,921 y el ratio AUC-PR/prevalencia pasa de 8,6× a 16,9×. No es falla del modelo ni drift de covariables: es **caída de la prevalencia** (vacunación). Es decir, **prevalence drift ≠ covariate drift**, y el OCSVM, al ser no supervisado y ciego a Y, no puede detectar cambios en P(Y).

## Limitación y siguiente fase

El OCSVM monitorea solo X; falta monitoreo de calibración supervisada. Para la Entrega Final: **SHAP**, **recalibración bayesiana** de probabilidades y despliegue en **Streamlit**.

## Frase de cierre

> "El modelo no se degradó: la población enfermó menos. El reto no era mejorar el clasificador, sino distinguir entre drift de datos y cambio epidemiológico."
