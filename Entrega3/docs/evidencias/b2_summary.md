# B2. Modelo final y umbral

## Configuración definitiva

| Elemento | Valor |
|---|---|
| Modelo | Regresión logística |
| `C` | 0.1 |
| `class_weight` | `balanced` |
| `max_iter` | 1000 |
| Semilla | 42 |
| Entrenamiento | 100.000 casos de 2020 |
| Calibración/umbral | 100.000 casos de 2021 |
| Test final | 100.000 casos de 2022 |

## Validación y discriminación

| Métrica | Valor |
|---|---:|
| CV AUC-PR (5 folds, 2020) | 0.2666 |
| AUC-PR 2021 | 0.1992 |
| ROC-AUC 2021 | 0.8918 |
| Brier 2021 | 0.1091 |
| AUC-PR 2022 | 0.1854 |
| ROC-AUC 2022 | 0.9209 |
| Brier 2022 | 0.1631 |

El modelo reproduce la fase 2: AUC-PR 0,1992 en 2021 y 0,1854 en 2022.

## Puntos de operación

| Punto | Umbral | 2021 Recall | 2021 Precision | 2021 F1 | 2022 Recall | 2022 Precision | 2022 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Costo FN/FP=10 (default)** | **0.75** | 0.5307 | 0.1758 | 0.2641 | 0.8069 | 0.0709 | 0.1303 |
| F1-óptimo | 0.82 | 0.4336 | 0.2103 | 0.2832 | 0.7559 | 0.0914 | 0.1630 |
| Recall ≥ 80 % | 0.43 | 0.8083 | 0.0928 | 0.1665 | 0.9344 | 0.0365 | 0.0703 |

## Tabla de costo (2021)

| FN/FP | Umbral | Recall | Precision | F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.95 | 0.1203 | 0.3256 | 0.1757 |
| 2 | 0.95 | 0.1203 | 0.3256 | 0.1757 |
| 5 | 0.90 | 0.2742 | 0.2653 | 0.2697 |
| **10** | **0.75** | **0.5307** | **0.1758** | **0.2641** |
| 20 | 0.60 | 0.6906 | 0.1288 | 0.2171 |
| 50 | 0.34 | 0.8668 | 0.0800 | 0.1465 |
| 100 | 0.19 | 0.9385 | 0.0580 | 0.1093 |

## Hallazgo de B2

El umbral óptimo de la fase 2 (0,16) se calculó sobre validación cruzada de 2020. Al pasar a 2021, ese umbral produce recall 95,3 % y precisión 5,3 % (F1 0,1006). El umbral que maximiza F1 en 2021 es **0,82**, y el que minimiza el costo con FN/FP=10 es **0,75**. Es decir, el umbral no es transportable entre cohortes con distinta prevalencia; esto se aborda en B3-B4 con calibración.

## Artefacto

- `artifacts/model.joblib`: pipeline + umbrales + puntos de operación + caso de referencia.
- `artifacts/metrics.json`: CV, tabla de umbrales, tabla de costos y evaluación 2021/2022.
- Paridad del caso de referencia: `0.0`.

## Decisión provisional

Se adopta **0,75** como umbral default por costo clínico (FN 10× FP). Los umbrales 0,82 (F1) y 0,43 (recall ≥ 80 %) quedan como puntos de operación alternativos para la app. La calibración de B3-B4 puede mover este default.
