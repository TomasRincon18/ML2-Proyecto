# B6. Estabilidad, errores por subgrupo, curva de decisión y monitor

## Objetivo

Cuantificar la estabilidad de la importancia SHAP, analizar dónde falla el
modelo final, evaluar su utilidad clínica con una curva de decisión y reutilizar
el monitor OCSVM de la fase 2.

## 1. Estabilidad de la importancia SHAP

Bootstrap de 1.000 réplicas sobre la muestra de 5.000 casos de 2022.

| Variable | Importancia media \|SHAP\| | IC 95 % |
|---|---:|---|
| **Edad** | **1.663** | [1.632, 1.697] |
| Sexo | 0.350 | [0.349, 0.353] |
| Ciudad | 0.218 | [0.214, 0.221] |
| Fuente de contagio | 0.162 | [0.157, 0.167] |

La dominancia de la edad es estable. Sexo, ciudad y fuente tienen efectos
pequeños pero con intervalos estrechos.

## 2. Errores por subgrupo (2022, umbral 0.091)

### Edad

| Grupo | n | Prevalencia | Recall | Precision | F1 | Gap calibración |
|---|---:|---:|---:|---:|---:|---:|
| 0-17 | 9.088 | 0.0009 | 0.000 | N/A | 0.000 | -0.0004 |
| 18-39 | 40.566 | 0.0007 | 0.000 | N/A | 0.000 | +0.0026 |
| 40-59 | 29.395 | 0.0023 | 0.000 | 0.000 | 0.000 | +0.0158 |
| 60-79 | 17.019 | 0.0242 | 0.684 | 0.042 | 0.080 | +0.0673 |
| 80+ | 3.932 | 0.1478 | 1.000 | 0.148 | 0.258 | +0.1686 |

**Hallazgo crítico:** al umbral 0,091 el modelo **no detecta ningún caso grave
menor de 60 años**. La sensibilidad global (0,786) proviene casi por completo de
los mayores de 60. En la práctica, el modelo se comporta como un clasificador
etario.

### Sexo

| Sexo | n | Prevalencia | Recall | Precision | F1 | Gap |
|---|---:|---:|---:|---:|---:|---:|
| F | 57.558 | 0.0079 | 0.702 | 0.070 | 0.128 | +0.019 |
| M | 42.442 | 0.0151 | 0.846 | 0.090 | 0.162 | +0.030 |

### Fuente de contagio

| Fuente | n | Prevalencia | Recall | Precision | F1 | Gap |
|---|---:|---:|---:|---:|---:|---:|
| COMUNITARIA | 83.284 | 0.0115 | 0.791 | 0.080 | 0.145 | +0.026 |
| RELACIONADO | 16.687 | 0.0082 | 0.752 | 0.095 | 0.169 | +0.015 |
| IMPORTADO | 29 | 0.0000 | N/A | N/A | N/A | +0.021 |

### Ciudades

El desempeño varía: Bogotá F1 0,102; Otros municipios 0,177; Pasto 0,286.
Todas las ciudades presentan **gap de calibración positivo** (sobreestimación),
entre +0,008 y +0,045.

## 3. Curva de decisión

| Umbral | Beneficio neto modelo | Tratar a todos | No tratar |
|---:|---:|---:|---:|
| 0.010 | **0.0064** | 0.0010 | 0 |
| 0.020 | 0.0043 | -0.0092 | 0 |
| 0.050 | 0.0010 | -0.0411 | 0 |
| 0.080 | -0.0007 | -0.0750 | 0 |
| **0.091** | **-0.0010** | -0.0868 | 0 |
| 0.200 | -0.0027 | -0.2363 | 0 |

El modelo supera a "tratar a todos" en todo el rango, pero su beneficio neto es
positivo solo para umbrales bajos (< 0,06). En el umbral 0,091 queda
ligeramente negativo.

**Implicación:** con \(C_{FN}/C_{FP}=10\), el umbral 0,091 está en el borde. Si
el costo clínico del falso negativo es mayor (umbral ≈ 0,05), el beneficio neto
se vuelve positivo. Es una decisión que debe acordarse con el equipo clínico; la
app debe permitir ajustar el umbral.

## 4. Monitor de drift de la fase 2

| Cohorte | OCSVM inliers | Isolation Forest |
|---|---:|---:|
| 2020 test (out-of-sample) | 98.96 % | 98.93 % |
| 2021 Delta | 99.08 % | 98.08 % |
| 2022 Ómicron | 97.99 % | 98.92 % |
| Simulación 5 % ruido | 99.97 % | 100.00 % |
| Simulación 10 % ruido | 95.94 % | 100.00 % |
| Simulación 30 % ruido | 72.00 % | 100.00 % |

**Lectura:** las cohortes reales siguen dentro del soporte de \(P(X)\). El OCSVM
detecta el drift simulado; Isolation Forest es insensible. El monitor cumple su
función de vigilar covariables, pero **no detecta el cambio de prevalencia ni el
cambio de \(P(Y|X)\)** que explica los errores del clasificador.

## 5. Implicaciones

1. El modelo es una **herramienta de triage para mayores de 60**, no un tamizaje
   poblacional. No debe usarse para descartar gravedad en jóvenes.
2. La sobreestimación es sistemática en todos los grupos, y crece con la edad
   (+0,169 en 80+).
3. El umbral 0,091 debe revisarse con el costo clínico real; 0,05 mejora el
   beneficio neto y la sensibilidad.
4. El monitor OCSVM y la calibración supervisada son complementarios:
   - OCSVM vigila \(P(X)\).
   - Calibración y análisis de subgrupos vigilan \(P(Y|X)\) y \(P(Y)\).

## Artefactos

- `artifacts/b6_results.json`
- `figures/shap_importance_ci.png`
- `figures/errores_edad.png`
- `figures/errores_sexo_fuente.png`
- `figures/decision_curve.png`
- `figures/monitor_drift.png`
- `docs/evidencias/b6_results.log`
