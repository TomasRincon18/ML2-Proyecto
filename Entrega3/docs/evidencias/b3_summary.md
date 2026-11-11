# B3. Calibración I: estático, Platt de origen y EM

## Objetivo

Evaluar si las probabilidades del modelo final están calibradas y si una
corrección de prevalencia no supervisada (EM) mejora la calibración al pasar de
2020 a 2021 y 2022.

## Métodos comparados

| Método | Descripción | Usa etiquetas de 2021/2022 |
|---|---|---|
| **Estático** | Probabilidades crudas del modelo con `class_weight='balanced'` | No |
| **Platt de origen** | Regresión logística 1D sobre el logit de predicciones out-of-fold de 2020 | No |
| **EM de prevalencia** | Saerens-Latinne-Decaestecker sobre las probabilidades Platt | No |
| **Oracle** | Corrección de prevalencia con la prevalencia real de la cohorte | Sí (solo diagnóstico) |

## Resultados

### 2021 (Delta)

| Método | Media predicha | Brier | Log loss | ECE | AUC-PR |
|---|---:|---:|---:|---:|---:|
| Estático | 0.2277 | 0.1091 | 0.3467 | 0.2044 | 0.1992 |
| **Platt origen** | **0.0226** | **0.0204** | **0.0819** | **0.0021** | **0.1992** |
| EM | 0.0000 | 0.0233 | 0.3215 | 0.0233 | 0.0240 |
| Oracle | 0.0191 | 0.0204 | 0.0824 | 0.0042 | 0.1992 |

Prevalencia real: **0.0233**. EM estimó **0.0000** tras 895 iteraciones.

### 2022 (Ómicron)

| Método | Media predicha | Brier | Log loss | ECE | AUC-PR |
|---|---:|---:|---:|---:|---:|
| Estático | 0.2873 | 0.1631 | 0.5001 | 0.2763 | 0.1854 |
| **Platt origen** | 0.0363 | 0.0126 | 0.0565 | 0.0254 | 0.1854 |
| EM | 0.0811 | 0.0279 | 0.1070 | 0.0701 | 0.1854 |
| Oracle | 0.0163 | **0.0098** | **0.0429** | **0.0055** | 0.1854 |

Prevalencia real: **0.0110**. EM estimó **0.0811** tras 60 iteraciones.

## Hallazgos

1. **El modelo crudo está descalibrado en nivel.** La media predicha es 0,23 en
   2021 y 0,29 en 2022, frente a prevalencias reales de 0,023 y 0,011. El
   `class_weight='balanced'` desplaza las probabilidades, no el ranking.
2. **Platt de origen corrige casi toda la descalibración.** Sin usar etiquetas
   de 2021/2022, el Brier cae de 0,109 a 0,020 en 2021 y de 0,163 a 0,013 en
   2022; el ECE cae a 0,002 y 0,025. El AUC-PR no cambia porque la
   transformación es monótona.
3. **EM no es viable en este problema.** En 2021 colapsa a prevalencia 0 y
   destruye el ranking (AUC-PR 0,024); en 2022 sobreestima la prevalencia
   (0,081 vs 0,011) y empeora el Brier frente a Platt. La suposición de
   *prior shift* puro no se sostiene: también cambió \(P(Y|X)\).
4. **El oracle muestra el techo.** Con la prevalencia real, el Brier baja a
   0,0204 en 2021 y 0,0098 en 2022. Platt ya captura casi todo el beneficio en
   2021; en 2022 queda un margen pequeño.

## Implicación para B4

La calibración supervisada con etiquetas de 2021 (Platt/isotónica) y el
reentrenamiento con 2020+2021 son las alternativas viables. EM queda descartado
como mecanismo principal, pero se reporta como evidencia de que el cambio no es
un simple *prior shift*.

## Artefactos

- `src/calibration.py`: Platt, EM, curvas de fiabilidad y ECE.
- `src/experiments_calibration.py`: experimento reproducible.
- `artifacts/calibration_b3.json`: métricas completas.
- `figures/calibracion_2021.png`, `figures/calibracion_2022.png`.
- `figures/brier_prevalencia.png`.
- `docs/evidencias/b3_calibration.log`.
