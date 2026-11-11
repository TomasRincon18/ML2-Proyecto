# B4. Calibración II y selección del modelo final

## Objetivo

Comparar las estrategias viables de calibración y adaptación, elegir el modelo
final y fijar el umbral de decisión para el despliegue.

## Estrategias comparadas

| Estrategia | Entrenamiento | Calibración | Usa 2021 |
|---|---|---|---|
| Estático | 2020 | Ninguna | No |
| Platt origen | 2020 | Platt sobre OOF de 2020 | No |
| Platt 2021 | 2020 | Platt sobre 2021 | Sí |
| Isotónica 2021 | 2020 | Isotónica sobre 2021 | Sí |
| **Reentreno + Platt** | **2020 + 80 % de 2021** | **Platt sobre 20 % holdout de 2021** | **Sí** |
| Reentreno + isotónica | 2020 + 80 % de 2021 | Isotónica sobre 20 % holdout | Sí |

El umbral de decisión para probabilidades calibradas es el bayesiano:

\[
t^* = \frac{1}{1 + C_{FN}/C_{FP}} = \frac{1}{1+10} = 0.0909
\]

## Resultados en 2022

| Estrategia | Brier | ECE | Log loss | AUC-PR | ROC-AUC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Estático | 0.16308 | 0.27628 | 0.50014 | 0.1854 | 0.9209 | 0.8069 | 0.0709 | 0.1303 |
| Platt origen | 0.01264 | 0.02536 | 0.05650 | 0.1854 | 0.9209 | 0.7832 | 0.0785 | 0.1427 |
| Platt 2021 | 0.01213 | 0.02522 | 0.05627 | 0.1854 | 0.9209 | 0.7832 | 0.0784 | 0.1426 |
| Isotónica 2021 | 0.01188 | 0.02489 | 0.05687 | 0.1726 | 0.9202 | 0.8051 | 0.0709 | 0.1303 |
| **Reentreno + Platt** | **0.01192** | **0.02380** | **0.05479** | **0.1903** | **0.9228** | 0.7860 | **0.0814** | **0.1475** |
| Reentreno + isotónica | 0.01193 | 0.02385 | 0.05602 | 0.1683 | 0.9219 | 0.8306 | 0.0683 | 0.1262 |

## Bootstrap (300 réplicas de 30.000 casos)

| Estrategia | AUC-PR media [IC 95 %] | Brier media [IC 95 %] | F1 media [IC 95 %] |
|---|---|---|---|
| Estático | 0.1899 [0.1518, 0.2278] | 0.1632 [0.1602, 0.1658] | 0.1303 [0.1169, 0.1427] |
| Platt origen | 0.1899 [0.1518, 0.2278] | 0.0127 [0.0119, 0.0135] | 0.1427 [0.1277, 0.1573] |
| Platt 2021 | 0.1899 [0.1518, 0.2278] | 0.0122 [0.0114, 0.0130] | 0.1425 [0.1276, 0.1572] |
| **Reentreno + Platt** | **0.1947 [0.1579, 0.2326]** | **0.0119 [0.0112, 0.0128]** | **0.1475 [0.1321, 0.1625]** |

Diferencia reentreno + Platt vs Platt 2021:

| Métrica | Diferencia media [IC 95 %] | P(mejora) |
|---|---|---:|
| AUC-PR | +0.0049 [+0.0004, +0.0093] | 0.98 |
| Brier | -0.0002 [-0.0003, -0.0002] | 1.00 |
| F1 | +0.0050 [+0.0019, +0.0077] | 1.00 |

`P(mejora)` usa la dirección correcta de cada métrica: mayor AUC-PR/F1 y menor
Brier. La mejora es pequeña pero consistente: los intervalos excluyen cero para
Brier y F1, y apenas para AUC-PR.

## Decisión

Se elige **reentreno 2020 + 80 % de 2021 + Platt sobre el 20 % holdout de 2021**,
con umbral **0.0909** para \(C_{FN}/C_{FP}=10\).

Razones:

1. Mejor AUC-PR (0.1903) y F1 (0.1475) en 2022.
2. Mejor Brier (0.01192) y ECE (0.02380) entre las alternativas viables.
3. Usa información reciente (2021) sin tocar 2022.
4. Sigue siendo una regresión logística interpretable.

## Artefacto final

- `artifacts/model_final.joblib`:
  - `FinalModel` con pipeline + `PlattCalibrator` + umbral 0.0909.
  - 180.000 casos de entrenamiento (2020 + 80 % de 2021).
  - 20.000 casos de calibración (20 % de 2021).
  - Métricas de 2022 y caso de referencia.
- Paridad tras recargar: **0.0**.

## Análisis de sensibilidad: beta calibration

Se probó un calibrador beta (Kull et al., 2017) sobre el mismo holdout de 2021:

\[
\text{logit}(\hat p) = 1.0978\log(p) - 0.7895\log(1-p) - 3.3120
\]

| Método | Brier | ECE | Log loss | AUC-PR | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Platt | 0.01192 | 0.02380 | 0.05479 | 0.1903 | 0.7860 | 0.0814 | **0.1475** |
| Beta | **0.01173** | **0.02364** | **0.05474** | 0.1903 | 0.7969 | 0.0790 | 0.1438 |

Beta mejora marginalmente Brier/ECE, pero baja el F1 y no corrige el decil
superior (0.2107 predicho vs 0.0850 observado). **Se mantiene Platt** en el
artefacto final.

## Advertencias

- **Brier skill negativo** (-0.098): el Brier del modelo (0.01192) es
  ligeramente peor que predecir siempre la prevalencia (0.01086). El valor del
  modelo está en el ranking (AUC-PR 0.1903 ≈ 17× la prevalencia) y en la
  decisión, no en batir la tasa base en Brier.
- **2022 no es un test ciego**: ya se había observado en la fase 2. Se reporta
  como evaluación temporal final y se declara esta limitación.
- **EM queda descartado** como mecanismo principal (B3): el cambio no es un
  *prior shift* puro.
- **Isotónica** calibra bien pero degrada el ranking (AUC-PR 0.1726); por eso no
  se elige.

## Siguiente

B5 usará `model_final.joblib` para SHAP global, SHAP local y PDP de edad.
