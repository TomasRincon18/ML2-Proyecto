# Universidad Externado de Colombia

**Facultad:** Ciencia de datos
**Asignatura:** Machine Learning II
**Proyecto Integrador**

# Detección de *Population Drift* en modelos de predicción de gravedad clínica de COVID-19 en Colombia

**Segunda Entrega — Fase 2: datos, modelamiento y validación**

---

**Integrantes:**

- Ailyn Sofía Gómez Rodríguez
- Juan Tomas Rincón Pinzón

**Docente:** Wilmer Darío Pineda Ríos

**Bogotá D.C., Colombia**

**2026**

---

# Introducción

Esta segunda entrega ejecuta la Fase 2: consolida el conjunto de datos, construye un flujo de modelamiento reproducible sin fuga, reproduce y adapta el método del artículo base (Jones y Farrow, 2025) y evalúa los modelos con rigor. El proyecto mantiene dos piezas independientes y complementarias: (1) un **modelo supervisado de gravedad clínica** (el "diagnóstico ML desplegado"), y (2) un **OCSVM** que monitorea el *population drift* —el cambio en la distribución de las variables de entrada entre entrenamiento y despliegue—, que es el método que se reproduce del artículo. Esta entrega aporta un hallazgo honesto y matizado: en los datos colombianos, el desempeño del modelo de gravedad no se degrada por un drift de covariables, sino por un **cambio de prevalencia** (la proporción de casos graves cayó con la vacunación y las variantes), un fenómeno que el OCSVM —que solo observa las variables de entrada— no puede, ni debe, detectar.

---

# 1. Consolidación de los datos

## 1.1. Fuente y calidad

El conjunto es **"Casos positivos de COVID-19 en Colombia"** del Instituto Nacional de Salud (datos.gov.co, recurso `gt2j-8ykr`, licencia CC BY-SA 4.0). Se descargaron **6 390 844 registros** con 9 columnas. El Cuadro 1 resume el proceso de calidad.

**Cuadro 1.** Estado de calidad de los datos y transformaciones.

| Paso | Registros | Observación |
|---|---|---|
| Crudos | 6 390 844 | 9 columnas |
| `estado` nulo | −41 267 (0,65 %) | sin gravedad registrada, se excluyen |
| Fecha fuera de 2020–2022 | −~39 300 | 2023–2024 (fase endémica, fuera de alcance) |
| Edad fuera de [0, 120] | −0 | filtro de resguardo, no eliminó filas |
| Tras filtros | 6 349 577 | prevalencia grave = 2,26 % |

Notas de calidad: no hay nulos en edad, sexo ni fechas; 1 057 municipios (agrupados a 16); 3,1 M de filas idénticas en las 9 columnas (48,7 %), que **no son duplicados reales** —la columna `ID de caso` no se descargó y con 9 variables es esperable que pacientes distintos compartan valores—, por lo que no se eliminaron.

## 1.2. Variable objetivo y predictores

- **Objetivo `grave`**: `grave = 1` si `estado ∈ {Fallecido, Grave}` o `ubicación = "Hospital UCI"`; `grave = 0` si `estado ∈ {Leve, Moderado}`. Prevalencia global 2,26 %.
- **Predictores (4)**: `edad` (años), `sexo`, `ciudad` (top-15 + "Otros municipios") y `tipo de contagio`.
- **Cambios respecto a la Entrega 1 (declarados)**: se descartó "tipo de atención" porque contiene el desenlace (`Fallecido`, `Hospital UCI`) y filtraría el objetivo; la regla del objetivo ahora incluye explícitamente UCI y ya no usa "recuperado" (que en los datos es una columna de desenlace, no de gravedad); y se redujo a 4 predictores de bajo riesgo de fuga.

## 1.3. Partición temporal y muestreo

Partición **temporal** (no aleatoria), fiel a la lógica del artículo: 2020 (ola original, sin vacunación) como entrenamiento; 2021 (Delta) y 2022 (Ómicron) como despliegue. La prevalencia **desciende con el tiempo** (Cuadro 2), documentando el cambio de la tasa base. Tras ello, muestreo estratificado de **100 000 registros por ola** (300 000 en total). La Figura 1 verifica la estructura: la edad se concentra en adultos jóvenes, la clase grave es minoritaria y las ciudades capitales dominan el volumen (con "Otros municipios" como categoría agregada mayor).

**Cuadro 2.** Casos y prevalencia de gravedad por ola (antes del muestreo).

| Ola | Casos | P(grave) |
|---|---|---|
| 2020 (original) | 1 746 914 | 2,82 % |
| 2021 (Delta) | 3 506 739 | 2,33 % |
| 2022 (Ómicron) | 1 057 752 | 1,10 % |

![Distribución de edad por clase, prevalencia por ola y casos por ciudad](../figures/verificacion_estructura.png)

**Figura 1.** Verificación de la estructura del dataset reducido.

## 1.4. Simulación fundamentada

Se reprodujo el diseño del artículo adaptándolo a la edad, la variable clave de gravedad. **Adaptación justificada**: el artículo desplaza una variable a sus extremos; en la edad, desplazar hacia el mínimo colapsaría en 0, por lo que se modeló un **drift de envejecimiento** —la población desplegada es más vieja— anclando en el percentil 99 de la edad (85 años) + 0,4 σ (σ = 17,98), con ruido gaussiano del 5 %, 10 % y 30 % de σ. Es realista: en los casos del INS la gravedad crece con la edad (P(grave) de 0,1 % en <18 años a 31,8 % en ≥80), de modo que un corrimiento hacia los adultos mayores es el drift de mayor riesgo clínico. La Figura 2 verifica que la simulación reproduce la estructura prevista (un cluster que se desplaza a ~92 años y cuya dispersión crece con el ruido: sd 0,90 / 1,81 / 5,43).

![Distribución de edad real vs. simulada](../figures/verificacion_simulacion.png)

**Figura 2.** Distribución de edad real (2020) frente a las poblaciones simuladas de envejecimiento.

---

# 2. Flujo reproducible sin fuga de datos

Todo el preprocesamiento y el modelado se integró en `Pipeline` de `scikit-learn` (con `ColumnTransformer` y `Pipeline` de `imbalanced-learn`), con semilla fija (`SEED = 42`). Orden de etapas y prevención de fuga:

1. **Partición temporal antes de cualquier ajuste.**
2. **Preprocesamiento dentro del `Pipeline`** (imputación → one-hot → escalamiento), ajustado solo con el train de cada fold.
3. **SMOTE dentro del `Pipeline`**, generando sintéticos solo sobre el train de cada fold.
4. **`ubicación` excluida como predictor** (contiene el desenlace); solo se usó para construir `grave`.
5. **OCSVM entrenado sin la etiqueta** (no supervisado).

El código se entrega en `01_reduccion.ipynb` y `02_modelado.ipynb`, con `requirements.txt` y `src/descargar_ins.py` (descarga del dataset crudo).

---

# 3. Reproducción y adaptación del método

## 3.1. Validación de la reproducción en el propio terreno del artículo

Antes de adaptar, se **reprodujo el experimento del artículo sobre su propio dataset** (Wisconsin Breast Cancer): 21 variables tras eliminar las 9 colineales (|r| > 0,9), SMOTE hasta 714, estandarización, OCSVM (RBF, γ automático, ν = 0,01) y desplazamiento de `radius_mean` ±0,4 σ con ruido en todas las variables (5 semillas). El resultado coincide en orden de magnitud y tendencia con el artículo (Cuadro 3), lo que valida que nuestra implementación reproduce el método.

**Cuadro 3.** Réplica del experimento en Wisconsin (5 semillas) vs. artículo.

| Ruido | Artículo | Nuestra réplica |
|---|---|---|
| 5 % | 0,27 % | 0,08 % |
| 10 % | 4,86 % | 2,28 % |
| 30 % | 8,51 % | 3,75 % |

## 3.2. OCSVM como capa de monitoreo y sensibilidad de ν

Se reprodujo el OCSVM (RBF, γ automático, ν = 0,01) sobre 5 000 filas de 2020 (submuestreo por el costo O(n²) del kernel). Sobre las poblaciones simuladas reproduce la tendencia del artículo (Cuadro 4 y Figura 3): **a mayor ruido, más inliers** (23,4 % → 25,9 % → 36,7 %), porque el cluster desplazado se solapa más con la frontera.

Se evaluó la **sensibilidad de ν** (prometida como "calibrable" en la Entrega 1): con el drift al 30 %, ν = 0,01 deja 36,7 % de inliers, ν = 0,05 deja 1,5 % y ν = 0,10 deja 0,1 %; el costo es la falsa alarma sobre 2020 no usada (99,1 % / 94,5 % / 89,4 % de inliers). Es decir, ν es el dial del compromiso sensibilidad/falsa alarma.

**Cuadro 4.** Proporción de inliers en poblaciones simuladas: artículo vs. nuestro OCSVM e Isolation Forest.

| Ruido | Artículo | OCSVM (nuestro) | Isolation Forest |
|---|---|---|---|
| 5 % | 0,27 % | 23,4 % | 92,7 % |
| 10 % | 4,86 % | 25,9 % | 92,1 % |
| 30 % | 8,51 % | 36,7 % | 91,6 % |

El **Isolation Forest** responde menos que el OCSVM (8 % de outliers vs. 63 % a 30 % de ruido). Verificamos que esto **no es un artefacto de la codificación one-hot**: entrenado solo con la edad numérica sigue siendo poco sensible (98,3 % de inliers). Es una diferencia genuina de sensibilidad entre métodos, a favor del OCSVM para este tipo de drift.

![Sensibilidad al drift de OCSVM e Isolation Forest](../figures/sensibilidad_drift.png)

**Figura 3.** Proporción de inliers según el ruido para OCSVM e Isolation Forest.

## 3.3. Modelos de gravedad y línea base

Se compararon **Random Forest, XGBoost, LightGBM y SVM-RBF** (esta última con aproximación **Nystroem** para ser comparable sobre las 100 000 filas) frente a las líneas base de **mayoría** y **regresión logística**. Hiperparámetros con `GridSearchCV` (CV estratificada, k = 5, métrica AUC-PR) y **umbral elegido dentro de la CV** (maximizando F1), no fijo en 0,5.

---

# 4. Validación, métricas e interpretación

## 4.1. Métricas y umbral

Dado el desbalance y que el error costoso es el **falso negativo** (grave no detectado), se reportan **AUC-PR, F1, sensibilidad y precisión de la clase grave**, más ROC-AUC. El **umbral se seleccionó en validación cruzada**, lo que corrige la sobre-confianza inducida por SMOTE: el umbral óptimo resultó ~0,86 (no 0,5), elevando el F1 de ~0,21 a ~0,35.

**Cuadro 5.** Desempeño en validación cruzada (umbral elegido en CV).

| Modelo | AUC-PR | ROC-AUC | F1 (umbral CV) |
|---|---|---|---|
| Reg. logística | 0,265 | 0,909 | 0,346 |
| SVM RBF (Nystroem) | 0,267 | 0,908 | 0,346 |
| XGBoost | 0,264 | 0,907 | 0,344 |
| LightGBM | 0,256 | 0,906 | 0,339 |
| Random Forest | 0,228 | 0,901 | 0,318 |

La línea base de mayoría tiene F1 = 0 en CV y en las olas. **Hallazgo honesto**: con 4 predictores, los modelos avanzados **no superan a la regresión logística** (AUC-PR ~0,26 en todos, salvo Random Forest). El **mejor modelo elegido es la regresión logística**: empatado en discriminación, el más interpretable y estable, y el mejor fuera de muestra. La SVM-RBF se reporta como referencia del curso (con Nystroem ya es comparable), no como ganadora.

## 4.2. ¿Drift de covariables o cambio de prevalencia?

La evaluación temporal (Cuadro 6) muestra que el **ROC-AUC no cae** (0,891 → 0,921 de 2021 a 2022) y que el **AUC-PR relativo a la prevalencia mejora** (8,5× → 16,7×). La caída del AUC-PR absoluto se explica por el **descenso de la prevalencia** (2,82 % → 1,10 %), no por pérdida de capacidad discriminativa.

**Cuadro 6.** Evaluación temporal (regresión logística).

| Periodo | P(grave) | ROC-AUC | AUC-PR | AUC-PR / prevalencia |
|---|---|---|---|---|
| 2021 (Delta) | 2,33 % | 0,891 | 0,198 | 8,5× |
| 2022 (Ómicron) | 1,10 % | 0,921 | 0,183 | 16,7× |

Esto es coherente con el monitor de drift: las **covariables apenas cambiaron** (KS de edad 0,026 en 2021 y 0,061 en 2022; PSI de las categóricas ≤ 0,13), y el **OCSVM lo confirma** con ~99 % de inliers en ambas olas, igual que su falsa alarma natural sobre 2020 no usada (99,07 %). En otras palabras: **lo que cambió en 2022 no fue la distribución de X, sino la tasa base**, un cambio de *prior/calibración* que el OCSVM —que solo observa X— no puede detectar. Este es un límite esperado del método, no un fallo, y refina la lectura de la Entrega 1.

La Figura 4 (curva PR) y la Figura 5 (ROC) del mejor modelo sobre 2021, junto con su matriz de confusión a umbral 0,86 (sensibilidad 0,37; precisión 0,23), y la **calibración** (en 2021, el tercer tercil predice 0,53 cuando la tasa real es 0,063), evidencian la sobre-confianza heredada de SMOTE y justifican el umbral ajustado en CV.

![Curva precisión-sensibilidad](../figures/curva_pr_2021.png)

**Figura 4.** Curva precisión-sensibilidad del mejor modelo sobre 2021.

![Curva ROC](../figures/roc_2021.png)

**Figura 5.** Curva ROC del mejor modelo sobre 2021.

## 4.3. Interpretación preliminar

La **importancia por permutación** (Figura 6) confirma que la **edad domina** el poder predictivo (0,189 de caída en AUC-PR al permutarla), seguida de sexo (0,027), tipo de contagio (0,023) y ciudad (0,012). Esto es clínicamente coherente y justifica la elección de la edad como variable de la simulación de drift. La estabilidad entre folds es razonable (desviaciones ≤ 0,02–0,06 en AUC-PR), y el error de mayor costo son los graves no detectados.

![Importancia por permutación](../figures/importancia_permutacion.png)

**Figura 6.** Importancia por permutación (disminución de AUC-PR) del mejor modelo.

---

# Conclusión

La segunda entrega consolidó y verificó un dataset real colombiano (6,4 M registros del INS, reducidos a 300 000), reprodujo fielmente el OCSVM del artículo —validándolo primero en Wisconsin (0,08/2,28/3,75 % de inliers) y mostrando su sensibilidad a ν—, y lo adaptó a un drift de envejecimiento justificado por la estructura etaria de los casos. El hallazgo central es matizado y honesto: en Colombia, el desempeño del modelo de gravedad **no se degrada por drift de covariables, sino por cambio de prevalencia**, un límite que el OCSVM no puede detectar por diseño. La comparación de modelos muestra que, con 4 predictores, la complejidad no aporta y la regresión logística es la elección razonable. Quedan para la entrega final la interpretación con SHAP y el despliegue.

---

# Nota sobre el uso de IA

El equipo utilizó herramientas de inteligencia artificial generativa como apoyo para estructurar y redactar este documento y para construir y depurar el código reproducible, a partir de decisiones metodológicas discutidas y tomadas por el equipo. Todas las decisiones técnicas pueden ser explicadas y sustentadas por los integrantes.

---

# Referencias

[1] Jones, W. S., & Farrow, D. J. (2025). One-class support vector machines for detecting population drift in deployed machine learning medical diagnostics. *Scientific Reports*, 15, 12157.

[2] Instituto Nacional de Salud (INS). *Casos positivos de COVID-19 en Colombia*. Datos Abiertos Colombia. https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia/gt2j-8ykr

[3] Schölkopf, B., et al. (2001). Estimating the support of a high-dimensional distribution. *Neural Computation*, 13(7), 1443–1471.

[4] Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

[5] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

[6] Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. *Journal of Machine Learning Research*, 18(17), 1–5.

[7] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD*, 785–794.

[8] Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.

[9] Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *Proceedings of ICDM*, 413–422.
