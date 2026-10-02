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

Esta segunda entrega ejecuta la Fase 2 del proyecto: consolida el conjunto de datos, construye un flujo de modelamiento reproducible sin fuga de información, reproduce y adapta el método del artículo base (Jones y Farrow, 2025) con las técnicas del curso, y evalúa el desempeño de los modelos con rigor. El objetivo sigue siendo doble: (1) predecir la **gravedad clínica** de un caso de COVID-19 a partir de variables disponibles al momento del diagnóstico, y (2) monitorear el ***population drift*** —el cambio en la distribución de las variables de entrada entre entrenamiento y despliegue— mediante un *One-Class SVM* (OCSVM) entrenado sin la variable objetivo. El artículo base propone exactamente este OCSVM como capa de monitoreo no supervisada; aquí lo reproducimos y lo adaptamos al contexto colombiano, usando datos reales del Instituto Nacional de Salud (INS) y una simulación controlada que replica el diseño experimental de los autores.

---

# 1. Consolidación de los datos

## 1.1. Fuente y caracterización

El conjunto principal es **"Casos positivos de COVID-19 en Colombia"** del Instituto Nacional de Salud (datos.gov.co, recurso `gt2j-8ykr`). Se descargaron **6 390 844 registros** (9 columnas seleccionadas por pertinencia: fecha de notificación, edad, unidad de medida de edad, sexo, departamento, municipio, tipo de contagio, ubicación del caso y estado). La fuente es oficial (procedencia `official`) y está licenciada bajo CC BY-SA 4.0.

## 1.2. Limpieza y construcción de la variable objetivo

Se aplicaron las siguientes transformaciones:

- **Normalización de texto** (mayúsculas y sin espacios) sobre `estado`, `sexo`, `ubicacion`, `fuente_tipo_contagio`, `ciudad_municipio_nom` y `departamento_nom`.
- **Edad en años**: la variable `edad` viene acompañada de `unidad_medida` (1 = años, 2 = meses, 3 = días); los meses y días se convirtieron a años, y se descartaron edades fuera de [0, 120].
- **Variable objetivo binaria `grave`**, construida según la regla acordada: `grave = 1` si `estado ∈ {Fallecido, Grave}` **o** `ubicacion = "Hospital UCI"`; `grave = 0` si `estado ∈ {Leve, Moderado}`. Se **excluyeron** los registros con `estado = "N/A"` (muertes no relacionadas con COVID-19), pues no constituyen gravedad atribuible a la enfermedad.

Tras definir el objetivo y excluir `N/A` se conservaron **6 349 577 registros**, con una prevalencia de gravedad del **2,26 %** (143 227 casos graves), un desbalance severo que justifica el uso de SMOTE y de métricas sensibles al desbalance.

## 1.3. Geografía, partición temporal y muestreo

Para la geografía se conservaron las **15 ciudades** con mayor número de casos y el resto se agrupó en la categoría `Otros municipios`, reduciendo la cardinalidad de 1 053 municipios a 16 categorías manejables para codificación y SMOTE.

La partición es **temporal**, reproduciendo la lógica del artículo (entrenar sobre una población y evaluar sobre otra potencialmente distinta): 2020 (ola original, sin vacunación) como entrenamiento; 2021 (Delta) y 2022 (Ómicron) como poblaciones de despliegue. La prevalencia de gravedad **desciende en el tiempo** (Cuadro 1), lo que documenta un drift real: la población cambió (vacunación, variantes) y con ella la distribución del desenlace.

**Cuadro 1.** Casos y prevalencia de gravedad por ola (datos consolidados antes del muestreo).

| Ola | Casos | P(grave) |
|---|---|---|
| 2020 (original) | 1 746 914 | 2,82 % |
| 2021 (Delta) | 3 506 739 | 2,33 % |
| 2022 (Ómicron) | 1 057 752 | 1,10 % |

Finalmente se realizó un **muestreo estratificado por ola** de **100 000 registros por ola** (300 000 en total), preservando la proporción de graves dentro de cada ola y manteniendo la comparabilidad entre olas para el OCSVM. La Figura 1 verifica la estructura del conjunto reducido: la edad se concentra en adultos jóvenes, la clase grave es minoritaria en todas las olas y las ciudades capitales dominan el volumen de casos.

![Distribución de edad por clase, prevalencia de gravedad por ola y casos por ciudad](../figures/verificacion_estructura.png)

**Figura 1.** Verificación de la estructura del dataset reducido: (izquierda) edad por clase, (centro) prevalencia de gravedad por ola, (derecha) casos por ciudad.

## 1.4. Simulación fundamentada

Se reprodujo el diseño experimental de Jones y Farrow. El artículo desplaza la variable clave (`radius_mean`) **±0,4 desviaciones estándar** desde sus valores extremos y añade ruido gaussiano del 5 %, 10 % y 30 % de la desviación estándar. En nuestro caso la variable clave es la **edad** (reconocida como el principal factor de riesgo de gravedad). Adaptación justificada: los extremos mínimo/máximo de la edad son degenerados (mínimo 0 con muchos casos, máximo 114 aislado), por lo que se ancló el desplazamiento en los **percentiles 1 y 99** de la edad (3,0 y 85,0 años) y se desplazó ±0,4 · σ (σ = 17,98 años). La Figura 2 verifica que la simulación reproduce la estructura prevista: dos subpoblaciones desplazadas hacia los extremos (jóvenes ~5 años y mayores ~92 años) con dispersión creciente según el ruido, frente a la distribución real unimodal del entrenamiento.

![Distribución de edad real (train) vs simulada por nivel de ruido](../figures/verificacion_simulacion.png)

**Figura 2.** Distribución de edad real (train 2020) frente a las poblaciones simuladas a 5 %, 10 % y 30 % de ruido.

---

# 2. Flujo reproducible sin fuga de datos

Todo el preprocesamiento y el modelado se integraron en **`Pipeline`** de `scikit-learn` (con `ColumnTransformer` y `Pipeline` de `imbalanced-learn`), de modo que las mismas transformaciones se aplican de forma idéntica y trazable, con **semilla fija** (`SEED = 42`) para garantizar la reproducibilidad.

**Orden de etapas y prevención de fuga:**

1. **Partición temporal antes de cualquier ajuste**: los conjuntos de entrenamiento (2020) y despliegue (2021, 2022) se separan por fecha antes de tocar los datos, de modo que ninguna información de las olas posteriores contamina el entrenamiento.
2. **Preprocesamiento dentro del `Pipeline`** (`ColumnTransformer`): imputación (mediana) sobre las numéricas → codificación *one-hot* sobre las categóricas → escalamiento `StandardScaler` sobre las numéricas. Cada etapa se ajusta **solo con el train de cada fold/partición**; el test nunca participa en el ajuste del escalador ni del codificador.
3. **SMOTE dentro del `Pipeline`**: el sobremuestreo sintético de la clase grave se genera únicamente sobre los datos de entrenamiento de cada fold, evitando que ejemplos sintéticos derivados del test filtren información.
4. **`ubicacion` excluida como predictor**: contiene el desenlace (`Fallecido`, `Hospital UCI`) y su uso como variable de entrada filtraría la variable objetivo; solo se empleó para *construir* `grave`, nunca para predecirlo.
5. **OCSVM entrenado sin la etiqueta**: el monitor de drift se ajusta exclusivamente sobre las variables de entrada de 2020, sin usar la variable objetivo (es no supervisado).

El código reproducible se entrega en dos cuadernos (`01_reduccion.ipynb` y `02_modelado.ipynb`), acompañados de `requirements.txt` (dependencias fijadas) y de `src/descargar_ins.py` (descarga del dataset crudo). La semilla aleatoria `SEED = 42` está fijada en todo el flujo para garantizar la repetibilidad.

---

# 3. Reproducción y adaptación del método

## 3.1. OCSVM como capa de monitoreo (reproducción)

Se reprodujo el OCSVM del artículo: kernel **RBF**, `gamma = "auto"` (1 / n_features) y **ν = 0,01** (≈1 % de contaminación esperada). Por el costo cuadrático O(n²) del kernel RBF, se entrenó sobre un subconjunto aleatorio de 5 000 filas de 2020 (≈7 veces el tamaño del artículo, 714). Como método alternativo de detección —cubriendo una limitación explícita del artículo— se añadió **Isolation Forest** con el mismo nivel de contaminación.

El modelo se evaluó sobre 2020, 2021, 2022 y las tres poblaciones simuladas, midiendo la proporción de *inliers*. El Cuadro 2 reproduce la tendencia del artículo: **a mayor ruido, mayor proporción de inliers**, porque la población desplazada se solapa progresivamente con la frontera de entrenamiento.

**Cuadro 2.** Proporción de inliers en las poblaciones simuladas: artículo vs. nuestra reproducción.

| Nivel de ruido | Inliers (Jones & Farrow) | Inliers (nuestro OCSVM) | Inliers (Isolation Forest) |
|---|---|---|---|
| 5 % | 0,27 % | 16,1 % | 95,0 % |
| 10 % | 4,86 % | 17,6 % | 94,7 % |
| 30 % | 8,51 % | 29,5 % | 94,5 % |

La tendencia es la misma, aunque con mayor proporción de inliers, por el espacio de características mixto (una numérica y tres categóricas codificadas) y una frontera menos ajustada. Es notable que **Isolation Forest apenas responde** (~95 % de inliers en los tres niveles), mientras el OCSVM es claramente más sensible al desplazamiento sutil de la edad, lo que apoya la elección del OCSVM como monitor de drift (Figura 3). Sobre las olas reales, el OCSVM reportó 98,95 % (2021) y 98,12 % (2022) de inliers frente al 99,02 % del entrenamiento, indicando un desplazamiento leve pero creciente en los datos reales.

![Proporción de inliers en poblaciones simuladas según el nivel de ruido](../figures/sensibilidad_drift.png)

**Figura 3.** Sensibilidad al drift del OCSVM y de Isolation Forest ante poblaciones simuladas con ruido creciente.

## 3.2. Modelos de gravedad y línea base (adaptación)

Para el problema de gravedad se entrenaron y compararon modelos avanzados del curso —**Random Forest, XGBoost, LightGBM y SVM con kernel RBF**— frente a dos líneas base: **clasificador de mayoría** y **regresión logística** (nivel Machine Learning I). El ajuste de hiperparámetros se hizo con `GridSearchCV` y validación cruzada estratificada (k = 5), usando **AUC-PR** (*average precision*) como métrica de selección por el desbalance.

**Justificación de las adaptaciones respecto al artículo:** (i) el artículo trabaja con 21 variables numéricas; aquí las variables son mixtas, por lo que se codificaron las categóricas (*one-hot*) y se escalaron las numéricas; (ii) el artículo aplicó SMOTE para balancear las clases *antes* de entrenar el OCSVM; aquí el SMOTE se aplica solo al modelo supervisado, pues el OCSVM es no supervisado y no usa la etiqueta; (iii) se ancló la simulación en percentiles en lugar de mínimo/máximo (degenerados en edad); (iv) se añadió la comparación con Isolation Forest, que el artículo dejó como trabajo futuro. En conjunto, la adaptación respeta el propósito original —una capa de monitoreo no supervisada del drift— y lo extiende a datos reales colombianos.

---

# 4. Validación, métricas e interpretación preliminar

## 4.1. Estrategia de validación

Se combinaron tres estrategias complementarias:

1. **Validación cruzada estratificada (k = 5)** dentro de 2020 para el ajuste de hiperparámetros y el reporte de variabilidad (media ± desviación).
2. **Validación cruzada anidada** (CV externa para estimar desempeño, CV interna para ajustar hiperparámetros) sobre los modelos representativos, para obtener una estimación **no sesgada** del desempeño sin el sesgo de selección del `GridSearchCV`.
3. **Evaluación temporal fuera de muestra** sobre 2021 (Delta) y 2022 (Ómicron), que es la prueba real del drift.

## 4.2. Métricas

Dado el desbalance (2,08 % de graves) y que el error costoso es el **falso negativo** (un grave no detectado), se priorizaron **F1, AUC-PR y recall de la clase grave**, complementadas con precisión y ROC-AUC. El Cuadro 3 resume el desempeño con validación cruzada.

**Cuadro 3.** Desempeño en validación cruzada (media ± desviación entre folds).

| Modelo | F1 | Precisión | Recall | ROC-AUC | AUC-PR |
|---|---|---|---|---|---|
| Reg. logística | 0,217 ± 0,004 | 0,124 ± 0,003 | 0,851 ± 0,010 | 0,909 ± 0,003 | 0,267 ± 0,015 |
| Random Forest | 0,222 ± 0,004 | 0,128 ± 0,003 | 0,837 ± 0,019 | 0,902 ± 0,004 | 0,241 ± 0,015 |
| XGBoost | 0,218 ± 0,005 | 0,125 ± 0,003 | 0,849 ± 0,014 | 0,907 ± 0,003 | 0,266 ± 0,022 |
| LightGBM | **0,249 ± 0,005** | **0,149 ± 0,004** | 0,774 ± 0,025 | 0,901 ± 0,003 | 0,245 ± 0,018 |
| SVM RBF | 0,209 ± 0,016 | 0,120 ± 0,009 | 0,815 ± 0,089 | 0,907 ± 0,027 | **0,279 ± 0,062** |

La línea base de mayoría tiene F1 = 0 (no detecta ningún grave), por lo que todos los modelos avanzados aportan mejora sustancial. La CV anidada (Cuadro 4) coincide casi exactamente con la CV simple, confirmando que el ajuste de hiperparámetros **no introdujo sesgo de selección** relevante.

**Cuadro 4.** Validación cruzada anidada (estimación no sesgada).

| Modelo | AUC-PR | F1 | ROC-AUC |
|---|---|---|---|
| Reg. logística | 0,267 ± 0,015 | 0,217 ± 0,004 | 0,909 ± 0,003 |
| XGBoost | 0,266 ± 0,022 | 0,218 ± 0,005 | 0,907 ± 0,003 |
| LightGBM | 0,245 ± 0,018 | 0,249 ± 0,005 | 0,901 ± 0,003 |

## 4.3. Efecto del drift y primera lectura del mejor modelo

El desempeño **cae fuera de muestra** conforme avanzan las olas (Cuadro 5): el AUC-PR de todos los modelos desciende en 2022 (0,12–0,18) respecto de 2021 (0,17–0,20) y del valor de CV en 2020 (~0,25–0,28). Esto evidencia que la población cambió y que el modelo pierde capacidad discriminativa en la población desplazada —precisamente el fenómeno que el OCSVM de la sección 3 busca alertar.

**Cuadro 5.** Evaluación temporal fuera de muestra (AUC-PR).

| Modelo | 2021 (Delta) | 2022 (Ómicron) |
|---|---|---|
| Reg. logística | 0,198 | 0,183 |
| Random Forest | 0,171 | 0,115 |
| XGBoost | 0,192 | 0,165 |
| LightGBM | 0,178 | 0,154 |
| SVM RBF | 0,190 | 0,175 |

El mejor modelo por AUC-PR en CV es la **SVM RBF**. Sobre 2021 detecta el **76,8 % de los graves** (recall) con una precisión del 10,2 %; su matriz de confusión ([[81 979, 15 694], [540, 1 787]]) revela **540 falsos negativos** —graves no detectados—, que constituyen el error de mayor costo clínico. Las Figuras 4 y 5 muestran las curvas PR y ROC del mejor modelo sobre 2021.

![Curva Precision-Recall del mejor modelo sobre 2021](../figures/curva_pr_2021.png)

**Figura 4.** Curva Precision-Recall del mejor modelo (SVM RBF) sobre 2021, frente a la línea *no-skill*.

![Curva ROC del mejor modelo sobre 2021](../figures/roc_2021.png)

**Figura 5.** Curva ROC del mejor modelo sobre 2021.

La **importancia por permutación** (Figura 6) muestra que la **edad** domina el poder predictivo (0,305 de caída en AUC-PR al permutarla), seguida del tipo de contagio (0,097), la ciudad (0,067) y el sexo (0,020). Esto es clínicamente coherente con la epidemiología del COVID-19, donde la edad es el factor de riesgo dominante, y confirma que la variable elegida para la simulación de drift (la edad) es la más relevante para el problema.

![Importancia por permutación del mejor modelo](../figures/importancia_permutacion.png)

**Figura 6.** Importancia por permutación (disminución de AUC-PR) del mejor modelo.

---

# Conclusión

La segunda entrega consolidó un conjunto de datos real colombiano (6,4 M de registros del INS, reducidos a 300 000 estratificados por ola), verificó una simulación que reproduce el diseño de Jones y Farrow, e integró todo el flujo en `Pipeline` reproducibles sin fuga de información. Se reprodujo el OCSVM del artículo —confirmando su sensibilidad al drift y su tendencia "a mayor ruido, más inliers"— y se adaptó el método al contexto local con modelos de gravedad (RF, XGBoost, LightGBM, SVM-RBF) frente a líneas base. La validación (CV estratificada, anidada y temporal) mostró que la edad es el predictor dominante y que el desempeño se degrada en las olas posteriores, ratificando la utilidad del OCSVM como monitor de drift. Queda pendiente, para la entrega final, consolidar la interpretación con técnicas de explicabilidad (SHAP) y el despliegue básico.

---

# Nota sobre el uso de IA

El equipo utilizó herramientas de inteligencia artificial generativa como apoyo para estructurar y redactar este documento, así como para construir y depurar el código reproducible, a partir de las decisiones metodológicas discutidas y tomadas por el equipo. Todas las decisiones técnicas presentadas pueden ser explicadas y sustentadas por los integrantes.

---

# Referencias

[1] Jones, W. S., & Farrow, D. J. (2025). One-class support vector machines for detecting population drift in deployed machine learning medical diagnostics. *Scientific Reports*, 15, 12157.

[2] Instituto Nacional de Salud (INS). *Casos positivos de COVID-19 en Colombia*. Datos Abiertos Colombia. Disponible en: https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia/gt2j-8ykr

[3] Schölkopf, B., et al. (2001). Estimating the support of a high-dimensional distribution. *Neural Computation*, 13(7), 1443–1471.

[4] Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
