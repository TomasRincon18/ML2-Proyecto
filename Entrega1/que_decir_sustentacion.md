# Qué decir en la sustentación (guía rápida)

El plan: paper, datos, problema real, modelos. Frases cortas, sin enrollarse.

---

## 1. El paper

**Lo que dices**
El paper propone usar un One-Class SVM (OCSVM) para detectar population drift: cuando la población de pacientes que recibe el modelo ya no se parece a la que tenía cuando se entrenó. Lo bueno es que no necesita etiquetas (ground truth), porque es no supervisado: aprende la frontera de los datos de entrenamiento y lo que queda afuera es un posible drift.

**Técnico (lo importante)**
- Drift = `P_train(X) ≠ P_deploy(X)`: cambia la distribución de las variables entre entrenar y desplegar.
- OCSVM: detección de novedad/outliers, aprende una frontera alrededor de los datos; dentro = inlier, fuera = outlier.
- Config: kernel RBF, gamma automático, ν = 0.01 (~1% de outliers).
- Entrenado con Wisconsin Breast Cancer: 569 pacientes, SMOTE para balancear hasta 714, y tras quitar variables correlacionadas y escalar, 21 features.
- Para evaluarlo simularon drift moviendo `radius_mean` ±0.4 desviaciones, con ruido de 5/10/30%.

---

## 2. Parámetros y comparación del OCSVM

**Lo que dices**
El paper no compara el OCSVM contra otros métodos, eso lo deja como limitación. Y los parámetros son solo tres: kernel RBF, gamma automático y ν = 0.01.

**Técnico (lo importante)**
- No hay comparación con otros detectores (Isolation Forest, Robust Covariance, etc.): el paper evalúa solo el OCSVM y reconoce que faltó. Ese hueco es justo lo que vamos a llenar en el proyecto.
- Kernel RBF: fronteras curvas, capta relaciones no lineales.
- Gamma automático (1/n_features): simplifica el ajuste y evita sobreajuste.
- ν = 0.01: fracción esperada de outliers, más o menos el 1%.

---

## 3. Los resultados del paper

**Lo que dices**
Simularon poblaciones desplazadas con distintos niveles de ruido y vieron que, a más ruido, más pacientes nuevos caían dentro de la frontera (los tomaba como normales). O sea: detecta bien el drift claro, pero se le escapan los cambios sutiles.

**Técnico (lo importante)**
- Ruido 5%: 0.27% inliers. 10%: 4.86%. 30%: 8.51%.
- Más ruido = más solape con la población original = más falsos negativos.
- Lo que deja el paper: es un monitor útil, no una prueba definitiva, y su gran limitación es que no dice qué variable causó el outlier.

---

## 4. Nuestra base de datos

**Lo que dices**
Usamos los casos positivos de COVID-19 en Colombia del INS (datos.gov.co), varios millones de registros. La variable objetivo es la gravedad clínica (binaria), armada agrupando el estado del caso: grave/fallecido contra leve/moderado/recuperado. Predictores: edad, sexo, departamento, tipo de contagio, tipo de atención, etc.

**Técnico (lo importante)**
- Dataset real y verificable, supera de sobra el mínimo de 5.000 registros.
- Objetivo binario derivado de `Estado`.
- Estrategia: datos reales más una simulación fundamentada (moviendo `edad` ±0.4 SD con ruido de 5/10/30%, copiando el diseño del paper como control experimental).

---

## 5. Cómo lo planteamos como problema real

**Lo que dices**
Lo adaptamos al COVID colombiano: la pandemia es un caso natural de drift porque la población cambió con las variantes (Delta, Ómicron) y la vacunación. Entonces entrenamos con la primera ola (2020) y vemos si los pacientes de las olas siguientes siguen pareciéndose. Aparte, un OCSVM monitorea si lo que llega es comparable a lo de entrenamiento.

**Técnico (lo importante)**
- Partición temporal, no aleatoria: entrenar con la primera ola (2020, sin vacunación); evaluar con Delta/Ómicron (2021-2022, vacunación en curso). Es un drift documentado, no inventado.
- Dos piezas independientes: (1) modelo predictivo de gravedad, (2) OCSVM de monitoreo entrenado solo con las variables de entrada (sin la etiqueta).

---

## 6. Modelos que vamos a usar

**Lo que dices**
Para predecir gravedad: Random Forest, Boosting (XGBoost o LightGBM) y una SVM con kernel RBF. Para el monitoreo de drift, el OCSVM. Como línea base, un clasificador de mayoría y una regresión logística simple.

**Técnico (lo importante)**
- Métrica principal: F1-score y AUC-PR de la clase minoritaria (los graves), por el desbalance y porque el error caro es el falso negativo.
- Validación: cruzada estratificada k=5 dentro de entrenamiento; evaluación fuera de muestra en las olas posteriores; sin fuga (Pipeline).
- OCSVM: RBF, ν = 0.01 como punto de partida, calibrable; se mide la proporción de inliers/outliers.

---

## Cierre (una frase si falta tiempo)

Detectamos si un modelo de gravedad de COVID sigue siendo válido cuando llegan pacientes distintos, usando un OCSVM de monitoreo más un modelo predictivo, con datos reales del INS y una simulación controlada.
