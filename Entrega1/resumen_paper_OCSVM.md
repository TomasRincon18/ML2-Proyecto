# Resumen del paper (no se expone al profesor)

## One-class support vector machines for detecting population drift in deployed machine learning medical diagnostics

**Jones, W. S. & Farrow, D. J. (2025).** *Scientific Reports*, 15, 12157.
DOI: https://doi.org/10.1038/s41598-025-94427-x — Centro DAIM, University of Hull, Reino Unido.

---

## 1. Problema

Los modelos de ML para diagnóstico médico se entrenan sobre poblaciones de pacientes estrechas y específicas, elegidas para que coincidan con la población donde se usarán en la práctica (*intended use*). Pero la atención en salud es no estacionaria: demografía, tecnología, estándares de cuidado y comportamiento cambian. Esto produce **population drift** (también llamado covariate shift / data drift / domain drift), definido formalmente como:

```
P_train(X) ≠ P_deploy(X)
```

es decir, la distribución de las variables de entrada cambia entre entrenamiento y despliegue. Eso degrada el desempeño y la exactitud diagnóstica del modelo, con consecuencias potencialmente letales para los pacientes.

**Causas del drift** (clasificadas por los autores):
1. **Demografía / clínica**: nuevo contexto clínico, cambios de estándar de cuidado, tratamientos, estacionalidad, nuevas enfermedades (p. ej. COVID-19).
2. **Tecnología**: nuevos dispositivos de adquisición, exámenes, software, infraestructura.
3. **Comportamiento**: incentivos, reembolsos, cambios en prácticas clínicas o del paciente; incluso el propio uso del modelo puede inducir cambios (sobreconfianza, *automation bias*) y ataques adversariales.

**Por qué es difícil detectarlo hoy**:
- La recomendación principal es monitorear la *performance/accuracy* contra un *ground truth*. Es metodológicamente ideal pero **impráctico**: exige seguir pacientes durante años y exámenes extra → caro, lento y laborioso.
- Reentrenar con datos nuevos del entorno de despliegue tampoco es realista: no hay datos etiquetados disponibles fácilmente y el modelo reentrenado necesitaría nueva aprobación regulatoria.
- Alternativas no supervisadas previas:
  - **Duckworth et al.**: tracking de valores **SHAP** (XGBoost pre/post COVID). Limitación: son *surrogate markers*, no miden drift directamente; sin umbral claro.
  - **Ackerman et al.**: tracking de *confidence scores*. Limitación: no miden drift directamente; sin umbral claro.
  - **Medidas estadísticas** (Kolmogorov-Smirnov, Wasserstein, KL divergence, Local Lipschitz): cuantificables y visualizables por variable, pero subjetivas para decidir qué es drift y difíciles de interpretar en alta dimensión.

**Lo que falta**: una técnica *no supervisada* (sin etiquetas de pacientes nuevos) que mida el drift más directamente, se aplique a pacientes individuales, capture relaciones complejas en alta dimensión, requiera poca subjetividad y sea fácil de implementar. Los autores proponen el **OCSVM** como esa base.

---

## 2. Propuesta y objetivo

Usar un **One-Class Support Vector Machine (OCSVM)** entrenado sobre el dataset de entrenamiento original del sistema de diagnóstico. El OCSVM aprende la frontera del soporte de la distribución de entrenamiento y clasifica cada paciente nuevo como:
- **Inlier** → consistente con la población de entrenamiento → se puede usar el clasificador con más confianza.
- **Outlier** → desvía de la distribución original → señal de posible *population drift* → precaución con el diagnóstico.

Es una **capa de monitoreo de la población de entrada**, no un diagnóstico directo.

---

## 3. Método

### Dataset y preprocesamiento
- **Wisconsin Breast Cancer (Diagnostic)** — WBC: 569 registros de FNA (punción aspirativa con aguja fina) de masas mamarias.
  - 30 features numéricas/continuas (geometría del núcleo celular) + 1 target binario `diagnosis`: benigno (357) / maligno (212). Sin valores nulos.
- **Preprocesamiento**:
  1. **Multicolinealidad**: correlación de Pearson; se eliminan features con |r| > 0.9 → se remueven 9 (perimeter_mean, area_mean, radius_worst, perimeter_worst, area_worst, texture_worst, concave points_mean, perimeter_se, area_se) → quedan **21** variables.
  2. **SMOTE**: sobremuestrea malignos de 212 → 357 → balance 357/357 (**714** instancias), para que ambas clases estén igualmente representadas en el espacio del OCSVM.
  3. Se **elimina** la etiqueta `diagnosis` (OCSVM es no supervisado, solo features).
  4. **Escalado** (estandarización a media 0 y desviación 1) para que ninguna variable domine por magnitud.

### Configuración del OCSVM (Tabla 1)
| Parámetro | Valor |
|---|---|
| Kernel | RBF |
| Gamma | automático (= 1 / n_features) |
| Contaminación ν | 0.01 |

- **RBF**: fronteras curvas para relaciones no lineales.
- **Gamma automático**: escala la influencia según el número de features, simplifica tuning y reduce sobreajuste.
- **ν = 0.01**: asume ~1% de outliers.
- Implementado en Python 3.9.13, scikit-learn 1.0.2, clase `OneClassSVM`.
- Base teórica: **Schölkopf et al. (2001)**, *Estimating the support of a high-dimensional distribution*.

### Simulación y evaluación
- No hay datos reales de drift disponibles → se **simula** drift.
- Se toman las filas reales con el **máximo y mínimo** de `radius_mean` y se desplazan **±0.4 desviaciones estándar** de esa feature (valores altos ≈ tumores tardíos/malignos; bajos ≈ tempranos/benignos).
- Se añade **ruido gaussiano** a todas las features en tres niveles: **5%, 10%, 30%** de la SD de la población real.
- Por nivel: 10.000 registros (5.000 debajo del mínimo + 5.000 por encima del máximo) → **30.000** en total.
- `radius_mean` se elige por su peso en la clasificación tumoral (ref. Basciftci & Ünal, 2019).
- Hipótesis: a mayor ruido, más dispersión y solapamiento → **más inliers** detectados.
- Para visualizar: se entrena un OCSVM de solo 2 variables (`radius_mean` y `compactness_mean`) y se grafica la frontera de decisión (Fig. 1).

---

## 4. Resultados

| Ruido (%) | Inliers (de 10.000) | % inliers |
|---|---|---|
| 5 | 27 | 0.27% |
| 10 | 486 | 4.86% |
| 30 | 851 | 8.51% |

**Lectura**: a mayor ruido, mayor solapamiento con la distribución de entrenamiento → más *outliers* caen dentro de la frontera (más falsos negativos). Confirma que el OCSVM es **sensible al drift**, pero que su capacidad de detección **disminuye** cuando el desplazamiento es sutil. Los resultados también son sensibles a la **distancia del centro** de la población outlier.

---

## 5. Discusión y limitaciones

**Logros / conclusión del paper**: prueba de concepto de que el OCSVM detecta drift sutil y puede hacer detección en **tiempo real** sobre pacientes individuales, **sin ground truth**. Apoya una adopción segura del diagnóstico con ML.

**Limitaciones reconocidas**:
1. **Baja explicabilidad/transparencia**: no se identifica qué feature o combinación causa la clasificación (hay trabajo incipiente integrando XAI con OCSVM).
2. **Salida binaria**: dicotomizar inlier/outlier descarta información valiosa (cita a Altman & Royston 2006; Royston et al. 2007).
3. **Solo datos simulados**: no se evalúa sobre poblaciones reales.
4. **Sin comparación** con otros one-class classifiers (Isolation Forest, Robust Covariance) ni con reducción de dimensionalidad.
5. **Simulación limitada**: no modela la **covarianza** entre features ni desplazamientos más sutiles de los centros.
6. No queda claro si funciona para **imágenes médicas** (aunque no hay razón para descartarlo, p. ej. post-filtros convolucionales).

**Trabajo futuro propuesto**:
- Evaluar en entornos clínicos y datasets reales.
- Explorar otros clasificadores one-class y métodos estadísticos/ML complementarios (Mahalanobis, autoencoders, cosine distance), y soluciones bayesianas profundas para aprendizaje continuo.
- Probar reducción de dimensionalidad.
- Modelar mejor las poblaciones outlier (covarianza, desplazamientos sutiles).
- Salidas continuas en lugar de binarias.
- Tracking/​predicción **prospectiva** del drift antes de que afecte el desempeño.
- Abordar implicaciones regulatorias.
- Extender a otros dominios más allá del diagnóstico médico.

---

## 6. Puntos clave para retener

1. **Definición**: population drift = `P_train(X) ≠ P_deploy(X)`.
2. **OCSVM**: no supervisado, aprende una frontera del soporte de la distribución; inlier/outlier; sin etiquetas.
3. **Configuración**: RBF, gamma automático, ν = 0.01.
4. **Datos**: WBC, 569 → SMOTE → 714, 21 features, estandarizadas.
5. **Simulación**: `radius_mean` ±0.4 SD, ruido 5/10/30%, 10.000 registros/nivel.
6. **Resultado**: 0.27% / 4.86% / 8.51% de inliers → sensible a drift, menos sensible a drift sutil.
7. **Valor**: capa de monitoreo en tiempo real sin ground truth, complementaria (no sustitutiva) de modelos robustos a drift.
8. **Gaps que nuestro proyecto aprovecha**: datos reales, comparación con otros métodos, simulación con covarianza, explicabilidad, salida continua.
