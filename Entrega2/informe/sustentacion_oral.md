# Guion y Estructura para la Sustentación Oral — Segunda Entrega (Fase 2)

**Proyecto Integrador:** Detección de *Population Drift* en modelos de predicción de gravedad clínica de COVID-19 en Colombia  
**Asignatura:** Machine Learning II — Universidad Externado de Colombia  
**Integrantes:** Ailyn Sofía Gómez Rodríguez, Juan Tomás Rincón Pinzón  
**Docente:** Wilmer Darío Pineda Ríos  

---

## 🎯 Objetivo de la Sustentación (10 a 12 minutos)

Defender con solvencia técnica y estadística:
1. La consolidación y calidad de los datos del INS (6,3 M depurados a 300 k).
2. El diseño fundamentado de la simulación de drift de envejecimiento y la prevención estricta de fuga de datos en el `Pipeline`.
3. La réplica del paper de Jones & Farrow (2025) y su adaptación a Colombia con OCSVM.
4. Los resultados de modelamiento, el empate estadístico en Validación Cruzada Anidada y la justificación de la Regresión Logística.
5. El hallazgo metodológico clave: **Prevalence Drift vs. Covariate Drift**.

---

## 📊 Estructura Sugerida de Diapositivas

### Diapositiva 1: Portada y Contexto
- **Título:** Detección de *Population Drift* en predicción de gravedad de COVID-19 en Colombia.
- **Autores:** Ailyn Sofía Gómez Rodríguez y Juan Tomás Rincón Pinzón.
- **Mensaje Clave:** Adaptamos el método de Jones & Farrow (2025) (*One-Class SVM*) para monitorear el soporte poblacional de pacientes colombianos ante cambios temporales y epidemiológicos.

### Diapositiva 2: Datos, Calidad y Prevención de Fuga
- **Fuente:** 6 390 844 registros crudos del INS (datos abiertos).
- **Depuración:** Exclusión de nulos en estado (0,65 %) y restricción a 2020–2022 (excluyendo fase endémica) $\to$ 6 311 405 casos.
- **Muestreo:** 100 000 casos estratificados por ola (2020: 2,82 % graves, 2021: 2,33 %, 2022: 1,10 %).
- **Fuga evitada:** Preprocesador (`StandardScaler` + `OneHotEncoder`) ajustado estrictamente dentro del train de cada fold; exclusión de `"tipo de atención"` por contener el desenlace clínico.

### Diapositiva 3: Simulación Fundamentada en Colombia
- **Diseño:** Desplazamiento de edad anclado en el percentil 99 ($85\text{ años} + 0,4\sigma \approx 92,2\text{ años}$) con ruidos gaussianos de 5 %, 10 % y 30 % de $\sigma$.
- **Justificación clínica:** La letalidad por COVID-19 en Colombia crece drásticamente con la edad (de 0,1 % en jóvenes a >30 % en ancianos); un drift de envejecimiento es el escenario de mayor riesgo de error clínico.
- **Verificación:** Mostrar gráfico de distribución real vs simulada (Figura 2).

### Diapositiva 4: Réplica del Paper y Monitoreo OCSVM
- **Réplica Wisconsin:** OCSVM ($\nu=0,01$, RBF) reproduce fielmente la tendencia del paper: a mayor ruido, más inliers por solapamiento con la frontera.
- **Monitoreo en Colombia:**
  - Tasa de falsa alarma *out-of-sample* en 2020 Test (95k casos) = 1,04 % (98,96 % inliers), exactamente calibrada a $\nu = 0,01$.
  - En 2021 (99,08 %) y 2022 (97,99 %) el OCSVM detecta que el soporte de $X$ es estable.
  - OCSVM vs Isolation Forest: OCSVM detecta el corrimiento (inliers caen a 72 % en Sim 30 %), mientras que Isolation Forest es insensible (100 % inliers).

### Diapositiva 5: Comparación de Modelos y Validación Anidada
- **Métrica Principal:** AUC-PR (Average Precision) y ROC-AUC por desbalance severo.
- **Resultados CV (5 folds):**
  - Regresión Logística: $\text{AUC-PR} = 0,268 \pm 0,015$, $\text{ROC-AUC} = 0,909 \pm 0,003$.
  - LightGBM: $\text{AUC-PR} = 0,266 \pm 0,016$.
  - XGBoost: $\text{AUC-PR} = 0,265 \pm 0,020$.
  - Random Forest: $\text{AUC-PR} = 0,252 \pm 0,014$.
  - SVM-RBF Exacto: $\text{AUC-PR} = 0,252 \pm 0,029$.
- **Validación Cruzada Anidada (5 outer x 3 inner):** Demuestra que no hay sobreajuste de hiperparámetros (Nested AUC-PR de Regresión Logística $= 0,2662 \pm 0,0143$).
- **Selección:** Se elige la Regresión Logística por principio de parsimonia, interpretabilidad clínica directa y estabilidad.

### Diapositiva 6: Evaluación Temporal y Hallazgo Metodológico Clave
- **El Dilema:** En 2022 el $F_1$ aparente cae, pero el ROC-AUC aumenta a 0,921 y el ratio AUC-PR/prevalencia salta de $8,6\times$ a $16,9\times$.
- **Explicación:** La caída de métricas tradicionales no es por falla del clasificador ni por drift de covariables ($P(X)$ no cambió según Kolmogorov-Smirnov, $D=0,06$), sino por la **caída de la tasa base de gravedad ($2,82\% \to 1,10\%$)** gracias a la vacunación.
- **Límite del OCSVM:** Al ser no supervisado y ciego a $Y$, el OCSVM no detecta cambios en la prevalencia clínica $P(Y)$, delimitando cuándo se requiere monitoreo de calibración supervisada.

### Diapositiva 7: Conclusiones y Proyección a la Entrega Final
- **Logros Fase 2:** Flujo reproducible consolidado, réplica validada, monitoreo de drift caracterizado y modelo supervisado evaluado con rigor.
- **Próximos Pasos (Fase 3 / Entrega Final):**
  1. Interpretabilidad individual y global profunda con SHAP (valores de Shapley).
  2. Recalibración Bayesiana de probabilidades para compensar el cambio de prevalencia.
  3. Despliegue funcional e interactivo en Streamlit con explicaciones visuales.

---

## 🎙️ Preguntas Típicas del Profesor y Respuestas Clave

1. **¿Por qué la Regresión Logística si el curso es de Machine Learning Avanzado?**
   > *Respuesta:* Entrenamos y optimizamos todos los modelos avanzados del curso (XGBoost, LightGBM, Random Forest y SVM con kernel RBF). En los datos con 4 covariables clave, los modelos avanzados empatan estadísticamente con la Regresión Logística ($\text{AUC-PR } 0,268 \text{ vs } 0,266$, diferencia menor a la desviación estándar entre folds $\pm 0,015$). Aplicando rigor metodológico y el principio de parsimonia médica, se selecciona el modelo más interpretable y estable.

2. **¿Cómo garantizaron que no hubo fuga de datos (*data leakage*)?**
   > *Respuesta:* Separamos temporalmente las cohortes antes de cualquier cómputo. Todas las transformaciones (`StandardScaler`, `OneHotEncoder`) se integraron en un `Pipeline` ajustado únicamente con los datos de entrenamiento de cada fold. Además, eliminamos `"tipo de atención"` para no filtrar información del desenlace.

3. **¿Por qué el OCSVM dio 99 % de inliers en 2021 y 2022 si hubo pandemia y variantes?**
   > *Respuesta:* Porque el OCSVM monitorea el espacio de entrada $X$ (edad, sexo, ciudad, contagio), cuyas distribuciones demográficas no cambiaron sustancialmente ($D_{KS} \le 0,06$). Lo que cambió fue la biología y la vacunación, que redujeron la probabilidad de gravedad dado el mismo perfil ($P(Y|X)$ y $P(Y)$). El OCSVM cumplió su función de certificar que la población de entrada no se salió del soporte conocido.
