# Guion de sustentación. Entrega final

**Duración objetivo:** 10 a 12 minutos, más preguntas.  
**Formato:** `slides/index.html` (reveal.js, funciona sin internet).  
**Reparto:** Ailyn presenta las diapositivas 1 a 6; Juan Tomás presenta las 7 a 12. Los dos deben poder responder por todo el proyecto.

## Reparto y tiempos

| Tramo | Diapositivas | Quién | Tiempo |
|---|---|---|---|
| Portada, problema, artículo y datos | 1-4 | Ailyn | 4 min |
| Protocolo, réplica OCSVM y modelo final | 5-7 | Ailyn | 3 min |
| Calibración, interpretabilidad y errores | 8-10 | Juan Tomás | 3 min |
| Despliegue, limitaciones y cierre | 11-12 | Juan Tomás | 2 min |

## Diapositiva 1. Portada

No empezar leyendo la lista de integrantes. Abrir con el contraste:

> "El proyecto empezó con un monitor para detectar si cambia la población de pacientes. Al final, el hallazgo fue otro: la población de entrada se mantuvo estable, pero cambió la prevalencia y cambió la relación entre perfil y gravedad."

Señalar las tres cifras de la portada: AUC-PR 0,190, 99 % de inliers del OCSVM y 300.000 casos analizados.

**Transición:** "Para entender por qué esas tres cifras conviven, hay que separar tres tipos de cambio."

## Diapositiva 2. El problema

Explicar las tres tarjetas con un ejemplo:

> "P(X) es quién llega: edad, sexo, ciudad, fuente de contagio. P(Y|X) es qué tan grave se pone ese paciente. P(Y) es qué proporción de los casos termina en gravedad."

Cerrar con la frase del recuadro:

> "En Colombia, P(X) se mantuvo estable. La prevalencia cayó de 2,82 % a 1,10 % y también cambió P(Y|X). El monitor de P(X) no podía ver esos cambios."

**Transición:** "Ese es justamente el alcance del artículo que tomamos como base."

## Diapositiva 3. El artículo base

Resumir el método en dos frases:

> "Jones y Farrow entrenan un OCSVM con la población de referencia. Si un caso cae fuera del soporte, lo marcan como posible drift. No usan la etiqueta, así que no pueden ver cambios de prevalencia ni de P(Y|X)."

No entrar en la matemática del kernel salvo que pregunten. La tarjeta de la derecha resume el flujo: P(X) de referencia, inlier, outlier.

**Transición:** "Con ese método, pasamos a los datos del INS."

## Diapositiva 4. Datos

> "Trabajamos con 6,39 millones de registros del INS. Después de depurar quedaron 6,31 millones. Para que el cómputo fuera viable y el balance temporal fuera estricto, tomamos 100.000 casos por ola."

Señalar las barras de prevalencia: 2,82 %, 2,33 % y 1,10 %. Es el dato que explica por qué las métricas tradicionales se ven peor en 2022.

Mencionar las cuatro variables y que se excluyó tipo de atención por fuga de información.

**Transición:** "La partición temporal es la que evita que el modelo vea el futuro."

## Diapositiva 5. Protocolo

> "2020 entrena, 2021 calibra y 2022 evalúa. La separación ocurre antes de cualquier transformación. El Pipeline ajusta el escalamiento y el one-hot solo con el train de cada partición."

Aclarar que el modelo final usa 2020 completo más el 80 % de 2021, y que el 20 % restante de 2021 se usa para calibrar. 2022 no se toca.

**Transición:** "Con ese protocolo, primero probamos si el OCSVM detectaba el cambio."

## Diapositiva 6. Réplica del OCSVM

> "En 2020 test, 2021 y 2022 el OCSVM deja entre 98 % y 99 % de inliers. La población de entrada se mantiene dentro del soporte."

Contrastar con la simulación:

> "Con 30 % de ruido, los inliers bajan a 72 %. Isolation Forest se queda en 100 %. El monitor sí detecta el drift controlado, pero no el cambio real de prevalencia."

**Transición:** "Entonces el problema no estaba en P(X). Estaba en el clasificador y en su calibración."

## Diapositiva 7. Modelo final

> "El modelo final es una regresión logística reentrenada con 2020 y el 80 % de 2021, calibrada con Platt sobre el 20 % restante. El umbral 0,0909 sale de asumir que un falso negativo cuesta diez veces más que un falso positivo."

Leer solo las métricas principales: AUC-PR 0,190, ROC-AUC 0,923, Brier 0,0119, F1 0,147. Aclarar que el AUC-PR es 17,3 veces la prevalencia.

**Transición:** "El siguiente problema era que el modelo crudo no se podía leer como riesgo."

## Diapositiva 8. Calibración

> "El modelo crudo predice una media de 0,287 cuando la prevalencia es 0,011. El ranking sirve, pero el número no se puede leer como riesgo. Platt corrige casi toda esa descalibración."

Mencionar que EM falló en 2021 y 2022. Beta calibration mejoró Brier, pero bajó F1 y no corrigió el decil superior. Se mantuvo Platt.

Cerrar con el recuadro:

> "El Brier skill es -0,098: el valor está en el ranking y en la decisión, no en batir la tasa base."

**Transición:** "Si el modelo ordena bien, la siguiente pregunta es qué variable manda."

## Diapositiva 9. Interpretabilidad

> "La edad pesa 1,66 en |SHAP|, 4,7 veces más que la siguiente variable. El bootstrap confirma que la dominancia es estable."

Señalar el PDP:

> "El umbral 0,091 se cruza hacia los 69 años. A los 80 el riesgo medio es 0,213."

Usar un caso: "Un hombre de 86 años de otros municipios tiene riesgo 0,482; la edad aporta casi toda la explicación."

**Transición:** "Pero un modelo que depende tanto de la edad tiene un costo en los grupos jóvenes."

## Diapositiva 10. Errores y utilidad

Decirlo sin rodeos:

> "Al umbral 0,091 el modelo no detecta ningún caso grave menor de 60 años. La sensibilidad global viene de los mayores de 60. La curva de decisión muestra que el umbral está en el borde: con 0,05 el beneficio neto se vuelve positivo."

Conclusión práctica:

> "Es una herramienta de triage para mayores de 60, no un tamizaje poblacional."

**Transición:** "Aun con esa limitación, el sistema se puede usar y auditar."

## Diapositiva 11. Despliegue

> "La app tiene tres pestañas: predicción individual, lote con monitor y ficha del modelo. La predicción individual muestra el riesgo calibrado, una advertencia por edad, el control de umbral y dos explicaciones: what-if en probabilidad y SHAP en log-odds."

Mencionar Docker:

> "El contenedor pasó el healthcheck, la paridad fue 0,0 y la predicción de 86 años coincidió con la app en host: 24,6 %."

**Transición:** "Cierro con lo que el proyecto deja y lo que falta."

## Diapositiva 12. Cierre

> "2022 no es un test ciego. Faltan comorbilidades, vacunación y variante. El modelo es etario y el Brier skill es negativo. El OCSVM solo vigila P(X)."

Proyección: holdout temporal ciego, más covariables, modelos no lineales calibrados, monitoreo de P(Y|X) y MLOps.

Cerrar con la idea del recuadro:

> "El OCSVM vigila P(X). El cambio que importaba estaba en P(Y) y P(Y|X)."

Pasar a la demo o a preguntas.

## Preguntas probables

**¿Por qué regresión logística y no XGBoost o LightGBM?**  
Entrenamos los modelos avanzados en la fase 2. La regresión logística empató en AUC-PR y fue más estable e interpretable. El modelo final se eligió por desempeño y parsimonia, no por comodidad.

**¿Cómo evitaron fuga de datos?**  
La partición temporal se hizo antes de cualquier transformación. El escalamiento y el one-hot se ajustaron dentro del Pipeline, solo con el train de cada fold. El test se usó una vez para reportar.

**¿Por qué el OCSVM da 99 % de inliers si hubo pandemia?**  
Porque vigila P(X), no P(Y|X). La edad, el sexo, la ciudad y la fuente de contagio no cambiaron tanto. Lo que cambió fue la probabilidad de gravedad, por vacunación y variantes.

**¿Entonces el OCSVM no sirve?**  
Sirve para lo que fue diseñado: certificar que la población de entrada sigue dentro del soporte. No sirve para detectar cambios de prevalencia ni de calibración. El proyecto muestra ese límite y añade la capa supervisada.

**¿Por qué el umbral 0,0909?**  
Porque iguala el costo esperado de un falso negativo con diez veces el de un falso positivo. Si el costo clínico es mayor, el umbral debe bajar. La app permite ajustarlo y la curva de decisión muestra el efecto.

**¿Por qué no detecta graves en jóvenes?**  
Porque la edad domina y el umbral está alto para ese grupo. Es una limitación real del modelo y de los datos disponibles. No se debe usar para descartar gravedad en menores de 60.

**El Brier skill es negativo. ¿El modelo sirve?**  
Sirve para ordenar casos y para decidir con un umbral, no para reducir el error cuadrático frente a la tasa base. El AUC-PR es 17,3 veces la prevalencia. La calibración es razonable, pero el decil superior sigue sobreestimado.

**¿Cómo garantizan la reproducibilidad?**  
Semilla 42, versiones fijadas, Pipeline, artefacto con caso de referencia, paridad 0,0, AppTest y Docker con healthcheck. El repositorio tiene el código y las instrucciones.

## Demo

1. Abrir `http://localhost:8501`.
2. Pestaña 1: edad 86, mujer, Bogotá, comunitaria. Mostrar riesgo y decisión.
3. Cambiar edad a 30. Mostrar la advertencia.
4. Pestaña 3: mostrar métricas, calibración y la tabla por edad.
5. Si la demo falla, usar las capturas de `docs/evidencias/` o el PDF del deck.

## Checklist de rúbrica

| Criterio | Evidencia |
|---|---|
| Modelo final y reproducción | Diapositiva 7, informe §1, `model_final.joblib` |
| Interpretación | Diapositiva 9, informe §3, SHAP y PDP |
| Despliegue | Diapositiva 11, app, Docker, capturas |
| Calidad y limitaciones | Diapositiva 12, informe §5, Brier skill, modelo etario |
| Comunicación | Informe de 5 páginas, deck de 12 diapositivas y este guion |

## Ensayo

- Cronometrar dos veces. Ailyn debe cerrar su tramo en 7 minutos; Juan Tomás en 5.
- Practicar la transición de la diapositiva 7 a la 8.
- Tener abiertas la app y las capturas por si falla la red.
- Revisar que el proyector no recorte las slides 16:9.
