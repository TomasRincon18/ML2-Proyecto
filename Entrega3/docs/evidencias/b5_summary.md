# B5. Interpretabilidad: SHAP global, local y PDP

## Objetivo

Explicar el modelo final (`model_final.joblib`) en términos de las cuatro
variables originales: edad, sexo, ciudad y fuente de contagio.

## Método

- SHAP `LinearExplainer` sobre el pipeline transformado.
- Muestra: 5.000 casos de 2022; background: 1.000 casos.
- Los valores SHAP de las columnas one-hot se suman por variable original.
- Los valores están en **log-odds** del modelo base; la probabilidad calibrada
  se obtiene con el calibrador Platt.
- Base value: **-1.3566** (log-odds).

## Importancia global

| Variable | Importancia media \|SHAP\| | Lectura |
|---|---:|---|
| **Edad** | **1.664** | Domina el modelo; 4,7× la siguiente variable |
| Sexo | 0.351 | Ser hombre aumenta el riesgo |
| Ciudad | 0.218 | Efecto moderado por municipio |
| Fuente de contagio | 0.162 | Efecto pequeño |

## Casos locales

| Caso | Perfil | Riesgo calibrado | Contribuciones principales |
|---|---|---:|---|
| Bajo riesgo | 30 años, F, CALI | 0.002 | Edad negativa; sexo y ciudad pequeños |
| Alto riesgo | 86 años, M, OTROS MUNICIPIOS | 0.482 | Edad +4.43; sexo +0.44; ciudad +0.31 |
| Caso límite | 73 años, F, BOGOTA | 0.091 | Edad +3.12; sexo -0.29; ciudad -0.13 |

## Dependencia parcial de la edad

| Edad | Riesgo medio calibrado |
|---:|---:|
| 0 | 0.0002 |
| 20 | 0.0012 |
| 40 | 0.0075 |
| 60 | 0.0438 |
| 70 | 0.1008 |
| 80 | 0.2133 |
| 100 | 0.6059 |

El riesgo medio cruza el umbral **0.0909 alrededor de los 69 años**. La curva es
monótona creciente y acelerada después de los 60.

## Lectura

1. **La edad es el factor dominante**, consistente con la fase 2 y con la
   simulación de envejecimiento del OCSVM.
2. **Sexo y ciudad aportan señales secundarias**; la fuente de contagio es
   marginal.
3. El caso límite (73 años, F, Bogotá) muestra cómo la edad empuja el riesgo
   hacia el umbral y el sexo femenino lo reduce parcialmente.
4. El modelo es **aditivo e interpretable**: cada contribución se suma en
   log-odds y la probabilidad calibrada se obtiene al final.

## Cómo leer las figuras y los valores

- **Log-odds, no probabilidad.** Los SHAP están en la escala del logit del
  modelo base. No se suman para obtener una probabilidad: la probabilidad
  calibrada se obtiene aplicando el calibrador Platt al logit final. Por eso el
  waterfall muestra `f(x)` y `E[f(X)]` en log-odds, mientras el título muestra
  el riesgo calibrado.
- **Categorías de referencia del one-hot.** El pipeline usa
  `OneHotEncoder(drop='first')`. Las referencias son **F** (sexo),
  **BARRANQUILLA** (ciudad) y **COMUNITARIA** (fuente). Su contribución es cero
  por construcción; la importancia de ciudad es relativa a Barranquilla.
- **Beeswarm.** El color de sexo, ciudad y fuente usa códigos internos, no
  etiquetas legibles. La barra es la referencia principal de importancia global;
  el beeswarm sirve sobre todo para ver la dirección de la edad: edades mayores
  (rosa) empujan el riesgo hacia arriba, edades menores (azul) hacia abajo.
- **PDP.** La banda P10-P90 muestra heterogeneidad entre pacientes, no un
  intervalo de confianza del promedio. El cruce en 69 años es un promedio
  poblacional, no un corte clínico ni una regla de decisión por edad.
- **Background.** El valor base (-1.3566) se calculó con 1.000 casos de 2022;
  por eso corresponde al caso promedio de la cohorte de despliegue, no el de
  2020.
- **Casos ilustrativos.** Los tres perfiles se seleccionaron de una muestra de
  5.000 casos; no agotan todos los perfiles posibles.
- **Estabilidad.** La importancia de sexo, ciudad y fuente es pequeña; su
  estabilidad se cuantifica con bootstrap en B6.

## Limitaciones

- SHAP explica el **ranking/log-odds**, no la probabilidad calibrada directamente.
- La importancia por grupos depende de la categoría de referencia del one-hot.
- El modelo es lineal y aditivo: no hay interacciones que SHAP pueda revelar.
- Persiste la sobreestimación del decil superior (predicho 0.217 vs observado
  0.085 en 2022).
- La interpretación es asociativa, no causal.

## Artefactos

- `artifacts/shap_summary.json`
- `artifacts/shap_values.npz`
- `figures/shap_global_bar.png`
- `figures/shap_beeswarm.png`
- `figures/shap_local_bajo_riesgo.png`
- `figures/shap_local_alto_riesgo.png`
- `figures/shap_local_caso_límite.png`
- `figures/pdp_edad.png`
- `docs/evidencias/b5_shap.log`
