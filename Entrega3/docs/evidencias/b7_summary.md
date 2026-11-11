# B7. Aplicación Streamlit: predicción individual

## Qué se construyó

- `app/app.py`: pestaña 1 de predicción individual.
- `src/core.py`: lógica compartida de serving (predicción y explicación).
- Artefacto `model_final.joblib` ampliado con:
  - `shap`: background, coeficientes, grupos y referencias.
  - `categories`: opciones de sexo, ciudad y fuente.

## Funcionalidad de la pestaña 1

- Formulario: edad, sexo, ciudad y fuente de contagio.
- **Riesgo calibrado** y decisión (grave / no grave) según el umbral.
- **Advertencia para menores de 60**: el modelo no detecta gravedad en ese
  grupo.
- **Control de umbral** en la barra lateral, con presets 0.05 / 0.091 / 0.20 y
  el costo FN/FP implícito.
- **Explicación en dos capas:**
  1. *Simple*: "what-if". Cambio del riesgo calibrado si cada variable tomara
     su referencia (edad mediana, sexo femenino, Bogotá, contagio comunitario).
  2. *Técnica*: contribuciones SHAP en log-odds (modelo lineal exacto).

## Verificación

- AppTest (`streamlit.testing.v1`) sin excepciones en 3 casos:

| Caso | Riesgo | Decisión | Advertencia edad |
|---|---|---|---|
| 86 M, OTROS MUNICIPIOS, COMUNITARIA | 48.2 % | Grave (alerta) | No |
| 30 F, CALI, COMUNITARIA | 0.2 % | No grave | Sí |
| 73 F, BOGOTA, RELACIONADO | 5.4 % | No grave | No |

- Reconstrucción SHAP exacta: `base + Σ contribuciones = logit` (error < 1e-8).
- Paridad del caso de referencia: 0.0.

## Notas

- Las capturas de pantalla quedan para B9 (requieren navegador); por ahora la
  evidencia funcional es el AppTest.
- Las pestañas 2 (lote + monitor) y 3 (ficha del modelo) se implementan en B8.

## Artefactos

- `app/app.py`
- `src/core.py`
- `src/build_final_artifact.py` (ampliado con `shap` y `categories`)
- `artifacts/model_final.joblib` (regenerado)
- `docs/evidencias/b7_apptest.log`
