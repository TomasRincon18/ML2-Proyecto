# Proyecto Integrador, Machine Learning II (Entrega Final)

**Detección de *population drift* en la predicción de gravedad clínica de COVID-19 en Colombia.**

- **Integrantes:** Ailyn Sofía Gómez Rodríguez, Juan Tomás Rincón Pinzón
- **Docente:** Wilmer Darío Pineda Ríos
- **Programa:** Ciencia de Datos, Universidad Externado de Colombia (2026)

La entrega final reúne el modelo definitivo, su calibración, la interpretación
con SHAP y el despliegue en Streamlit con Docker. El dataset consolidado y el
monitor OCSVM vienen de la Entrega 2.

---

## Estructura

```text
Entrega3/
├── app/
│   └── app.py                 # Streamlit: predicción, lote + monitor, ficha
├── src/
│   ├── config.py              # rutas, semilla, variables y umbrales
│   ├── data.py                # carga, validación y partición temporal
│   ├── pipeline.py            # ColumnTransformer + regresión logística
│   ├── train_final.py         # smoke test y modelo base (B1-B2)
│   ├── calibration.py         # Platt, EM, curvas de fiabilidad y ECE
│   ├── final_model.py         # FinalModel + PlattCalibrator + BetaCalibrator
│   ├── build_final_artifact.py# reentreno + calibración + artefacto final
│   ├── core.py                # serving: predicción, SHAP local y what-if
│   ├── check_parity.py        # verificación de paridad
│   ├── apptest.py             # prueba funcional Streamlit
│   └── experiments_*.py       # B3-B6: calibración, SHAP, errores, monitor
├── artifacts/                 # model_final.joblib, métricas y JSON de análisis
├── figures/                   # figuras finales
├── docs/evidencias/           # capturas, logs y resúmenes por bloque
├── scripts/docker_check.sh    # build + run + healthcheck + paridad + AppTest
├── Dockerfile
├── .streamlit/config.toml
├── requirements.txt           # análisis completo
├── requirements-app.txt       # runtime de la app (Docker)
├── informe/                   # informe final y PDF
├── slides/                    # deck reveal.js, PDF de respaldo y guion
├── notebooks/                 # cuaderno narrativo
├── data/README.md             # apunta al dataset de la Entrega 2
└── README.md
```

## Datos

El dataset consolidado está en la Entrega 2:

```text
../Entrega2/data/processed/dataset_reducido.csv
```

`src/config.py` lo resuelve por defecto. Si se mueve, define:

```bash
export ML2_DATA_PATH=/ruta/al/dataset_reducido.csv
```

El CSV tiene 300.000 registros: 100.000 por ola (2020, 2021, 2022), con
prevalencia de gravedad de 2,82 %, 2,33 % y 1,10 % respectivamente.

## Reproducción

### 1. Verificación rápida (B1)

```bash
cd Entrega3
python src/train_final.py --smoke
```

Debe imprimir el número de filas, las columnas transformadas y las métricas del
smoke test, sin errores.

### 2. Modelo base (B2)

```bash
python src/train_final.py
```

Genera el modelo base de 2020 y su evaluación 2021/2022:

- `artifacts/model.joblib`
- `artifacts/metrics.json`

### 3. Modelo final (B4)

```bash
python src/build_final_artifact.py
python src/check_parity.py
```

Genera `artifacts/model_final.joblib` con:

- reentreno 2020 + 80 % de 2021,
- calibración Platt sobre el 20 % de 2021,
- umbral 0.0909 (FN/FP = 10),
- metadatos SHAP y categorías para la app.

La paridad esperada es `0.00e+00`.

### 4. Aplicación Streamlit

El entorno local está en `.venv` con `virtualenv --system-site-packages`
(necesario porque el Python del sistema está gestionado por PEP 668 y no trae
`ensurepip`). Streamlit 1.67.0 quedó instalado allí.

```bash
.venv/bin/streamlit run app/app.py
```

Abre `http://localhost:8501`.

Si se recrea el entorno:

```bash
virtualenv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

### 5. Prueba funcional de la app

```bash
.venv/bin/python src/apptest.py
```

Verifica predicción individual, advertencia por edad y lote sintético con
etiquetas. Debe terminar en `APP TEST OK`.

### 6. Docker

```bash
bash scripts/docker_check.sh
```

o, si el socket de Docker requiere permisos:

```bash
sudo bash scripts/docker_check.sh
```

El script construye la imagen `ml2-entrega3:latest`, ejecuta el contenedor en
`localhost:8501`, verifica el healthcheck, la paridad y el AppTest.

**Verificado:** el contenedor `ml2-entrega3` se construyó y ejecutó con `sudo`.
`/_stcore/health` responde `ok`, la app abre en `localhost:8501` y la predicción
de 86 años coincide con la app en host (24.6 %). Captura:
`docs/evidencias/streamlit_docker.png`.

Para detenerlo:

```bash
sudo docker stop ml2-entrega3
sudo docker rm ml2-entrega3
```

### 7. Informe y diapositivas

```bash
python src/build_pdf.py
```

Genera `informe/informe_final.pdf` (5 páginas). Para la sustentación, abre
`slides/index.html` en el navegador. El deck funciona sin internet, tiene un PDF
de respaldo en `slides/slides.pdf` y el guion está en `slides/guion.md`.

### Evidencia visual

Las capturas están en `docs/evidencias/`:

- `streamlit_inicio.png`
- `streamlit_prediccion_alto.png`
- `streamlit_prediccion_bajo.png`
- `streamlit_lote.png`
- `streamlit_ficha_modelo.png`

## Decisiones metodológicas

| Elemento | Decisión |
|---|---|
| Entrenamiento base | 2020 (ola original) |
| Modelo final | Reentreno 2020 + 80 % de 2021; Platt sobre el 20 % de 2021 |
| Test final | 2022 (Ómicron) |
| Modelo | Regresión logística, `C=0.1`, `class_weight='balanced'` |
| Umbral | 0.0909 (FN/FP = 10); alternativa 0.05 para mayor sensibilidad |
| Métrica principal | AUC-PR (Average Precision) |
| Métricas secundarias | ROC-AUC, Brier, F1, precisión, recall |
| Monitor | OCSVM / Isolation Forest de la fase 2 (P(X)) |
| Semilla | `SEED = 42` |
| Fuga | Pipeline ajustado solo con train; test una sola vez |

## Estado

- [x] B1: estructura, configuración, pipeline y smoke test.
- [x] B2: modelo final, umbral y evaluación 2021/2022.
- [x] B3: calibración I (estático, Platt de origen y EM).
- [x] B4: calibración II y modelo final (reentreno + Platt, umbral 0.0909; beta calibration probada y descartada).
- [x] B5: SHAP global/local y PDP de edad.
- [x] B6: estabilidad, errores por subgrupo, curva de decisión y monitor reutilizado.
- [x] B7: app Streamlit, predicción individual con explicación en dos capas.
- [x] B8: app, lote + monitor y ficha del modelo.
- [x] B9: Dockerfile, paridad, README y capturas (contenedor verificado con `sudo`).
- [x] B10: informe I (secciones 1-3, figuras y cuadros).
- [x] B11: informe II (secciones 4-5, conclusión y PDF de 5 páginas).
- [x] B12: deck reveal.js, PDF de respaldo, guion y checklist.

## Entregables finales

| Entregable | Ruta |
|---|---|
| Informe final | `informe/informe_final.pdf` |
| Código y artefacto | `src/`, `artifacts/model_final.joblib` |
| App Streamlit | `app/app.py` |
| Docker | `Dockerfile`, `scripts/docker_check.sh` |
| Deck | `slides/index.html`, `slides/slides.pdf` |
| Guion | `slides/guion.md` |
