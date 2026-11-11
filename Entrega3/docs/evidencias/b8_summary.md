# B8. App: lote + monitor y ficha del modelo

## Pestaña 2. Predicción por lotes + monitor

- Carga de CSV con plantilla descargable.
- Validación de columnas, edades inválidas y categorías no vistas.
- Predicción vectorizada con el mismo pipeline y umbral de la app.
- Métricas del lote si el CSV incluye `grave`:
  AUC-PR, ROC-AUC, Brier, recall, F1 y matriz de confusión.
- Histograma del riesgo calibrado.
- Comparación descriptiva con el entrenamiento:
  - edad media y percentiles 5/95,
  - % femenino,
  - % contagio comunitario,
  - ciudad más frecuente.
- Monitor de la fase 2: tabla OCSVM/IForest y figura `monitor_drift.png`.

**Alcance del monitor:** no se reentrena un OCSVM por lote. Se muestra el monitor
validado de la fase 2 y una comparación descriptiva del lote nuevo. Es coherente
con la decisión de reutilizar el monitor y evita prometer detección de drift por
lote sin validación.

## Pestaña 3. Ficha del modelo

- Métricas 2022: AUC-PR, ROC-AUC, Brier, recall y F1.
- Estrategia, datos de entrenamiento, calibración, test y umbral.
- Curva de fiabilidad y curva de decisión.
- Importancia global SHAP y PDP de edad.
- Tabla de errores por grupo de edad.
- Limitaciones explícitas:
  - modelo etario y sin detección de graves < 60 al umbral 0,091;
  - Brier skill negativo;
  - sobreestimación del decil superior;
  - 2022 no es test ciego;
  - el monitor vigila P(X), no P(Y|X) ni P(Y).

## Verificación

AppTest:

- Carga inicial: 0 excepciones.
- Lote de 200 casos con etiquetas: 0 excepciones.
- Se generaron 4 tablas y las métricas del lote.
- Pestañas 2 y 3 renderizan sin errores.

## Artefactos

- `app/app.py`
- `src/core.py` (`predict_frame`)
- `artifacts/model_final.joblib` (con `reference_summary`)
- `docs/evidencias/b8_apptest.log`

## Pendiente

- Capturas de pantalla en B9.
- Docker y paridad en B9.
