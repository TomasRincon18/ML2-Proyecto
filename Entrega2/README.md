# Proyecto Integrador — Machine Learning II (Entrega 2)

**Detección de *population drift* en modelos de predicción de gravedad clínica de COVID-19 en Colombia.**

Reproduce y adapta el método de Jones & Farrow (2025): un *One-Class SVM* (OCSVM) como capa de monitoreo del *population drift*, aplicado a un modelo de gravedad clínica sobre datos reales del INS (Colombia).

## Estructura

```
Entrega2/
├── data/
│   ├── raw/casos_covid_colombia.csv        # dataset crudo (no versionado, ~514 MB)
│   └── processed/dataset_reducido.csv      # dataset consolidado (300 000 filas)
├── notebooks/
│   ├── 01_reduccion.ipynb                  # consolidación, limpieza y muestreo
│   └── 02_modelado.ipynb                   # pipeline, simulación, OCSVM, modelos, validación
├── src/descargar_ins.py                    # descarga el dataset crudo desde datos.gov.co
├── figures/                                # figuras generadas por los notebooks
├── informe/informe_entrega2.md             # informe de la entrega
└── requirements.txt
```

## Reproducción

1. Instalar dependencias: `pip install -r requirements.txt`.
2. Descargar el dataset (si no está en `data/raw/`): `python src/descargar_ins.py`.
3. Ejecutar `notebooks/01_reduccion.ipynb` (genera `dataset_reducido.csv`).
4. Ejecutar `notebooks/02_modelado.ipynb` (modelado y validación).

Semilla fija `SEED = 42` en todo el flujo.

## Nota sobre los datos

`data/raw/casos_covid_colombia.csv` (~514 MB) no se versiona (ver `.gitignore`). Generarlo con `src/descargar_ins.py` desde el portal oficial de datos abiertos del INS (recurso `gt2j-8ykr`, licencia CC BY-SA 4.0).
