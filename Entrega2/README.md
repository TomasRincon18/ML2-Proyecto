# Proyecto Integrador — Machine Learning II (Segunda Entrega)

**Detección de *Population Drift* en modelos de predicción de gravedad clínica de COVID-19 en Colombia.**

- **Integrantes:** Ailyn Sofía Gómez Rodríguez, Juan Tomás Rincón Pinzón  
- **Docente:** Wilmer Darío Pineda Ríos  
- **Facultad:** Ciencia de Datos — Universidad Externado de Colombia (2026)  

Este repositorio reproduce y adapta el método de **Jones & Farrow (2025)** (*Scientific Reports*): implementación de un *One-Class Support Vector Machine* (OCSVM) como monitor de *population drift* en las covariables de entrada ($P(X)$), aplicado a la predicción de gravedad clínica sobre 300 000 casos del Instituto Nacional de Salud (INS).

---

## 📁 Estructura del Repositorio

```
Entrega2/
├── data/
│   ├── raw/                              # Datos crudos del INS (no versionados por tamaño, ~514 MB)
│   └── processed/dataset_reducido.csv    # Dataset consolidado y estratificado (300 000 registros, ~28.7 MB)
├── notebooks/
│   ├── 01_reduccion.ipynb                # Consolidación, limpieza, agrupamiento municipal y muestreo
│   └── 02_modelado.ipynb                 # Pipeline, simulación, OCSVM, modelos, Nested CV, evaluación temporal
├── src/
│   ├── descargar_ins.py                  # Descarga determinista del dataset oficial del INS vía API SODA
│   ├── ejecutar_experimentos.py          # Script integral para ejecutar la suite completa de modelamiento y generar figuras
│   ├── build_pdf.py                      # Compilador de informe Markdown a PDF con formato académico (4-5 págs)
│   └── resultados_completos.json         # Métricas detalladas, variabilidad (media +- std) y parámetros óptimos
├── figures/                              # Figuras en alta resolución generadas por los experimentos
│   ├── verificacion_estructura.png
│   ├── verificacion_simulacion.png
│   ├── sensibilidad_drift.png
│   ├── curva_pr_2021.png
│   ├── roc_2021.png
│   └── importancia_permutacion.png
├── informe/
│   ├── informe_entrega2.md               # Informe técnico estructurado
│   ├── informe_entrega2.pdf              # PDF compilado conforme a las directrices de la guía (5 páginas exactas)
│   └── sustentacion_oral.md              # Guion y estructura de diapositivas para la sustentación oral
└── requirements.txt                      # Dependencias del proyecto
```

---

## 🚀 Instrucciones de Reproducción

### 1. Preparar el Entorno Virtual

Se recomienda utilizar Python 3.11 o 3.12 y crear un entorno virtual aislado:

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# En Windows (PowerShell):
.venv\Scripts\Activate.ps1
# En Linux / macOS:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Obtención de Datos

El dataset consolidado ya se encuentra disponible en `data/processed/dataset_reducido.csv` (300 000 filas: 100 000 por cada ola epidemiológica 2020, 2021 y 2022).

Si se desea volver a descargar y procesar los más de 6,3 millones de registros crudos del INS:
```bash
# Descarga desde datos.gov.co
python src/descargar_ins.py

# Ejecución de la reducción y muestreo
jupyter nbconvert --to notebook --execute --inplace notebooks/01_reduccion.ipynb
```

### 3. Ejecución del Modelamiento y Validación

Para ejecutar todos los modelos (Regresión Logística, Random Forest, XGBoost, LightGBM, SVM-RBF), el OCSVM, la Validación Cruzada Anidada (*Nested CV*) y regenerar todas las figuras:

```bash
# Opción A: Ejecución interactiva en Jupyter Notebook
jupyter notebook notebooks/02_modelado.ipynb

# Opción B: Ejecución automatizada vía script
python src/ejecutar_experimentos.py
```

### 4. Compilación del Informe a PDF

Para compilar el informe Markdown a un PDF académico de 5 páginas:
```bash
python src/build_pdf.py
```

---

## 🔬 Principales Hallazgos Metodológicos

1. **Réplica del Paper:** Se validó la réplica del OCSVM sobre Wisconsin Breast Cancer (Jones & Farrow, 2025), confirmando que a mayor ruido existe mayor solapamiento con la frontera y menor tasa de detección de outliers.
2. **Monitoreo en Colombia:** En las cohortes del INS, el OCSVM detecta ~99 % de inliers en 2021 y 2022 (igual que su tasa fuera de muestra en 2020), indicando que el soporte de covariables $X$ es estable.
3. **Drift de Covariables vs. Cambio de Prevalencia:** La aparente caída en el $F_1$ no obedece a un *covariate drift* sino a un **cambio en la tasa base de gravedad** ($2,82 \% \to 1,10 \%$), el cual no es detectable por un modelo no supervisado de entrada $X$.
4. **Parsimonia en Modelamiento:** La Regresión Logística empata estadísticamente con ensambles avanzados (AUC-PR $0,268 \pm 0,015$ vs $0,266 \pm 0,016$) y es seleccionada por su alta interpretabilidad y estabilidad.
