# Datos

La Entrega 3 reutiliza el dataset consolidado de la Entrega 2:

```text
../../Entrega2/data/processed/dataset_reducido.csv
```

No se duplica para no sumar 28 MB al repositorio. `src/config.py` resuelve la
ruta relativa y se puede sobreescribir con `ML2_DATA_PATH`.

## Contenido

- 300.000 registros: 100.000 por ola (2020, 2021, 2022).
- Variable objetivo: `grave` (1 = grave/fallecido/UCI; 0 = leve/moderado).
- Predictores: `edad_anios`, `sexo`, `ciudad_top`, `fuente_tipo_contagio`.
- Partición: 2020 entrena, 2021 calibra, 2022 test final.
