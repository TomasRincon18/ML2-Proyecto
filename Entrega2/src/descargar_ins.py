"""
Descarga el dataset de casos positivos de COVID-19 en Colombia (INS)
desde datos.gov.co usando la API SODA, seleccionando solo las columnas
necesarias para el proyecto y paginando hasta obtener todas las filas.

Columnas seleccionadas (9):
  fecha_de_notificaci_n, edad, unidad_medida, sexo, departamento_nom,
  ciudad_municipio_nom, fuente_tipo_contagio, ubicacion, estado
"""
import os
import time

import requests

BASE = "https://www.datos.gov.co/resource/gt2j-8ykr.csv"
COLUMNAS = (
    "fecha_de_notificaci_n,edad,unidad_medida,sexo,departamento_nom,"
    "ciudad_municipio_nom,fuente_tipo_contagio,ubicacion,estado"
)
LIMIT = 50_000
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "casos_covid_colombia.csv")
OUT = os.path.abspath(OUT)


def _total_rows():
    r = requests.get(
        "https://www.datos.gov.co/resource/gt2j-8ykr.json?$select=count(*)&$limit=1",
        timeout=60,
    )
    return int(r.json()[0]["count"])


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    offset = 0
    total = None
    paginas = 0
    t0 = time.time()

    with open(OUT, "w", encoding="utf-8") as f:
        while True:
            url = f"{BASE}?$select={COLUMNAS}&$limit={LIMIT}&$offset={offset}"
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            text = resp.text

            if offset == 0:
                f.write(text)
            else:
                f.writelines(text.splitlines(True)[1:])

            n = text.count("\n")
            if n <= 1:
                break
            if total is None:
                total = _total_rows()
                print(f"Total de filas a descargar: {total:,}", flush=True)
            offset += n
            paginas += 1
            if paginas % 5 == 0:
                print(f"  filas ~ {offset:,} | paginas {paginas} | {time.time()-t0:.0f}s", flush=True)
            if n < LIMIT:
                break

    print(f"Descarga completa: {offset:,} filas, {os.path.getsize(OUT)/1e6:.1f} MB, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
