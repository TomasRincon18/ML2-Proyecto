"""
Descarga el dataset de casos positivos de COVID-19 en Colombia (INS)
desde datos.gov.co usando la API SODA, seleccionando solo las 9 columnas
necesarias para el proyecto y paginando ordenadamente por :id.

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
    try:
        r = requests.get(
            "https://www.datos.gov.co/resource/gt2j-8ykr.json?$select=count(*)&$limit=1",
            timeout=60,
        )
        return int(r.json()[0]["count"])
    except Exception as e:
        print(f"No se pudo obtener el conteo total previo: {e}")
        return None


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    offset = 0
    total = _total_rows()
    if total:
        print(f"Total de registros a descargar: {total:,}", flush=True)

    paginas = 0
    t0 = time.time()

    with open(OUT, "w", encoding="utf-8") as f:
        while True:
            # Ordenamiento por :id para garantizar paginación estable y sin duplicados
            url = f"{BASE}?$select={COLUMNAS}&$limit={LIMIT}&$offset={offset}&$order=:id"
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            lines = resp.text.splitlines(True)

            if len(lines) <= 1:
                break

            if offset == 0:
                f.writelines(lines)
            else:
                f.writelines(lines[1:])  # Omitir cabecera en páginas subsecuentes

            rows_downloaded = len(lines) - 1
            offset += rows_downloaded
            paginas += 1

            if paginas % 10 == 0 or rows_downloaded < LIMIT:
                porcentaje = f"({offset/total*100:.1f}%)" if total else ""
                print(f"  Descargadas {offset:,} filas {porcentaje} | {paginas} páginas | {time.time()-t0:.0f}s", flush=True)

            if rows_downloaded < LIMIT:
                break

    print(f"\nDescarga finalizada con éxito:")
    print(f"  Archivo: {OUT}")
    print(f"  Total filas descargadas: {offset:,}")
    print(f"  Tamaño: {os.path.getsize(OUT)/1e6:.1f} MB")
    print(f"  Tiempo total: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
