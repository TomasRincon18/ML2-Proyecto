"""Prueba funcional de la app (Streamlit AppTest) para Docker y CI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parents[1]
APP = BASE / "app" / "app.py"


def _set_select(at: AppTest, label: str, value: str) -> None:
    next(sb for sb in at.selectbox if sb.label == label).select(value)


def _submit(at: AppTest) -> None:
    next(b for b in at.button if b.label == "Predecir").click().run()


def _individual_case(age: int, sex: str, city: str, source: str) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    next(n for n in at.number_input if n.label == "Edad (años)").set_value(age)
    _set_select(at, "Sexo", sex)
    _set_select(at, "Ciudad", city)
    _set_select(at, "Fuente de contagio", source)
    _submit(at)

    assert not at.exception, [e.value for e in at.exception]
    assert len(at.metric) >= 2, "No se renderizaron las métricas"
    return at


def main() -> None:
    print("=== APPTEST — APP COMPLETA ===")

    high = _individual_case(86, "M", "OTROS MUNICIPIOS", "COMUNITARIA")
    print("alto riesgo:", [m.value for m in high.metric[:2]])

    low = _individual_case(30, "F", "CALI", "COMUNITARIA")
    print("bajo riesgo:", [m.value for m in low.metric[:2]])
    assert len(low.warning) >= 1, "Falta la advertencia para menores de 60"

    # Lote sintético con etiquetas.
    batch = pd.DataFrame(
        {
            "edad_anios": [30, 45, 60, 75, 86] * 20,
            "sexo": ["F", "M"] * 50,
            "ciudad_top": ["BOGOTA", "CALI", "OTROS MUNICIPIOS", "MEDELLIN", "PASTO"] * 20,
            "fuente_tipo_contagio": ["COMUNITARIA", "RELACIONADO"] * 50,
            "grave": [0, 0, 1, 1, 0] * 20,
        }
    )
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    at.file_uploader[0].set_value(
        ("lote.csv", batch.to_csv(index=False).encode(), "text/csv")
    ).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = {m.label for m in at.metric}
    assert {"Filas", "Riesgo medio", "Alertas"} <= labels
    print("lote: metrics", sorted(labels))
    print("APP TEST OK")


if __name__ == "__main__":
    main()
