"""Compila el informe final Markdown a PDF con WeasyPrint."""

from __future__ import annotations

import re
from pathlib import Path

import markdown
from pypdf import PdfReader
from weasyprint import HTML

BASE = Path(__file__).resolve().parents[1]
INFORME_DIR = BASE / "informe"
MD_PATH = INFORME_DIR / "informe_final.md"
HTML_PATH = INFORME_DIR / "informe_final.html"
PDF_PATH = INFORME_DIR / "informe_final.pdf"


def preprocess(text: str) -> str:
    # LaTeX simple a texto/HTML, sin depender de MathJax.
    text = text.replace("\\(", "").replace("\\)", "")
    text = re.sub(
        r"\\\[\s*(.*?)\s*\\\]",
        lambda _: (
            '<div class="formula">'
            "t* = 1 / (1 + C<sub>FN</sub>/C<sub>FP</sub>) = 1 / 11 = 0,0909"
            "</div>"
        ),
        text,
        flags=re.S,
    )
    return text


CSS = """
@page {
  size: letter;
  margin: 1.1cm 1.4cm 1.2cm 1.4cm;
  @bottom-center {
    content: counter(page);
    font-size: 7pt;
    color: #6c757d;
  }
}
body {
  font-family: "DejaVu Sans", "Segoe UI", Arial, sans-serif;
  font-size: 8.1pt;
  line-height: 1.26;
  color: #212529;
}
h1 {
  font-size: 10.2pt;
  color: #1d3557;
  margin: 0.55em 0 0.2em 0;
  border-bottom: 1px solid #457b9d;
  padding-bottom: 1px;
  page-break-after: avoid;
}
h2 {
  font-size: 9.0pt;
  color: #457b9d;
  margin: 0.45em 0 0.15em 0;
  page-break-after: avoid;
}
h3 {
  font-size: 8.4pt;
  color: #1d3557;
  margin: 0.35em 0 0.1em 0;
  page-break-after: avoid;
}
p {
  text-align: justify;
  margin: 0.2em 0;
}
ul, ol {
  margin: 0.15em 0;
  padding-left: 1.2em;
}
li {
  margin-bottom: 0.06em;
  text-align: justify;
}
strong { color: #0b1d3a; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.35em auto;
  font-size: 7.0pt;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #ced4da;
  padding: 2px 3.5px;
  text-align: center;
}
th {
  background: #e9ecef;
  color: #1d3557;
}
tr:nth-child(even) { background: #f8f9fa; }
img {
  display: block;
  margin: 0.25em auto;
  max-width: 76%;
  max-height: 4.0cm;
  object-fit: contain;
  page-break-inside: avoid;
}
hr {
  border: none;
  border-top: 1px solid #dee2e6;
  margin: 0.4em 0;
}
code {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 7.2pt;
  background: #f1f3f5;
  padding: 0 2px;
}
.formula {
  text-align: center;
  font-weight: 700;
  margin: 0.35em 0;
}
a { color: #1d3557; text-decoration: none; }
"""


def main() -> None:
    text = preprocess(MD_PATH.read_text(encoding="utf-8"))
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "attr_list", "sane_lists"],
        output_format="html5",
    )
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Informe final — Machine Learning II</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""
    HTML_PATH.write_text(html, encoding="utf-8")
    HTML(string=html, base_url=str(INFORME_DIR)).write_pdf(str(PDF_PATH))

    pages = len(PdfReader(str(PDF_PATH)).pages)
    size_kb = round(PDF_PATH.stat().st_size / 1024)
    print(f"PDF: {PDF_PATH}")
    print(f"Páginas: {pages}")
    print(f"Tamaño: {size_kb} KB")
    if pages <= 6:
        print("ESTADO: dentro del rango razonable (<= 6 páginas totales)")
    else:
        print("ADVERTENCIA: más de 6 páginas; recortar figuras o tablas")


if __name__ == "__main__":
    main()
