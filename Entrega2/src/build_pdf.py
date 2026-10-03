import os
import subprocess
import markdown
import pypdf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(BASE, "informe", "informe_entrega2.md")
HTML_PATH = os.path.join(BASE, "informe", "informe_entrega2.html")
PDF_PATH = os.path.join(BASE, "informe", "informe_entrega2.pdf")

with open(MD_PATH, encoding="utf-8") as f:
    text = f.read()

# Convertir Markdown a HTML
body = markdown.markdown(
    text,
    extensions=["tables", "fenced_code", "attr_list", "sane_lists"],
    output_format="html5",
)

CUSTOM_CSS = """
@page {
    size: letter;
    margin: 1.2cm 1.4cm 1.3cm 1.4cm;
}

@media print {
    body {
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 8.2pt;
    line-height: 1.28;
    color: #212529;
    background-color: #ffffff;
}

h1 {
    font-size: 10.5pt;
    color: #1d3557;
    margin-top: 0.6em;
    margin-bottom: 0.2em;
    border-bottom: 1px solid #457b9d;
    padding-bottom: 1px;
    font-weight: 700;
    break-after: avoid;
    page-break-after: avoid;
}

h2 {
    font-size: 9.2pt;
    color: #457b9d;
    margin-top: 0.5em;
    margin-bottom: 0.15em;
    font-weight: 600;
    break-after: avoid;
    page-break-after: avoid;
}

h3 {
    font-size: 8.5pt;
    color: #1d3557;
    margin-top: 0.4em;
    margin-bottom: 0.12em;
    font-weight: 600;
    break-after: avoid;
    page-break-after: avoid;
}

p {
    text-align: justify;
    margin-top: 0.22em;
    margin-bottom: 0.22em;
    text-justify: inter-word;
}

ul, ol {
    margin-top: 0.18em;
    margin-bottom: 0.18em;
    padding-left: 1.2em;
}

li {
    margin-bottom: 0.08em;
    text-align: justify;
}

strong {
    color: #0b1d3a;
    font-weight: 700;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.4em auto;
    font-size: 7.2pt;
    break-inside: avoid;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #ced4da;
    padding: 2.2px 4px;
    text-align: center;
}

th {
    background-color: #e9ecef;
    color: #1d3557;
    font-weight: 700;
}

tr:nth-child(even) {
    background-color: #f8f9fa;
}

img {
    display: block;
    margin: 0.3em auto;
    max-width: 78%;
    max-height: 4.1cm;
    object-fit: contain;
    break-inside: avoid;
    page-break-inside: avoid;
}

hr {
    border: none;
    border-top: 1px solid #dee2e6;
    margin: 0.45em 0;
}

code {
    font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 7.4pt;
    background-color: #f1f3f5;
    padding: 1px 3px;
    border-radius: 2px;
}
"""

html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <title>Informe Entrega 2 — Machine Learning II</title>
    <style>
    {CUSTOM_CSS}
    </style>
</head>
<body>
{body}
</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)

# Compilación a PDF vía Headless Microsoft Edge (estándar nativo de Windows)
edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(edge_exe):
    edge_exe = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

cmd = [
    edge_exe,
    "--headless",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={PDF_PATH}",
    HTML_PATH
]

print("Compilando PDF con motor Chromium (Microsoft Edge)...")
subprocess.run(cmd, check=True)

reader = pypdf.PdfReader(PDF_PATH)
num_pages = len(reader.pages)

print(f"\n==========================================")
print(f"PDF generado con éxito: {PDF_PATH}")
print(f"Número total de páginas: {num_pages}")
print(f"Tamaño del archivo: {round(os.path.getsize(PDF_PATH) / 1024)} KB")
if 3 <= num_pages <= 5:
    print(f"ESTADO: CUMPLE ESTRICTAMENTE LA GUÍA (3 a 5 páginas)")
else:
    print(f"ADVERTENCIA: Número de páginas fuera de rango ({num_pages})")
print(f"==========================================")
