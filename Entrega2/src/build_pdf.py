import os
import markdown
from weasyprint import HTML

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(BASE, "informe", "informe_entrega2.md")
OUT = os.path.join(BASE, "informe", "informe_entrega2.pdf")

with open(MD, encoding="utf-8") as f:
    text = f.read()

body = markdown.markdown(
    text,
    extensions=["tables", "fenced_code", "attr_list", "sane_lists"],
    output_format="html5",
)

CSS = """
@page {
    size: A4;
    margin: 2cm 2cm 2.2cm 2cm;
    @bottom-center { content: counter(page) " / " counter(pages); font-size: 9pt; color: #666; }
}
body {
    font-family: 'DejaVu Sans', 'Helvetica', sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #1a1a1a;
}
h1 {
    font-size: 16pt;
    text-align: center;
    margin: 0.6em 0 0.3em 0;
    color: #0b2545;
    page-break-before: always;
}
h1:first-of-type { page-break-before: avoid; }
h2 {
    font-size: 12.5pt;
    color: #0b2545;
    border-bottom: 1px solid #ccc;
    padding-bottom: 2px;
    margin-top: 1.2em;
}
h3 { font-size: 11pt; color: #134074; }
p { text-align: justify; margin: 0.5em 0; }
strong { color: #0b2545; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.7em 0;
    font-size: 9pt;
}
th, td { border: 1px solid #bbb; padding: 4px 7px; text-align: center; }
th { background: #eef2f7; color: #0b2545; }
img { display: block; margin: 0.8em auto; max-width: 95%; max-height: 15cm; }
hr { border: none; border-top: 1px solid #999; margin: 1.5em 0; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; background: #f2f2f2; padding: 0 2px; }
"""

html = f"<html><head><meta charset='utf-8'></head><body>{body}</body></html>"
HTML(string=html, base_url=os.path.join(BASE, "informe")).write_pdf(OUT, stylesheets=[__import__("weasyprint").CSS(string=CSS)])

print("PDF generado en:", OUT)
print("Tamaño:", round(os.path.getsize(OUT) / 1024), "KB")
