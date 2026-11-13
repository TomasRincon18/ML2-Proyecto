# B12. Deck, guion y cierre

## Qué se hizo

- `slides/index.html`: deck reveal.js de 12 diapositivas, con reveal.js y
  imágenes locales.
- `slides/assets/`: figuras específicas de diapositiva, con estilo y tipografía propios.
- `src/build_slide_figures.py`: genera calibración, SHAP, PDP, errores, decisión, monitor y captura de la app para el deck.
- Rediseño visual: portada oscura, kickers, tarjetas, timeline, métricas grandes, figuras enmarcadas y cierre.
- `slides/reveal/`: reveal.js y tema, sin dependencia de CDN.
- `slides/slides.pdf`: respaldo de 12 páginas.
- `slides/guion.md`: guion por diapositiva, reparto, tiempos, preguntas
  probables y plan de demo.
- `slides/README.md`: cómo abrir y navegar el deck.

## Reparto

| Tramo | Diapositivas | Quién |
|---|---|---|
| Problema, artículo, datos, protocolo y OCSVM | 1-6 | Ailyn |
| Modelo final, calibración, interpretabilidad y errores | 7-10 | Juan Tomás |
| Despliegue y cierre | 11-12 | Juan Tomás |

## Verificación

- 12 diapositivas detectadas por reveal.js.
- Sin errores de JavaScript ni recursos faltantes.
- Imágenes locales, sin CDN.
- PDF de respaldo: 12 páginas.
- El deck actualizado está en `slides/index.html`; `slides/slides.pdf` queda como respaldo.
- La diapositiva 8 se ajustó para que no se cortara el contenido.
- El deck se probó desde `file://` y desde un servidor local, sin errores ni recursos faltantes.

## Checklist de rúbrica

| Criterio | Evidencia |
|---|---|
| Modelo final y reproducción | Informe §1, Cuadro 1, `model_final.joblib` |
| Interpretación | Informe §3, SHAP, PDP, Cuadro 3 |
| Despliegue | Informe §4, app, Docker, `streamlit_docker.png` |
| Calidad y limitaciones | Informe §5, Brier skill, modelo etario, 2022 no ciego |
| Comunicación | Informe de 5 páginas, deck de 12 diapositivas, guion |

## Revisión post-bloque

**¿Cambios importantes?** **No.** El deck y el guion están alineados con el
informe y con los resultados.

Riesgo controlado: la demo en vivo depende de que el contenedor o el entorno
local estén corriendo. Si falla, están las capturas y el PDF de respaldo.

## Cierre del proyecto

Con B12 quedan completos B1 a B12. Los entregables principales son:

- `informe/informe_final.pdf`
- `app/app.py`
- `artifacts/model_final.joblib`
- `Dockerfile` y `scripts/docker_check.sh`
- `slides/index.html` y `slides/guion.md`
