# B10. Informe I: secciones 1-3

## Qué se escribió

`informe/informe_final.md` con:

- Portada, introducción y referencias.
- **Sección 1:** modelo final, configuración, métricas 2022, comparación de
  estrategias y contraste con el artículo.
- **Sección 2:** calibración, fracaso de EM, beta calibration descartada,
  decisión por reentreno + Platt y límites (Brier skill negativo, decil
  superior sobreestimado).
- **Sección 3:** SHAP global/local, PDP de edad, errores por subgrupo, curva de
  decisión y una lectura clínica realista (modelo etario, sin detección < 60).

## Figuras y tablas

| Elemento | Archivo / contenido |
|---|---|
| Cuadro 1 | Configuración y métricas del modelo final |
| Cuadro 2 | Comparación de estrategias 2022 |
| Cuadro 3 | Errores por grupo de edad |
| Figura 1 | `figures/calibracion_final_2022.png` |
| Figura 2 | `figures/shap_importance_ci.png` |
| Figura 3 | `figures/pdp_edad.png` |
| Figura 4 | `figures/decision_curve.png` |

## Cifras centrales verificadas

- Modelo final: AUC-PR 0,1903; ROC-AUC 0,9228; Brier 0,01192; ECE 0,02380.
- Decisión @ 0,0909: recall 0,786; precision 0,081; F1 0,1475.
- Bootstrap: reentreno + Platt vs Platt 2021: AUC-PR +0,0049; F1 +0,0050.
- SHAP: edad 1,663 [1,632; 1,697]; sexo 0,350; ciudad 0,218; fuente 0,162.
- PDP: cruce del umbral ~69 años; 0,213 a los 80; 0,606 a los 100.
- Subgrupos: recall 0 en < 60; 0,684 en 60-79; 1,000 en 80+.
- Curva de decisión: -0,0010 en 0,091; +0,0010 en 0,05.

## Pendiente para B11

- Secciones 4 (despliegue) y 5 (limitaciones y proyección).
- Conclusión.
- PDF con WeasyPrint, control de 3-5 páginas y revisión ortográfica.
- Posible recorte de figuras/tablas si el PDF excede el límite.

## Revisión post-bloque

**¿Cambios importantes?** **No.** La narrativa ya incorpora los hallazgos de B6
(modelo etario, umbral en el borde, Brier skill negativo).

Riesgo controlado: el informe tiene 4 figuras y 3 cuadros; si el PDF excede las
5 páginas, en B11 se recorta la Figura 2 o el Cuadro 2.
