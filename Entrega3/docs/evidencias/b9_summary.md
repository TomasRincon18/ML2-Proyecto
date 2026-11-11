# B9. Docker, paridad, README y capturas

## Qué se preparó

- `Dockerfile` con `python:3.12-slim` y `requirements-app.txt` (versiones fijadas).
- `.dockerignore` y `.streamlit/config.toml`.
- `src/check_parity.py`: verifica la paridad del artefacto tras recargarlo.
- `src/apptest.py`: prueba funcional de la app (individual, advertencia por edad y lote).
- `scripts/docker_check.sh`: build + run + healthcheck + paridad + AppTest + logs.
- Capturas de la app en `docs/evidencias/`.

## Verificación en host

| Prueba | Resultado |
|---|---|
| `python src/check_parity.py` | `parity_diff = 0.00e+00` |
| `python src/apptest.py` | `APP TEST OK` |
| Captura inicio | `streamlit_inicio.png` |
| Captura alto riesgo | `streamlit_prediccion_alto.png` (24.6 %) |
| Captura bajo riesgo | `streamlit_prediccion_bajo.png` (0.2 % + advertencia) |
| Captura lote | `streamlit_lote.png` (100 filas, riesgo medio 8.7 %) |
| Captura ficha | `streamlit_ficha_modelo.png` |

## Docker

Comandos:

```bash
cd Proyecto/Entrega3
sudo bash scripts/docker_check.sh
```

o, si el usuario está en el grupo `docker`:

```bash
bash scripts/docker_check.sh
```

**Estado:** ejecutado por el propietario con `sudo bash scripts/docker_check.sh`.
El contenedor `ml2-entrega3` quedó corriendo en `localhost:8501`.

| Verificación | Resultado |
|---|---|
| `GET /_stcore/health` | `ok` |
| `GET /` | HTTP 200 |
| Predicción en contenedor (86 años) | 24.6 % · Grave (alerta) |
| Captura del contenedor | `docs/evidencias/streamlit_docker.png` |

La app del contenedor reproduce el mismo resultado que la app en host, lo que
confirma la coherencia entre entrenamiento, artefacto y serving.

## Nota para el informe

- El despliegue funcional está verificado en host y en contenedor.
- El contenedor se construyó y ejecutó con `sudo` porque el socket de Docker
  requiere permisos en esta máquina.
- Para detenerlo: `sudo docker stop ml2-entrega3` y `sudo docker rm ml2-entrega3`.
