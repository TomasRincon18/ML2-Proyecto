#!/usr/bin/env bash
# Construye, ejecuta y verifica el contenedor de la Entrega 3.
# Uso:
#   sudo bash scripts/docker_check.sh
# o, si el usuario pertenece al grupo docker:
#   bash scripts/docker_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="ml2-entrega3:latest"
CONTAINER="ml2-entrega3"

echo "== build =="
docker build -t "$IMAGE" .

echo "== run =="
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" -p 8501:8501 "$IMAGE"

echo "== health =="
for i in $(seq 1 40); do
  if curl -sf http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
    echo "health: ok"
    break
  fi
  sleep 1
done

echo "== parity =="
docker exec "$CONTAINER" python src/check_parity.py

echo "== app test =="
docker exec "$CONTAINER" python src/apptest.py

echo "== status =="
docker ps --filter "name=$CONTAINER"
docker logs --tail 20 "$CONTAINER"
