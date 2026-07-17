#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-hugegraph}"
IMAGE_NAME="${2:-hugegraph/hugegraph:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/hugegraph/knowledge_forge.properties"

if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "HugeGraph container not found: ${CONTAINER_NAME}" >&2
  exit 1
fi

if docker exec "${CONTAINER_NAME}" test -f /hugegraph-server/conf/graphs/knowledge_forge.properties 2>/dev/null; then
  echo "knowledge_forge graph config already exists"
else
  docker cp "${CONFIG_FILE}" "${CONTAINER_NAME}:/hugegraph-server/conf/graphs/knowledge_forge.properties"
fi

docker stop "${CONTAINER_NAME}" >/dev/null
docker run --rm --volumes-from "${CONTAINER_NAME}" --entrypoint sh "${IMAGE_NAME}" \
  -lc "cd /hugegraph-server && bin/init-store.sh"
docker start "${CONTAINER_NAME}" >/dev/null

for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8080/graphs | grep -q 'knowledge_forge'; then
    echo "knowledge_forge physical graph is ready"
    exit 0
  fi
  sleep 1
done

echo "HugeGraph did not become ready in time" >&2
exit 1
