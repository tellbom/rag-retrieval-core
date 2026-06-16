#!/usr/bin/env bash
set -euo pipefail

# Sequential TEI startup for the current RAG config.
#
# Defaults match configs/base.json:
#   text2vec-large-chinese -> http://HOST:18081
#   bge-m3                 -> http://HOST:18082
#   bge-reranker-v2-m3     -> http://HOST:18084
#
# Typical use on the model server:
#   bash deploy/docker/start-hf-services.sh
#
# Useful overrides:
#   HF_MODELS_DIR=/root/models
#   TEI_IMAGE=ghcr.io/huggingface/text-embeddings-inference:cpu-1.7
#   TEXT2VEC_MODEL=/data/models/text2vec-large-chinese
#   START_RERANKER=0 bash deploy/docker/start-hf-services.sh
#   POLL_SECONDS=2 EMBED_TIMEOUT_SECONDS=300 RERANK_TIMEOUT_SECONDS=1800 bash ...

HF_MODELS_DIR="${HF_MODELS_DIR:-/root/models}"
TEI_IMAGE="${TEI_IMAGE:-ghcr.io/huggingface/text-embeddings-inference:cpu-1.7}"
POLL_SECONDS="${POLL_SECONDS:-2}"
EMBED_TIMEOUT_SECONDS="${EMBED_TIMEOUT_SECONDS:-300}"
RERANK_TIMEOUT_SECONDS="${RERANK_TIMEOUT_SECONDS:-1800}"
START_RERANKER="${START_RERANKER:-1}"
RECREATE_CONTAINERS="${RECREATE_CONTAINERS:-0}"
RESTART_POLICY="${RESTART_POLICY:-no}"
HOST_BIND="${HOST_BIND:-0.0.0.0}"

TEXT2VEC_NAME="${TEXT2VEC_NAME:-tei-text2vec-large-chinese}"
TEXT2VEC_MODEL="${TEXT2VEC_MODEL:-text2vec-large-chinese}"
TEXT2VEC_PORT="${TEXT2VEC_PORT:-18081}"

BGE_M3_NAME="${BGE_M3_NAME:-tei-bge-m3}"
BGE_M3_MODEL="${BGE_M3_MODEL:-bge-m3}"
BGE_M3_PORT="${BGE_M3_PORT:-18082}"

RERANKER_NAME="${RERANKER_NAME:-tei-bge-reranker-v2-m3}"
RERANKER_MODEL="${RERANKER_MODEL:-bge-reranker-v2-m3-onnx}"
RERANKER_PORT="${RERANKER_PORT:-18084}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 2
  }
}

container_exists() {
  docker inspect "$1" >/dev/null 2>&1
}

container_running() {
  [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || echo false)" = "true" ]
}

ensure_model_dir() {
  local model_dir
  model_dir="$(resolve_model_dir "$1")"
  [ -d "$model_dir" ] || {
    echo "Model directory not found: $model_dir" >&2
    exit 3
  }
}

resolve_model_dir() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$HF_MODELS_DIR" "$1" ;;
  esac
}

create_or_start_embedding() {
  local name="$1"
  local model="$2"
  local port="$3"
  local model_dir

  ensure_model_dir "$model"
  model_dir="$(resolve_model_dir "$model")"

  if container_exists "$name"; then
    docker update --restart="$RESTART_POLICY" "$name" >/dev/null
    if [ "$RECREATE_CONTAINERS" = "1" ]; then
      log "Removing existing container: $name"
      docker rm -f "$name" >/dev/null || true
    else
      log "Starting existing container: $name"
      docker start "$name" >/dev/null
      return
    fi
  fi

  log "Creating embedding container: $name -> port $port, model $model_dir"
  docker run -d \
    --name "$name" \
    --restart "$RESTART_POLICY" \
    -p "$HOST_BIND:$port:80" \
    -v "$model_dir:/data/model:ro" \
    -e HF_HUB_OFFLINE=1 \
    "$TEI_IMAGE" \
    --model-id "/data/model" >/dev/null
}

create_or_start_reranker() {
  local name="$1"
  local model="$2"
  local port="$3"
  local model_dir

  ensure_model_dir "$model"
  model_dir="$(resolve_model_dir "$model")"

  if container_exists "$name"; then
    docker update --restart="$RESTART_POLICY" "$name" >/dev/null
    if [ "$RECREATE_CONTAINERS" = "1" ]; then
      log "Removing existing container: $name"
      docker rm -f "$name" >/dev/null || true
    else
      log "Starting existing container: $name"
      docker start "$name" >/dev/null
      return
    fi
  fi

  log "Creating reranker container: $name -> port $port, model $model_dir"
  docker run -d \
    --name "$name" \
    --restart "$RESTART_POLICY" \
    -p "$HOST_BIND:$port:80" \
    -v "$model_dir:/data/model:ro" \
    -e HF_HUB_OFFLINE=1 \
    "$TEI_IMAGE" \
    --model-id "/data/model" >/dev/null
}

wait_http_ready() {
  local name="$1"
  local port="$2"
  local timeout="$3"
  local deadline=$((SECONDS + timeout))
  local round=0

  log "Waiting for $name /health on port $port, timeout=${timeout}s"
  while [ "$SECONDS" -lt "$deadline" ]; do
    round=$((round + 1))
    if ! container_running "$name"; then
      echo "$name exited before ready" >&2
      docker logs --tail 120 "$name" >&2 || true
      exit 4
    fi
    if curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      log "$name health ready after ${round} probe(s)"
      return
    fi
    sleep "$POLL_SECONDS"
  done

  echo "$name did not become ready within ${timeout}s" >&2
  docker logs --tail 160 "$name" >&2 || true
  exit 5
}

probe_embed() {
  local name="$1"
  local port="$2"
  local body

  body="$(curl -fsS --max-time 60 \
    -H 'Content-Type: application/json' \
    -d '{"inputs":"RAG embedding smoke test"}' \
    "http://127.0.0.1:$port/embed")"

  python3 - "$name" "$body" <<'PY'
import json
import sys
name, body = sys.argv[1], sys.argv[2]
data = json.loads(body)
if not isinstance(data, list) or not data or not isinstance(data[0], list):
    raise SystemExit(f"{name} /embed unexpected shape")
print(f"{name} /embed ok: dim={len(data[0])}")
PY
}

probe_embed_all() {
  local name="$1"
  local port="$2"
  local body
  local status

  body="$(mktemp)"
  status="$(curl -sS --max-time 90 \
    -o "$body" \
    -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -d '{"inputs":"RAG embed_all smoke test"}' \
    "http://127.0.0.1:$port/embed_all" || true)"

  if [ "$status" != "200" ]; then
    log "$name /embed_all not available: http=$status body=$(head -c 200 "$body")"
    rm -f "$body"
    return 0
  fi

  python3 - "$name" "$body" <<'PY'
import json
import sys
name, path = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as fh:
    data = json.load(fh)
if (
    not isinstance(data, list)
    or len(data) != 1
    or not isinstance(data[0], list)
):
    raise SystemExit(f"{name} /embed_all unexpected shape")
tokens = len(data[0])
dim = len(data[0][0]) if tokens else 0
print(f"{name} /embed_all ok: tokens={tokens} dim={dim}")
PY
  rm -f "$body"
}

probe_rerank() {
  local name="$1"
  local port="$2"
  local body

  body="$(curl -fsS --max-time 120 \
    -H 'Content-Type: application/json' \
    -d '{"query":"RAG smoke test","texts":["RAG smoke test document","unrelated text"]}' \
    "http://127.0.0.1:$port/rerank")"

  python3 - "$name" "$body" <<'PY'
import json
import sys
name, body = sys.argv[1], sys.argv[2]
data = json.loads(body)
if not isinstance(data, list) or not data or "score" not in data[0]:
    raise SystemExit(f"{name} /rerank unexpected shape")
print(f"{name} /rerank ok: results={len(data)} top_score={data[0]['score']}")
PY
}

start_embedding_and_probe() {
  local name="$1"
  local model="$2"
  local port="$3"

  create_or_start_embedding "$name" "$model" "$port"
  wait_http_ready "$name" "$port" "$EMBED_TIMEOUT_SECONDS"
  probe_embed "$name" "$port"
  probe_embed_all "$name" "$port"
}

start_reranker_and_probe() {
  local name="$1"
  local model="$2"
  local port="$3"

  create_or_start_reranker "$name" "$model" "$port"
  wait_http_ready "$name" "$port" "$RERANK_TIMEOUT_SECONDS"
  probe_rerank "$name" "$port"
}

main() {
  require_cmd docker
  require_cmd curl
  require_cmd python3

  log "TEI image: $TEI_IMAGE"
  log "Model directory: $HF_MODELS_DIR"
  log "Polling every ${POLL_SECONDS}s"

  start_embedding_and_probe "$TEXT2VEC_NAME" "$TEXT2VEC_MODEL" "$TEXT2VEC_PORT"
  start_embedding_and_probe "$BGE_M3_NAME" "$BGE_M3_MODEL" "$BGE_M3_PORT"

  if [ "$START_RERANKER" = "1" ]; then
    start_reranker_and_probe "$RERANKER_NAME" "$RERANKER_MODEL" "$RERANKER_PORT"
  else
    log "Skipping reranker because START_RERANKER=$START_RERANKER"
    if container_exists "$RERANKER_NAME"; then
      docker update --restart=no "$RERANKER_NAME" >/dev/null || true
      docker stop -t 20 "$RERANKER_NAME" >/dev/null 2>&1 || true
    fi
  fi

  log "Final container status:"
  docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' \
    | grep -E "NAMES|$TEXT2VEC_NAME|$BGE_M3_NAME|$RERANKER_NAME"
}

main "$@"
