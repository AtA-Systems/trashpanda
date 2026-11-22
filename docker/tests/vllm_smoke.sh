#!/usr/bin/env bash
# Serve a trashpanda export with vLLM and call it the way the LLM gateway does
# (OpenAI-compatible /v1/models and /v1/completions).
#
#   docker/tests/vllm_smoke.sh /path/to/export
#
# Needs a GPU with compute capability >= 7.0 (vLLM does not run on Pascal),
# Docker with the NVIDIA Container Toolkit, curl and python3 on the host.
# Use an export of a real recipe (e.g. 40m-titanx): the tiny recipe's
# 8-wide attention heads are below what vLLM's kernels support.
#
# Settings (environment):
#   VLLM_IMAGE  image to test with; pin it to the version the gateway runs
#   GPU         GPU index to use (default 0)
#   PORT        host port, bound to 127.0.0.1 only (default 18000)
#   DTYPE       vLLM --dtype (default auto: FP16 for an FP32 export)
#   GPU_UTIL    vLLM --gpu-memory-utilization (default 0.30)
set -euo pipefail

EXPORT=$(realpath "${1:?usage: vllm_smoke.sh EXPORT_DIR}")
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.31.0}
GPU=${GPU:-0}
PORT=${PORT:-18000}
DTYPE=${DTYPE:-auto}
GPU_UTIL=${GPU_UTIL:-0.30}
NAME="tp-vllm-smoke-$$"
SERVED="trashpanda-smoke"
BASE="http://127.0.0.1:${PORT}"

[[ -f "$EXPORT/config.json" && -f "$EXPORT/model.safetensors" ]] || { echo "FAIL: $EXPORT is not a trashpanda export"; exit 1; }

cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$GPU" | tr -d ' ')
if awk -v c="$cap" 'BEGIN { exit !(c < 7.0) }'; then
  echo "SKIP: GPU $GPU is compute capability $cap; vLLM needs 7.0 or newer (use Ollama or llama.cpp on this host)"
  exit 2
fi

cleanup() {
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "--- last vLLM log lines ---" >&2
    docker logs --tail 60 "$NAME" >&2 2>/dev/null || true
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  exit $status
}
trap cleanup EXIT

echo "[start] $VLLM_IMAGE on GPU $GPU (compute $cap), export $EXPORT"
# The image's entrypoint is `vllm serve`, so the model path is the first argument.
docker run -d --name "$NAME" --gpus "device=$GPU" --ipc=host \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$EXPORT:/models/tp:ro" \
  "$VLLM_IMAGE" /models/tp \
  --served-model-name "$SERVED" \
  --max-model-len 512 \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enforce-eager >/dev/null

echo "[wait] for $BASE/health (up to 5 minutes)"
for _ in $(seq 1 150); do
  if curl -fsS "$BASE/health" >/dev/null 2>&1; then break; fi
  if ! docker ps -q --filter "name=^${NAME}$" | grep -q .; then echo "FAIL: vLLM exited during startup"; exit 1; fi
  sleep 2
done
curl -fsS "$BASE/health" >/dev/null || { echo "FAIL: vLLM never became healthy"; exit 1; }

echo "[models]"
curl -fsS "$BASE/v1/models" | python3 -c "
import json, sys
ids = [m['id'] for m in json.load(sys.stdin)['data']]
assert '$SERVED' in ids, ids
print('  served:', ids)"

echo "[completion]"
curl -fsS "$BASE/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\": \"$SERVED\", \"prompt\": \"The water cycle\", \"max_tokens\": 24, \"temperature\": 0}" \
  | python3 -c "
import json, sys
r = json.load(sys.stdin)
text = r['choices'][0]['text']
used = r['usage']['completion_tokens']
assert used > 0 and text.strip(), r
print('  tokens:', used)
print('  text  :', repr(text))"

echo "PASS: vLLM served the export through the OpenAI API"
