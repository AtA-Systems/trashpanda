#!/usr/bin/env bash
# Test a trashpanda image on this host, from GPU visibility through to a
# Hugging Face transformers load and (optionally) vLLM serving.
#
#   docker/test-image.sh IMAGE [--data DIR] [--gpus N] [--steps N] [--vllm] [--out DIR]
#                        [--device DEV] [--gpu-memory-fraction F] [--recipe R] [--microbatch-size N]
#
#   IMAGE     e.g. trashpanda:dev-pascal or ghcr.io/ata-systems/trashpanda:0.1.0-modern-cu130
#   --data    a prepared dataset on the host (e.g. /data/datasets/fw-40m-titanx);
#             enables the GPU training run. Mounted read-only.
#   --gpus    GPUs for the GPU run and the multi-GPU tests (default: all visible)
#   --steps   training updates in the GPU run (default 20)
#   --vllm    after the GPU run, serve its export with vLLM (needs compute >= 7.0)
#   --out     where reports and the GPU run's export go (default ./test-results/<tag>-<time>)
#   --device  which GPU(s) the containers get: a docker --gpus value, e.g. all (default),
#             device=0, or device=MIG-<uuid> for one MIG slice
#   --gpu-memory-fraction  cap trashpanda at this share of each GPU's memory, e.g. 0.2 on a
#             card shared with a vLLM server (sets TRASHPANDA_CUDA_MEMORY_FRACTION)
#   --recipe / --microbatch-size  recipe for the GPU run (default 40m-titanx at its default size)
#
# Stages, each must pass (vLLM may SKIP on Pascal):
#   1 gpu-visible   nvidia-smi inside the container sees the GPUs
#   2 unit-tests    trashpanda's CPU unit + acceptance suite inside the image
#   3 multi-gpu     tests/test_multi_device.py (only with 2+ GPUs)
#   4 e2e-tiny      offline tiny run -> export -> transformers equivalence
#   5 e2e-gpu       40m-titanx on CUDA -> export -> transformers on GPU (needs --data)
#   6 vllm          OpenAI-API completion from vLLM serving that export (needs --vllm)
set -uo pipefail

IMAGE=${1:?usage: test-image.sh IMAGE [--data DIR] [--gpus N] [--steps N] [--vllm] [--out DIR] [--device DEV] [--gpu-memory-fraction F] [--recipe R] [--microbatch-size N]}
shift
DATA="" STEPS=20 VLLM=0 OUT="" GPUS="" DEVICE=all FRACTION="" RECIPE=40m-titanx MB=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE=$2; shift 2 ;;
    --gpu-memory-fraction) FRACTION=$2; shift 2 ;;
    --recipe) RECIPE=$2; shift 2 ;;
    --microbatch-size) MB=$2; shift 2 ;;
    --data) DATA=$(realpath "$2"); shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --vllm) VLLM=1; shift ;;
    --out) OUT=$2; shift 2 ;;
    *) echo "unknown option $1"; exit 64 ;;
  esac
done
HERE=$(cd "$(dirname "$0")" && pwd)
tag=$(echo "$IMAGE" | tr '/:' '__')
OUT=$(realpath -m "${OUT:-./test-results/${tag}-$(date +%Y%m%d-%H%M%S)}")
mkdir -p "$OUT"
if [[ -z "$GPUS" ]]; then
  if [[ "$DEVICE" == all ]]; then GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU'); else GPUS=1; fi
fi
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}

RUN=(docker run --rm --gpus "$DEVICE" --ipc=host --shm-size=8g --user "$(id -u):$(id -g)" -e HOME=/work -v "$OUT:/out" -v "$HF_CACHE:/hf")
[[ -n "$FRACTION" ]] && RUN+=(-e "TRASHPANDA_CUDA_MEMORY_FRACTION=$FRACTION")
GPU_RUN_ARGS=(--recipe "$RECIPE")
[[ -n "$MB" ]] && GPU_RUN_ARGS+=(--microbatch-size "$MB")
declare -A RESULT
order=()
stage() {  # stage NAME command...
  local name=$1; shift
  order+=("$name")
  echo; echo "===== $name ====="
  "$@" 2>&1 | tee "$OUT/$name.log"
  local status=${PIPESTATUS[0]}
  case $status in
    0) RESULT[$name]=PASS ;;
    2) RESULT[$name]=SKIP ;;
    *) RESULT[$name]=FAIL ;;
  esac
}

echo "image: $IMAGE   gpus: $GPUS ($DEVICE)   memory cap: ${FRACTION:-none}   out: $OUT"
docker image inspect "$IMAGE" --format 'labels: {{json .Config.Labels}}' 2>/dev/null || docker pull "$IMAGE"

stage gpu-visible "${RUN[@]}" "$IMAGE" nvidia-smi
# CPU-only stages hide the GPUs, so they pass on any host and any flavor.
CPU_ONLY=(-e CUDA_VISIBLE_DEVICES= -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)
stage unit-tests "${RUN[@]}" "${CPU_ONLY[@]}" -w /opt/trashpanda "$IMAGE" \
  python -m pytest -q -p no:cacheprovider -m "unit or acceptance"
if [[ "$GPUS" -ge 2 ]]; then
  stage multi-gpu "${RUN[@]}" -w /opt/trashpanda "$IMAGE" \
    python -m pytest -q -p no:cacheprovider tests/test_multi_device.py
fi
stage e2e-tiny "${RUN[@]}" "${CPU_ONLY[@]}" "$IMAGE" python /opt/trashpanda/docker/tests/e2e.py --mode tiny --out /out/e2e-tiny
if [[ -n "$DATA" ]]; then
  stage e2e-gpu "${RUN[@]}" -v "$DATA:/data/prepared:ro" "$IMAGE" \
    python /opt/trashpanda/docker/tests/e2e.py --mode gpu --data /data/prepared --gpus "$GPUS" --steps "$STEPS" "${GPU_RUN_ARGS[@]}" --out /out/e2e-gpu
  if [[ $VLLM -eq 1 ]]; then
    if [[ ${RESULT[e2e-gpu]} == PASS ]]; then
      stage vllm "$HERE/tests/vllm_smoke.sh" "$OUT/e2e-gpu/export"
    else
      order+=(vllm); RESULT[vllm]="SKIP (no export: e2e-gpu failed)"
    fi
  fi
fi

echo; echo "===== summary: $IMAGE ====="
failed=0
for name in "${order[@]}"; do
  printf '  %-12s %s\n' "$name" "${RESULT[$name]}"
  [[ ${RESULT[$name]} == FAIL ]] && failed=1
done
echo "  logs and reports: $OUT"
exit $failed
