#!/usr/bin/env bash
# FP32 SSM state, STP (no ReplaySSM), Qwen3.5-122B-A10B-NVFP4, TP=4
# Gold baseline: no precision emulation applied.
set -euo pipefail

IMG=${BENCH_IMG:-vllm/vllm-openai:nightly-6a9f24aa8cb856235528d01a829a4ba85fc1c19d}
MODEL=/home/scratch.ameyn_gpu_2/models/Qwen3.5-122B-A10B-NVFP4
GPUS=${CUDA_VISIBLE_DEVICES:-1,3,4,5}
PORT=${PORT:-8000}
CONTAINER=ssmprec_fp32_122b
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker rm -f "$CONTAINER" 2>/dev/null || true

exec docker run --name "$CONTAINER" \
  --gpus "\"device=$GPUS\"" \
  --network host --ipc host --shm-size 64g \
  -v /home/scratch.ameyn_gpu_2:/home/scratch.ameyn_gpu_2 \
  -e SSM_PRECISION_DTYPE=fp32 \
  -e PYTHONPATH="$REPO_ROOT" \
  --entrypoint bash \
  "$IMG" -c "
    python3 -m ssm_precision.run_server \
      --model $MODEL \
      --tensor-parallel-size 4 \
      --dtype bfloat16 \
      --enforce-eager \
      --mamba-ssm-cache-dtype float32 \
      --gpu-memory-utilization 0.85 \
      --no-enable-log-requests \
      --port $PORT
  "
