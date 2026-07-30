#!/usr/bin/env bash
# FP32 SSM state, STP (standard per-step, no ReplaySSM), Qwen3.5-35B-A3B
# Gold baseline: no precision emulation applied.
set -euo pipefail

MODEL="/raid/data/vgimpelson/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754"
if [ ! -d "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL" >&2
    echo "Re-download with: huggingface-cli download Qwen/Qwen3.5-35B-A3B --local-dir <path>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export SSM_PRECISION_DTYPE=fp32  # no-op — gold baseline

source /home/scratch.ameyn_gpu_2/venv/bin/activate
export PYTHONPATH="$REPO_ROOT/python:$PYTHONPATH"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --enforce-eager \
    --mamba-ssm-cache-dtype float32 \
    --disable-log-requests \
    --port 8000 \
    "$@"
