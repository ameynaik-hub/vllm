#!/usr/bin/env bash
# FP16 RTN emulated SSM state, STP (standard per-step, no ReplaySSM), Qwen3.5-35B-A3B
# After each Triton state update, Python applies ssm_state[idx] = ssm_state[idx].half().float()
set -euo pipefail

MODEL="/raid/data/vgimpelson/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754"
if [ ! -d "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL" >&2
    echo "Re-download with: huggingface-cli download Qwen/Qwen3.5-35B-A3B --local-dir <path>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export SSM_PRECISION_DTYPE=fp16_rtn  # FP16 round-to-nearest round-trip

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
