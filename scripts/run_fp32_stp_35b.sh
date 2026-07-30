#!/usr/bin/env bash
# FP32 SSM state, STP (no ReplaySSM), Qwen3.5-35B-A3B, TP=2
# Gold baseline: no precision emulation.
set -euo pipefail

IMG=${BENCH_IMG:-vllm/vllm-openai:nightly-6a9f24aa8cb856235528d01a829a4ba85fc1c19d}
MODEL=${MODEL_35B:-/tmp/models/Qwen3.5-35B-A3B}
GPUS=${CUDA_VISIBLE_DEVICES:-1,3}
PORT=${PORT:-8000}
CONTAINER=ssmprec_fp32_35b
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$MODEL" ]; then
    echo "ERROR: 35B model not found at $MODEL" >&2
    echo "Download: python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-35B-A3B', local_dir='$MODEL')\"" >&2
    exit 1
fi

docker rm -f "$CONTAINER" 2>/dev/null || true

exec docker run --name "$CONTAINER" \
  --gpus "\"device=$GPUS\"" \
  --network host --ipc host --shm-size 32g \
  -v /home/scratch.ameyn_gpu_2:/home/scratch.ameyn_gpu_2 \
  -v /tmp:/tmp \
  --entrypoint bash \
  "$IMG" -c "
    set -ex
    VLLM_PKG=\$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
    cp -r $REPO_ROOT/vllm/ssm_precision/ \"\$VLLM_PKG/ssm_precision/\"
    export SSM_PRECISION_DTYPE=fp32
    python3 $REPO_ROOT/scripts/patch_gdn_hook.py
    python3 -m vllm.entrypoints.openai.api_server \
      --model $MODEL \
      --tensor-parallel-size 2 \
      --dtype bfloat16 \
      --enforce-eager \
      --mamba-ssm-cache-dtype float32 \
      --no-enable-log-requests \
      --port $PORT
  "
