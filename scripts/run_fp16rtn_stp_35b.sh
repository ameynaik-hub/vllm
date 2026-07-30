#!/usr/bin/env bash
# FP16 RTN emulated SSM state, STP (no ReplaySSM), Qwen3.5-35B-A3B, TP=2
# NOTE: 35B model not on local disk. See run_fp32_stp_35b.sh for download instructions.
set -euo pipefail

IMG=${BENCH_IMG:-vllm/vllm-openai:nightly-6a9f24aa8cb856235528d01a829a4ba85fc1c19d}
MODEL=${MODEL_35B:-/home/scratch.ameyn_gpu_2/models/Qwen3.5-35B-A3B}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1}
PORT=${PORT:-8001}
CONTAINER=ssmprec_fp16rtn_35b
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$MODEL" ]; then
    echo "ERROR: 35B model not found at $MODEL" >&2
    echo "Download: huggingface-cli download Qwen/Qwen3.5-35B-A3B --local-dir $MODEL" >&2
    exit 1
fi

docker rm -f "$CONTAINER" 2>/dev/null || true

exec docker run --name "$CONTAINER" \
  --gpus "\"device=$GPUS\"" \
  --network host --ipc host --shm-size 64g \
  -v /home/scratch.ameyn_gpu_2:/home/scratch.ameyn_gpu_2 \
  --entrypoint bash \
  "$IMG" -c "
    set -ex
    VLLM_PKG=\$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
    cp -r $REPO_ROOT/vllm/. \"\$VLLM_PKG/\"
    export PYTHONPATH=$REPO_ROOT:\${PYTHONPATH:-}
    export SSM_PRECISION_DTYPE=fp16_rtn
    python3 -m vllm.entrypoints.openai.api_server \
      --model $MODEL \
      --tensor-parallel-size 2 \
      --dtype bfloat16 \
      --enforce-eager \
      --mamba-ssm-cache-dtype float32 \
      --no-enable-log-requests \
      --port $PORT
  "
