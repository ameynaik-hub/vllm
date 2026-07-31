# Eval Plan: GPQA + AIME via lm-eval

## What we're running

| Benchmark | lm-eval task | Questions |
|---|---|---|
| GPQA Diamond | `gpqa_diamond_cot_zeroshot` | 198 |
| AIME 2024 | `aime24` | 30 |
| AIME 2025 | `aime25` | 30 |

Two precision arms per model:
- **FP32**: gold baseline (`SSM_PRECISION_DTYPE=fp32`, hook is no-op)
- **FP16 RTN**: emulated (`SSM_PRECISION_DTYPE=fp16_rtn`, hook fires after each decode step)

Two models:
- `Qwen3.5-35B-A3B` — TP=2, GPUs 0,1
- `Qwen3.5-122B-A10B-NVFP4` — TP=2, GPUs 2,3

---

## How the hook works with lm-eval

`ssm_precision/run_lmeval.py` (already created) does:
```python
import ssm_precision.hook   # patches gdn.fused_recurrent_gated_delta_rule_packed_decode
import runpy
runpy.run_module("lm_eval.__main__", run_name="__main__", alter_sys=True)
```

The monkey-patch fires before lm-eval creates its `LLM()` instance, so every
GDN decode step goes through the wrapper. No vLLM source changes.

---

## Prerequisites (one-time)

### 1. Copy datasets to /tmp (Docker runs as root; NFS has root_squash)
```bash
mkdir -p /tmp/hf_home/datasets
cp -r /home/scratch.ameyn_gpu_2/hf_cache/datasets/Idavidrein___gpqa        /tmp/hf_home/datasets/
cp -r /home/scratch.ameyn_gpu_2/hf_cache/datasets/HuggingFaceH4___aime_2024 /tmp/hf_home/datasets/
cp -r /home/scratch.ameyn_gpu_2/hf_cache/datasets/MathArena___aime_2025     /tmp/hf_home/datasets/
```

### 2. Check GPU availability
```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```
Need: GPUs 0,1 free for 35B; GPUs 2,3 free for 122B.

---

## Run commands

### Variables (set once)
```bash
IMG=vllm/vllm-openai:nightly-6a9f24aa8cb856235528d01a829a4ba85fc1c19d
REPO=/home/scratch.ameyn_gpu_2/vllm-ssm-precision-study
SCRATCH=/home/scratch.ameyn_gpu_2
```

### Docker entrypoint template (inner bash -c block)
```bash
pip install -q 'lm-eval[api]' 2>&1 | grep -v notice
export PYTHONPATH=$REPO
export HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASETS_OFFLINE=1
export SSM_PRECISION_DTYPE=<fp32 or fp16_rtn>

python3 -m ssm_precision.run_lmeval \
  --model vllm \
  --model_args "pretrained=<MODEL>,tensor_parallel_size=<TP>,enforce_eager=True,\
dtype=bfloat16,mamba_ssm_cache_dtype=float32,gpu_memory_utilization=<UTIL>,\
max_model_len=32768,trust_remote_code=True" \
  --tasks gpqa_diamond_cot_zeroshot,aime24,aime25 \
  --apply_chat_template \
  --gen_kwargs "max_gen_toks=16384,temperature=0,do_sample=False" \
  --output_path $REPO/results/<MODEL_SHORT>/<PRECISION>/lm_eval \
  --log_samples \
  2>&1 | tee $REPO/results/<MODEL_SHORT>/<PRECISION>/lm_eval/run.log
```

### 35B FP32 (GPUs 0,1)
```bash
docker run -d --name eval_35b_fp32 \
  --gpus '"device=0,1"' --network host --ipc host --shm-size 32g \
  -v $SCRATCH:$SCRATCH -v /tmp:/tmp \
  -e HOME=/root \
  --entrypoint bash $IMG -c "
    pip install -q 'lm-eval[api]' 2>&1 | grep -v notice
    export PYTHONPATH=$REPO
    export HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASETS_OFFLINE=1
    export SSM_PRECISION_DTYPE=fp32
    python3 -m ssm_precision.run_lmeval \
      --model vllm \
      --model_args 'pretrained=$SCRATCH/models/Qwen3.5-35B-A3B,tensor_parallel_size=2,enforce_eager=True,dtype=bfloat16,mamba_ssm_cache_dtype=float32,gpu_memory_utilization=0.72,max_model_len=32768,trust_remote_code=True' \
      --tasks gpqa_diamond_cot_zeroshot,aime24,aime25 \
      --apply_chat_template \
      --gen_kwargs 'max_gen_toks=16384,temperature=0,do_sample=False' \
      --output_path $REPO/results/35b/fp32/lm_eval \
      --log_samples \
      2>&1 | tee $REPO/results/35b/fp32/lm_eval/run.log
  "
```

### 35B FP16 RTN (GPUs 0,1) — run after FP32 finishes
```bash
# same as above but: --name eval_35b_fp16rtn, SSM_PRECISION_DTYPE=fp16_rtn,
# --output_path .../fp16rtn/lm_eval
```

### 122B FP32 (GPUs 2,3)
```bash
# same template but:
#   pretrained=Qwen3.5-122B-A10B-NVFP4
#   tensor_parallel_size=2
#   gpu_memory_utilization=0.65
#   output_path .../122b/fp32/lm_eval
```

### 122B FP16 RTN (GPUs 2,3) — run after FP32 finishes
```bash
# SSM_PRECISION_DTYPE=fp16_rtn, output .../122b/fp16rtn/lm_eval
```

---

## Result files

lm-eval writes to `--output_path`:
```
results/
  35b/
    fp32/lm_eval/
      results_<timestamp>.json     ← scores summary
      samples_gpqa_diamond_*.jsonl ← per-question outputs
      samples_aime24_*.jsonl
      run.log
    fp16rtn/lm_eval/
      ...
  122b/
    fp32/lm_eval/
      ...
    fp16rtn/lm_eval/
      ...
```

Scores appear in `results_*.json` under:
```json
{
  "results": {
    "gpqa_diamond_cot_zeroshot": { "acc,none": 0.XX },
    "aime24":                    { "exact_match,none": X.X },
    "aime25":                    { "exact_match,none": X.X }
  }
}
```

---

## Known issues / gotchas

1. **GPU startup overhead**: Docker CUDA context + NCCL init uses ~44 GB at startup on B200.
   - 35B at TP=2: use `gpu_memory_utilization=0.72` (not 0.90)
   - 122B at TP=2: use `gpu_memory_utilization=0.65`

2. **NFS root_squash**: datasets must be copied to `/tmp/hf_home/datasets/` before running.
   Mount with `-v /tmp:/tmp` in Docker.

3. **Hook verification**: grep the run log for `[ssm_precision] monkey-patch active: fp16_rtn`
   — must appear for FP16 arms, must NOT appear for FP32 arms.

4. **Run 35B and 122B in parallel** on different GPU pairs to save wall-clock time.
   Run FP32 first, then FP16 RTN (same GPUs, sequentially).

---

## Timing estimate (per arm, k=1, temperature=0)

| Benchmark | Questions | Est. time at 600 tok/s |
|---|---|---|
| GPQA (198 × ~8K tokens) | 198 | ~45 min |
| AIME24 (30 × ~8K tokens) | 30 | ~7 min |
| AIME25 (30 × ~8K tokens) | 30 | ~7 min |
| **Total per arm** | | **~1 hour** |

4 arms (35B FP32, 35B FP16, 122B FP32, 122B FP16) run as 2 parallel pairs → **~2 hours total**.
