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

Two models, each on a **single GPU (TP=1)**:
- `Qwen3.5-35B-A3B` — 70 GB BF16, fits on one B200 (183 GB)
- `Qwen3.5-122B-A10B-NVFP4` — 61 GB NVFP4, fits on one B200 (183 GB)

All 4 arms run simultaneously on 4 separate GPUs (~1h wall-clock total).

```
GPU 0: 35B  FP32     ─┐
GPU 1: 35B  FP16 RTN ─┤  all at once
GPU 2: 122B FP32     ─┤
GPU 3: 122B FP16 RTN ─┘
```

---

## How the hook works with lm-eval

`ssm_precision/run_lmeval.py` does:
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
cp -r /home/scratch.ameyn_gpu_2/hf_cache/datasets/Idavidrein___gpqa         /tmp/hf_home/datasets/
cp -r /home/scratch.ameyn_gpu_2/hf_cache/datasets/HuggingFaceH4___aime_2024  /tmp/hf_home/datasets/
cp -r /home/scratch.ameyn_gpu_2/hf_cache/datasets/MathArena___aime_2025      /tmp/hf_home/datasets/
```

### 2. Check GPU availability
```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```
Need GPUs 0, 1, 2, 3 free (one per arm).

---

## Run commands

```bash
IMG=vllm/vllm-openai:nightly-6a9f24aa8cb856235528d01a829a4ba85fc1c19d
REPO=/home/scratch.ameyn_gpu_2/vllm-ssm-precision-study
SCRATCH=/home/scratch.ameyn_gpu_2
```

### 35B FP32 — GPU 0
```bash
docker run -d --name eval_35b_fp32 \
  --gpus '"device=0"' --network host --ipc host --shm-size 32g \
  -v $SCRATCH:$SCRATCH -v /tmp:/tmp -e HOME=/root \
  --entrypoint bash $IMG -c "
    pip install -q 'lm-eval[api]' 2>&1 | grep -v notice
    export PYTHONPATH=$REPO
    export HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASETS_OFFLINE=1
    export SSM_PRECISION_DTYPE=fp32
    python3 -m ssm_precision.run_lmeval \
      --model vllm \
      --model_args 'pretrained=$SCRATCH/models/Qwen3.5-35B-A3B,enforce_eager=True,dtype=bfloat16,mamba_ssm_cache_dtype=float32,gpu_memory_utilization=0.90,max_model_len=32768,trust_remote_code=True' \
      --tasks gpqa_diamond_cot_zeroshot,aime24,aime25 \
      --apply_chat_template \
      --gen_kwargs 'max_gen_toks=16384,temperature=0,do_sample=False' \
      --output_path $REPO/results/35b/fp32/lm_eval --log_samples \
      2>&1 | tee $REPO/results/35b/fp32/lm_eval/run.log
  "
```

### 35B FP16 RTN — GPU 1
```bash
docker run -d --name eval_35b_fp16rtn \
  --gpus '"device=1"' --network host --ipc host --shm-size 32g \
  -v $SCRATCH:$SCRATCH -v /tmp:/tmp -e HOME=/root \
  --entrypoint bash $IMG -c "
    pip install -q 'lm-eval[api]' 2>&1 | grep -v notice
    export PYTHONPATH=$REPO
    export HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASETS_OFFLINE=1
    export SSM_PRECISION_DTYPE=fp16_rtn
    python3 -m ssm_precision.run_lmeval \
      --model vllm \
      --model_args 'pretrained=$SCRATCH/models/Qwen3.5-35B-A3B,enforce_eager=True,dtype=bfloat16,mamba_ssm_cache_dtype=float32,gpu_memory_utilization=0.90,max_model_len=32768,trust_remote_code=True' \
      --tasks gpqa_diamond_cot_zeroshot,aime24,aime25 \
      --apply_chat_template \
      --gen_kwargs 'max_gen_toks=16384,temperature=0,do_sample=False' \
      --output_path $REPO/results/35b/fp16rtn/lm_eval --log_samples \
      2>&1 | tee $REPO/results/35b/fp16rtn/lm_eval/run.log
  "
```

### 122B FP32 — GPU 2
```bash
docker run -d --name eval_122b_fp32 \
  --gpus '"device=2"' --network host --ipc host --shm-size 64g \
  -v $SCRATCH:$SCRATCH -v /tmp:/tmp -e HOME=/root \
  --entrypoint bash $IMG -c "
    pip install -q 'lm-eval[api]' 2>&1 | grep -v notice
    export PYTHONPATH=$REPO
    export HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASETS_OFFLINE=1
    export SSM_PRECISION_DTYPE=fp32
    python3 -m ssm_precision.run_lmeval \
      --model vllm \
      --model_args 'pretrained=$SCRATCH/models/Qwen3.5-122B-A10B-NVFP4,enforce_eager=True,dtype=bfloat16,mamba_ssm_cache_dtype=float32,gpu_memory_utilization=0.90,max_model_len=32768,trust_remote_code=True' \
      --tasks gpqa_diamond_cot_zeroshot,aime24,aime25 \
      --apply_chat_template \
      --gen_kwargs 'max_gen_toks=16384,temperature=0,do_sample=False' \
      --output_path $REPO/results/122b/fp32/lm_eval --log_samples \
      2>&1 | tee $REPO/results/122b/fp32/lm_eval/run.log
  "
```

### 122B FP16 RTN — GPU 3
```bash
docker run -d --name eval_122b_fp16rtn \
  --gpus '"device=3"' --network host --ipc host --shm-size 64g \
  -v $SCRATCH:$SCRATCH -v /tmp:/tmp -e HOME=/root \
  --entrypoint bash $IMG -c "
    pip install -q 'lm-eval[api]' 2>&1 | grep -v notice
    export PYTHONPATH=$REPO
    export HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASETS_OFFLINE=1
    export SSM_PRECISION_DTYPE=fp16_rtn
    python3 -m ssm_precision.run_lmeval \
      --model vllm \
      --model_args 'pretrained=$SCRATCH/models/Qwen3.5-122B-A10B-NVFP4,enforce_eager=True,dtype=bfloat16,mamba_ssm_cache_dtype=float32,gpu_memory_utilization=0.90,max_model_len=32768,trust_remote_code=True' \
      --tasks gpqa_diamond_cot_zeroshot,aime24,aime25 \
      --apply_chat_template \
      --gen_kwargs 'max_gen_toks=16384,temperature=0,do_sample=False' \
      --output_path $REPO/results/122b/fp16rtn/lm_eval --log_samples \
      2>&1 | tee $REPO/results/122b/fp16rtn/lm_eval/run.log
  "
```

---

## Monitor

```bash
# Status
docker ps --filter "name=eval_" --format "table {{.Names}}\t{{.Status}}"

# Hook verification (fp16 arms only)
docker logs eval_35b_fp16rtn  2>&1 | grep ssm_precision
docker logs eval_122b_fp16rtn 2>&1 | grep ssm_precision
# Expected: [ssm_precision] monkey-patch active: fp16_rtn

# Progress
docker logs eval_35b_fp32 2>&1 | grep -E "Running|acc|score|Error" | tail -5
```

---

## Results location

```
results/
  35b/  fp32/lm_eval/results_*.json   ← scores
        fp16rtn/lm_eval/results_*.json
  122b/ fp32/lm_eval/results_*.json
        fp16rtn/lm_eval/results_*.json
```

Key fields in `results_*.json`:
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

## Gotchas

1. **NFS root_squash**: copy datasets to `/tmp/hf_home/` before running (see Prerequisites).
2. **Hook check**: FP32 arms must NOT show `[ssm_precision]`; FP16 arms must show it.
3. **TP=1, gpu_memory_utilization=0.90**: no NCCL overhead, clean startup on a free B200.
4. **Results dir must be world-writable** (Docker writes as root):
   ```bash
   chmod -R 777 /home/scratch.ameyn_gpu_2/vllm-ssm-precision-study/results/
   ```

---

## Timing estimate (TP=1, temperature=0)

| Benchmark | Questions | Est. time |
|---|---|---|
| GPQA (198 × ~8K tokens) | 198 | ~45 min |
| AIME24 + AIME25 (60 total) | 60 | ~15 min |
| **Total per arm** | | **~1 hour** |

All 4 arms run in parallel → **~1 hour wall-clock total**.
