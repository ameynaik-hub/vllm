# SSM Precision Study — Session Summary

## 1. Repo

**`vllm-ssm-precision-study`** at `/home/scratch.ameyn_gpu_2/vllm-ssm-precision-study`

- Git worktree of `/home/scratch.ameyn_gpu_2/vllm_replayssm` (the main ReplaySSM dev repo)
- Branch: `ssm-precision-fp16-emu`
- Based on: PR #49887 head (`5b96cdb` — `[Bench] ReplaySSM: route GDN prefill through Triton`)
- Remote: `ameynaik-hub/vllm` → `ssm-precision-fp16-emu`
- Model weights:
  - Qwen3.5-35B-A3B: `/home/scratch.ameyn_gpu_2/models/Qwen3.5-35B-A3B` (68 GB BF16)
  - Qwen3.5-122B-A10B-NVFP4: `/home/scratch.ameyn_gpu_2/models/Qwen3.5-122B-A10B-NVFP4`

---

## 2. Where the Changes Are

```
vllm-ssm-precision-study/
├── vllm/
│   ├── ssm_precision/                          ← NEW subpackage (copied into vllm namespace)
│   │   ├── __init__.py
│   │   └── state_quantizer.py                 ← core emulation logic
│   └── model_executor/layers/mamba/gdn/
│       └── qwen_gdn_linear_attn.py            ← MODIFIED (+15 lines)
├── ssm_precision/                              ← same files, top-level (for PYTHONPATH usage)
│   ├── __init__.py
│   └── state_quantizer.py
├── scripts/
│   ├── patch_gdn_hook.py                      ← runtime patcher for Docker deployments
│   ├── run_fp32_stp_35b.sh                    ← server launch: 35B FP32 STP
│   ├── run_fp16rtn_stp_35b.sh                 ← server launch: 35B FP16 RTN STP
│   ├── run_fp32_stp_122b.sh                   ← server launch: 122B FP32 STP
│   ├── run_fp16rtn_stp_122b.sh                ← server launch: 122B FP16 RTN STP
│   └── launch_fp16rtn_evals.sh                ← launches FP16 RTN eval containers
├── AMEYNOTES/
│   ├── quack_commit_explanation.md            ← explanation of Dao-AILab/quack 709b829
│   └── session_summary.md                     ← this file
└── results/
    ├── 35b/{fp32,fp16rtn}/                    ← eval outputs (qwen_eval.py + lm-eval)
    └── 122b/{fp32,fp16rtn}/
```

---

## 3. What the Changes Are

### Goal
Emulate lower-precision SSM state storage **without touching Triton kernels**. The GDN (Gated DeltaNet) hidden state `h` is written to a FP32 cache by the Triton kernel `fused_recurrent_gated_delta_rule_packed_decode_kernel`. After the kernel returns to Python, we apply a round-trip to emulate what a lower-precision cache would do.

This is **emulated quantization** (not a real hardware change): FP32 compute, FP32 stored, then immediately overwritten with the round-tripped value.

### `vllm/ssm_precision/state_quantizer.py` (new)
Pure-PyTorch `StateQuantizer` class. Adapted from Johnny's (Ze-Wei Liou's) `fp8_sr_replayssm.py` (manually verified).

Modes via `StateQuantizer.from_str(env_str)`:
| `SSM_PRECISION_DTYPE` | Effect |
|---|---|
| `fp32` | Returns `None` — no-op, hook skipped entirely |
| `fp16_rtn` | `x.half().float()` — FP16 round-to-nearest |
| `fp16_sr` | FP16 stochastic rounding via ULP-scaled dither |
| `fp8_sr` | Block-128 scaled e4m3 + SR (future FP8 pass) |

### `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (modified)
Two additions, ~15 lines total:

**Module-level** (after `logger = init_logger(__name__)`):
```python
import os as _os
from vllm.ssm_precision.state_quantizer import StateQuantizer as _SQ
_GDN_PRECISION = _os.environ.get("SSM_PRECISION_DTYPE", "fp32")
_gdn_quantizer = _SQ.from_str(_GDN_PRECISION)
if _gdn_quantizer is not None:
    print(f"[ssm_precision] GDN state hook active: {_GDN_PRECISION}", flush=True)
```

**In `_forward_core_decode_non_spec`** (after `fused_recurrent_gated_delta_rule_packed_decode` returns):
```python
if _gdn_quantizer is not None:
    idx = non_spec_state_indices_tensor[:num_actual_tokens]
    ssm_state[idx] = _gdn_quantizer(ssm_state[idx].float())
```

**Why `_forward_core_decode_non_spec`**: This is the STP (standard per-step, no ReplaySSM) decode path. The Triton kernel updates `ssm_state[active_slots]` in-place; the hook runs immediately after on the Python side. ReplaySSM paths are untouched.

**Required vLLM flags**:
- `--enforce-eager` — disables CUDA graph capture so the Python hook stays in the hot path
- `--mamba-ssm-cache-dtype float32` — vLLM stores the state in FP32; our hook provides the precision emulation

### `scripts/patch_gdn_hook.py` (new)
Since the vLLM Docker image has precompiled `.so` files and a different Python version of `qwen_gdn_linear_attn.py`, we can't simply overlay the whole worktree. This script patches the image's installed copy in-place at container startup. Idempotent (skips if already patched).

---

## 4. What We Were Running

### Study design
Compare **FP32** (gold baseline) vs **FP16 RTN** (emulated) SSM state on:
- **GPQA Diamond** (198 questions, lm-eval `gpqa_diamond_cot_zeroshot`)
- **AIME 2024** (30 problems, lm-eval `aime24`)
- **HMMT Feb 2025** (30 problems, `qwen_eval.py` — not in lm-eval)

Both models in **STP mode** (no ReplaySSM). Gen settings: temperature=0, max_tokens=16384.

### Verified (smoke tests passed)
| Model | Precision | GPUs | Port | Status |
|---|---|---|---|---|
| 35B-A3B | FP32 | 1,3 | 8000 | ✅ Verified — `[ssm_precision]` absent (correct) |
| 35B-A3B | FP16 RTN | 1,3 | 8001 | ✅ Verified — hook fires on all TP ranks |
| 122B-NVFP4 | FP32 | 1,3,4,5 | 8000 | ✅ Verified |
| 122B-NVFP4 | FP16 RTN | 1,3,4,5 | 8001 | ✅ Verified — hook fires on all TP ranks |

### Eval runs (incomplete at session end)
Full eval run was started multiple times but stopped. The infrastructure works; the runs just need to complete.

---

## 5. Commands to Run on GPU

### One-time setup (already done)
```bash
# The worktree already exists. Branch already pushed to fork.
# Model weights already present:
ls /home/scratch.ameyn_gpu_2/models/
# Qwen3.5-35B-A3B, Qwen3.5-122B-A10B-NVFP4, Qwen3.5-27B-FP8, Qwen3.5-397B-A17B-FP8
```

### Run FP32 evals (both models in parallel)

```bash
IMG=vllm/vllm-openai:nightly-6a9f24aa8cb856235528d01a829a4ba85fc1c19d
REPO=/home/scratch.ameyn_gpu_2/vllm-ssm-precision-study
HF=/home/scratch.ameyn_gpu_2/hf_cache
SCRATCH=/home/scratch.ameyn_gpu_2

# 35B FP32 — GPUs 0,1
docker run -d --name eval_35b_fp32 \
  --gpus '"device=0,1"' \
  --network host --ipc host --shm-size 32g \
  -v $SCRATCH:$SCRATCH \
  -e SSM_PRECISION_DTYPE=fp32 -e HOME=/root \
  --entrypoint bash "$IMG" -c "
    set -ex
    VLLM_PKG=\$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
    cp -r $REPO/vllm/ssm_precision/ \"\$VLLM_PKG/ssm_precision/\"
    python3 $REPO/scripts/patch_gdn_hook.py
    pip install -q 'lm-eval[api]' 2>&1 | grep -v notice
    mkdir -p /tmp/hf_home/datasets
    cp -r $HF/datasets/Idavidrein___gpqa        /tmp/hf_home/datasets/
    cp -r $HF/datasets/HuggingFaceH4___aime_2024 /tmp/hf_home/datasets/
    cp -r $HF/datasets/MathArena___hmmt_feb_2025  /tmp/hf_home/datasets/
    export HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASETS_OFFLINE=1
    MODEL=$SCRATCH/models/Qwen3.5-35B-A3B
    python3 -m lm_eval \
      --model vllm \
      --model_args \"pretrained=\$MODEL,tensor_parallel_size=2,enforce_eager=True,dtype=bfloat16,mamba_ssm_cache_dtype=float32,gpu_memory_utilization=0.72,max_model_len=32768,trust_remote_code=True\" \
      --tasks gpqa_diamond_cot_zeroshot,aime24 \
      --apply_chat_template \
      --gen_kwargs 'max_gen_toks=16384,temperature=0,do_sample=False' \
      --output_path $REPO/results/35b/fp32/lm_eval \
      --log_samples 2>&1 | tee $REPO/results/35b/fp32/lm_eval/run.log
    python3 $SCRATCH/qwen_eval.py \
      --model-id \$MODEL --dataset hmmt --modes standard --k 1 --max-tokens 16384 \
      --tensor-parallel-size 2 --enforce-eager \
      --mamba-ssm-cache-dtype float32 --gpu-memory-utilization 0.72 \
      --out-dir $REPO/results/35b/fp32
    echo EVAL_DONE_35B_FP32
  "

# 122B FP32 — GPUs 2,3
docker run -d --name eval_122b_fp32 \
  --gpus '"device=2,3"' \
  --network host --ipc host --shm-size 64g \
  -v $SCRATCH:$SCRATCH \
  -e SSM_PRECISION_DTYPE=fp32 -e HOME=/root \
  --entrypoint bash "$IMG" -c "
    set -ex
    VLLM_PKG=\$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
    cp -r $REPO/vllm/ssm_precision/ \"\$VLLM_PKG/ssm_precision/\"
    python3 $REPO/scripts/patch_gdn_hook.py
    pip install -q 'lm-eval[api]' 2>&1 | grep -v notice
    mkdir -p /tmp/hf_home/datasets
    cp -r $HF/datasets/Idavidrein___gpqa        /tmp/hf_home/datasets/
    cp -r $HF/datasets/HuggingFaceH4___aime_2024 /tmp/hf_home/datasets/
    cp -r $HF/datasets/MathArena___hmmt_feb_2025  /tmp/hf_home/datasets/
    export HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASETS_OFFLINE=1
    MODEL=$SCRATCH/models/Qwen3.5-122B-A10B-NVFP4
    python3 -m lm_eval \
      --model vllm \
      --model_args \"pretrained=\$MODEL,tensor_parallel_size=2,enforce_eager=True,dtype=bfloat16,mamba_ssm_cache_dtype=float32,gpu_memory_utilization=0.65,max_model_len=32768,trust_remote_code=True\" \
      --tasks gpqa_diamond_cot_zeroshot,aime24 \
      --apply_chat_template \
      --gen_kwargs 'max_gen_toks=16384,temperature=0,do_sample=False' \
      --output_path $REPO/results/122b/fp32/lm_eval \
      --log_samples 2>&1 | tee $REPO/results/122b/fp32/lm_eval/run.log
    python3 $SCRATCH/qwen_eval.py \
      --model-id \$MODEL --dataset hmmt --modes standard --k 1 --max-tokens 16384 \
      --tensor-parallel-size 2 --enforce-eager \
      --mamba-ssm-cache-dtype float32 --gpu-memory-utilization 0.65 \
      --max-model-len 32768 \
      --out-dir $REPO/results/122b/fp32
    echo EVAL_DONE_122B_FP32
  "
```

### Run FP16 RTN evals (after FP32 completes)
Identical to above but:
- `SSM_PRECISION_DTYPE=fp16_rtn`
- `--name eval_35b_fp16rtn` / `--name eval_122b_fp16rtn`
- `--output_path .../fp16rtn/lm_eval`
- `--out-dir .../fp16rtn`

Or use the pre-written script (needs updating to match above Docker command structure):
```bash
bash $REPO/scripts/launch_fp16rtn_evals.sh
```

### Monitor progress
```bash
docker logs -f eval_35b_fp32  2>&1 | grep -E "acc|score|Error|DONE|Running"
docker logs -f eval_122b_fp32 2>&1 | grep -E "acc|score|Error|DONE|Running"
```

### Verify hook is active (FP16 run)
```bash
docker logs eval_35b_fp16rtn 2>&1 | grep ssm_precision
# Should show: [ssm_precision] GDN state hook active: fp16_rtn
```

### GPU memory notes
- Docker CUDA context uses ~44 GB at startup (NCCL init overhead on B200)
- 35B BF16 at TP=2: use `--gpu-memory-utilization 0.72` (fits on clean GPUs)
- 122B NVFP4 at TP=2: use `--gpu-memory-utilization 0.65`
- Available clean GPUs on this node: check with `nvidia-smi` first; avoid GPU 5 which often has external load

---

## Key Files to Know

| Path | Purpose |
|---|---|
| `vllm/ssm_precision/state_quantizer.py` | The quantizer — FP16/FP8/SR modes |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:87-92,1689-1692` | The hook insertion points |
| `scripts/patch_gdn_hook.py` | Applies hook to Docker image's installed vllm |
| `AMEYNOTES/quack_commit_explanation.md` | Explains Dao-AILab/quack 709b829 (SW-emulated SR for SM80+) |
| `results/{35b,122b}/{fp32,fp16rtn}/lm_eval/` | lm-eval JSON outputs |
| `results/{35b,122b}/{fp32,fp16rtn}/*/hmmt/` | qwen_eval.py HMMT outputs |
