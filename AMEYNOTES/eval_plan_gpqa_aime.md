# Eval Plan: GPQA + AIME via lm-eval

## Settings (learned from runs)

**Use TP=2, not TP=1.**
TP=1 with 16K max tokens on a thinking model takes ~6-7h per arm. TP=2 halves that to ~3h.

---

## What we're running

| Benchmark | lm-eval task | Questions |
|---|---|---|
| GPQA Diamond | `gpqa_diamond_cot_zeroshot` | 198 |
| AIME 2024 | `aime24` | 30 |

Models:
- `Qwen3.5-35B-A3B` — BF16
- `Qwen3.5-122B-A10B-NVFP4` — NVFP4

Precision arms: `fp32`, `fp16_rtn`, `fp16_sr`, `fp8_rtn`, `fp8_sr`

---

## GPU layout (TP=2, 8 GPUs → 4 arms at once)

With TP=2 each arm needs 2 GPUs. 8 GPUs → 4 arms in parallel. Run 6 arms in 2 rounds:

**Round 1** (4 arms):
```
GPUs 0,1: 35B  precision_A
GPUs 2,3: 35B  precision_B
GPUs 4,5: 122B precision_A
GPUs 6,7: 122B precision_B
```

**Round 2** (2 arms):
```
GPUs 0,1: 35B  precision_C
GPUs 4,5: 122B precision_C
```

---

## Prerequisites (one-time)

```bash
mkdir -p /tmp/hf_home/datasets
cp -r /home/scratch.ameyn_gpu_2/hf_cache/datasets/Idavidrein___gpqa         /tmp/hf_home/datasets/
cp -r /home/scratch.ameyn_gpu_2/hf_cache/datasets/HuggingFaceH4___aime_2024  /tmp/hf_home/datasets/
```

---

## Run template

```bash
IMG=vllm/vllm-openai:nightly-6a9f24aa8cb856235528d01a829a4ba85fc1c19d
REPO=/home/scratch.ameyn_gpu_2/vllm-ssm-precision-study
SCRATCH=/home/scratch.ameyn_gpu_2

docker run -d --name <NAME> \
  --gpus '"device=<GPU0>,<GPU1>"' \
  --network host --ipc host --shm-size 64g \
  -v $SCRATCH:$SCRATCH -v /tmp:/tmp \
  -e HOME=/root -e SSM_PRECISION_DTYPE=<PRECISION> \
  --entrypoint bash $IMG -c "
    pip install -q 'lm-eval[api]' 2>&1 | grep -v notice
    SITELIB=\$(python3 -c 'import site; print(site.getsitepackages()[0])')
    cp -r $REPO/ssm_precision/ \$SITELIB/ssm_precision/
    export HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASETS_OFFLINE=1
    python3 -m ssm_precision.run_lmeval \
      --model vllm \
      --model_args 'pretrained=<MODEL>,tensor_parallel_size=2,enforce_eager=True,dtype=bfloat16,mamba_ssm_cache_dtype=float32,gpu_memory_utilization=0.85,max_model_len=32768,trust_remote_code=True' \
      --tasks gpqa_diamond_cot_zeroshot,aime24 \
      --apply_chat_template \
      --gen_kwargs 'max_gen_toks=16384,temperature=0,do_sample=False' \
      --output_path $REPO/results/<MODEL_SHORT>/<PRECISION>/lm_eval --log_samples \
      2>&1 | tee $REPO/results/<MODEL_SHORT>/<PRECISION>/lm_eval/run.log
  "
```

Key flags vs old plan:
- `tensor_parallel_size=2` ← was 1
- `gpu_memory_utilization=0.85` ← was 0.90 (TP=2 needs NCCL init buffer)
- `max_gen_toks=16384` ← unchanged
- `--shm-size 64g` ← was 32g (needed for NCCL)

---

## Full Round 1 commands (4 arms, copy-paste)

```bash
IMG=vllm/vllm-openai:nightly-6a9f24aa8cb856235528d01a829a4ba85fc1c19d
REPO=/home/scratch.ameyn_gpu_2/vllm-ssm-precision-study
SCRATCH=/home/scratch.ameyn_gpu_2
M35=$SCRATCH/models/Qwen3.5-35B-A3B
M122=$SCRATCH/models/Qwen3.5-122B-A10B-NVFP4

for cfg in \
  "eval_35b_fp32:0,1:$M35:fp32:35b/fp32" \
  "eval_35b_fp16rtn:2,3:$M35:fp16_rtn:35b/fp16rtn" \
  "eval_122b_fp32:4,5:$M122:fp32:122b/fp32" \
  "eval_122b_fp16rtn:6,7:$M122:fp16_rtn:122b/fp16rtn"; do
  IFS=: read -r name gpus model prec outkey <<< "$cfg"
  outdir=$REPO/results/$outkey/lm_eval
  mkdir -p $outdir
  docker rm -f $name 2>/dev/null || true
  docker run -d --name $name \
    --gpus "\"device=$gpus\"" \
    --network host --ipc host --shm-size 64g \
    -v $SCRATCH:$SCRATCH -v /tmp:/tmp \
    -e HOME=/root -e SSM_PRECISION_DTYPE=$prec \
    --entrypoint bash $IMG -c "
      pip install -q 'lm-eval[api]' 2>&1 | grep -v notice
      SITELIB=\$(python3 -c 'import site; print(site.getsitepackages()[0])')
      cp -r $REPO/ssm_precision/ \$SITELIB/ssm_precision/
      export HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASETS_OFFLINE=1
      python3 -m ssm_precision.run_lmeval \
        --model vllm \
        --model_args 'pretrained=$model,tensor_parallel_size=2,enforce_eager=True,dtype=bfloat16,mamba_ssm_cache_dtype=float32,gpu_memory_utilization=0.85,max_model_len=32768,trust_remote_code=True' \
        --tasks gpqa_diamond_cot_zeroshot,aime24 \
        --apply_chat_template \
        --gen_kwargs 'max_gen_toks=16384,temperature=0,do_sample=False' \
        --output_path $outdir --log_samples \
        2>&1 | tee $outdir/run.log
      echo DONE_$name
    " >> $REPO/results/${name}_tp2.log 2>&1
  echo "Launched $name on GPUs $gpus"
done
```

---

## Notes

- **TP=2 + NCCL startup memory**: expect ~22-44 GB used at startup (not model weights). Use `gpu_memory_utilization=0.85` to leave room. With 2× B200s at 183 GB each, 35B (35 GB/GPU) and 122B (30 GB/GPU) both fit easily.
- **Results dir**: lm-eval saves to `results/<model>/<precision>/lm_eval/__home__...__model/`
- **Hook verification**: `docker logs <name> 2>&1 | grep ssm_precision` — must appear for non-fp32 arms

---

## Timing (TP=2, max_gen_toks=8192)

| Stage | Est. time |
|---|---|
| Model load | ~4 min |
| GPQA 198 × ~16K tokens at ~1000 tok/s | ~60 min |
| AIME24 30 × ~16K tokens | ~10 min |
| **Total per arm** | **~75 min** |
| 2 rounds of 4 arms | **~2.5h total** |
