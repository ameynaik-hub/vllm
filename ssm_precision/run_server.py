"""Drop-in launcher: applies SSM precision monkey-patch then runs vLLM server.

Replace:
    python3 -m vllm.entrypoints.openai.api_server --model ...

With:
    PYTHONPATH=/path/to/vllm-ssm-precision-study \
    SSM_PRECISION_DTYPE=fp16_rtn \
    python3 -m ssm_precision.run_server --model ... --enforce-eager --mamba-ssm-cache-dtype float32

The patch is applied before vLLM loads the model, so all decode steps
go through the wrapped function.
"""

import ssm_precision.hook  # noqa: F401 — side-effect: installs monkey-patch

import runpy
runpy.run_module(
    "vllm.entrypoints.openai.api_server",
    run_name="__main__",
    alter_sys=True,
)
