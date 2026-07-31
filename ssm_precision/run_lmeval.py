"""Drop-in launcher: applies SSM precision monkey-patch then runs lm-eval.

Replace:
    python3 -m lm_eval --model vllm ...

With:
    PYTHONPATH=/path/to/vllm-ssm-precision-study \
    SSM_PRECISION_DTYPE=fp16_rtn \
    python3 -m ssm_precision.run_lmeval --model vllm ...
"""

import ssm_precision.hook  # noqa: F401 — side-effect: installs monkey-patch

import runpy
runpy.run_module("lm_eval.__main__", run_name="__main__", alter_sys=True)
