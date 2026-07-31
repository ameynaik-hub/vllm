"""Drop-in launcher: applies SSM precision monkey-patch then runs lm-eval.

Requires ssm_precision to be installed in site-packages (not via PYTHONPATH):
    cp -r ssm_precision/ $(python3 -c "import site; print(site.getsitepackages()[0])")/

Then run as:
    SSM_PRECISION_DTYPE=fp16_rtn python3 -m ssm_precision.run_lmeval --model vllm ...
"""

import ssm_precision.hook  # noqa: F401 — side-effect: installs monkey-patch

import runpy
runpy.run_module("lm_eval.__main__", run_name="__main__", alter_sys=True)
