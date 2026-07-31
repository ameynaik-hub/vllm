"""SSM state precision monkey-patch for GDN (Gated DeltaNet) decode.

Equivalent of Johnny's Mamba2 pattern applied to GDN:
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn
    gdn.fused_recurrent_gated_delta_rule_packed_decode = _wrapped

Zero vLLM source changes. The Triton kernel runs unchanged; after it writes
the state in FP32, the wrapper applies the precision round-trip in PyTorch.

Install this package into the Python environment's site-packages so it is
importable without PYTHONPATH — avoids shadowing the installed vllm:
    cp -r ssm_precision/ $(python3 -c "import site; print(site.getsitepackages()[0])")/
"""

import os

from ssm_precision.state_quantizer import StateQuantizer

_precision = os.environ.get("SSM_PRECISION_DTYPE", "fp32")
_quantizer = StateQuantizer.from_str(_precision)

if _quantizer is not None:
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as _gdn

    _orig = _gdn.fused_recurrent_gated_delta_rule_packed_decode

    def _wrapped(*args, **kwargs):
        _orig(*args, **kwargs)
        ssm_state = kwargs.get("initial_state")
        idx = kwargs.get("ssm_state_indices")
        if ssm_state is not None and idx is not None:
            ssm_state[idx] = _quantizer(ssm_state[idx].float())

    _gdn.fused_recurrent_gated_delta_rule_packed_decode = _wrapped
    print(f"[ssm_precision] monkey-patch active: {_precision}", flush=True)
