#!/usr/bin/env python3
"""Patch qwen_gdn_linear_attn.py in the installed vllm package to add the
SSM precision emulation hook. Run once inside the Docker container before
starting the server.

The patch is idempotent: if the hook is already present, it exits cleanly.
"""
import os
import sys
import importlib.util
import shutil

# Find the installed vllm package path
spec = importlib.util.find_spec("vllm")
if spec is None:
    sys.exit("ERROR: vllm not found on sys.path")
vllm_dir = os.path.dirname(spec.origin)
target = os.path.join(
    vllm_dir,
    "model_executor", "layers", "mamba", "gdn", "qwen_gdn_linear_attn.py"
)

if not os.path.exists(target):
    sys.exit(f"ERROR: target file not found: {target}")

with open(target) as f:
    src = f.read()

# Idempotency check
if "ssm_precision" in src:
    print(f"[patch_gdn_hook] Hook already present in {target}, skipping.")
    sys.exit(0)

HOOK_IMPORTS = """\
import os as _os  # noqa: E402
from vllm.ssm_precision.state_quantizer import StateQuantizer as _SQ  # noqa: E402
_GDN_PRECISION = _os.environ.get("SSM_PRECISION_DTYPE", "fp32")
_gdn_quantizer = _SQ.from_str(_GDN_PRECISION)
if _gdn_quantizer is not None:
    print(f"[ssm_precision] GDN state hook active: {_GDN_PRECISION}", flush=True)
"""

HOOK_CALL = """\
        # Emulated state precision: apply round-trip after kernel writes state in FP32.
        if _gdn_quantizer is not None:
            idx = non_spec_state_indices_tensor[:num_actual_tokens]
            ssm_state[idx] = _gdn_quantizer(ssm_state[idx].float())
"""

# --- Insert hook imports after logger = init_logger(__name__) ---
LOGGER_LINE = "logger = init_logger(__name__)"
if LOGGER_LINE not in src:
    sys.exit(f"ERROR: anchor '{LOGGER_LINE}' not found in {target}")

src = src.replace(
    LOGGER_LINE,
    LOGGER_LINE + "\n\n" + HOOK_IMPORTS,
    1,
)

# --- Insert hook call after packed-decode kernel call ---
# The packed-decode call ends with a line containing use_qk_l2norm_in_kernel=True,
# followed by the closing ) and then `return`.
# We look for the closing paren + return that belongs to _forward_core_decode_non_spec.
PACKED_DECODE_ANCHOR = "            use_qk_l2norm_in_kernel=True,\n        )\n        return"
if PACKED_DECODE_ANCHOR not in src:
    # Try without use_qk_l2norm_in_kernel (older versions may not have it)
    PACKED_DECODE_ANCHOR2 = "            initial_state=ssm_state,\n        )\n        return"
    if PACKED_DECODE_ANCHOR2 not in src:
        sys.exit("ERROR: could not find packed-decode anchor in source. "
                 "Inspect the file manually and update this script.")
    PACKED_DECODE_ANCHOR = PACKED_DECODE_ANCHOR2

src = src.replace(
    PACKED_DECODE_ANCHOR,
    PACKED_DECODE_ANCHOR.rstrip("        return") + "\n" + HOOK_CALL + "        return",
    1,
)

# Make a backup
shutil.copy2(target, target + ".orig")

with open(target, "w") as f:
    f.write(src)

print(f"[patch_gdn_hook] Patched {target}")
print(f"[patch_gdn_hook] SSM_PRECISION_DTYPE={os.environ.get('SSM_PRECISION_DTYPE', 'fp32')}")
