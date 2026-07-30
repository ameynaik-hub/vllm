# Quack Commit 709b829: Software-Emulated Stochastic Rounding

**Commit**: [`709b829`](https://github.com/Dao-AILab/quack/commit/709b829e6f873c1cb5c8866fc945bd3447be3fae)
**Title**: Add software emulated stochastic rounding
**Repo**: Dao-AILab/quack (CUTLASS/CuTe DSL utility library)
**Files changed**: `quack/rounding.py`, `quack/copy_utils.py`, `tests/test_linear.py`

---

## What Problem This Solves

Stochastic rounding (SR) is a key technique for low-precision training and inference: instead of
always rounding to the nearest grid point, you round up with probability proportional to how close
you are to the upper point. This gives an **unbiased** quantization error — E[q(x)] = x — which
prevents systematic drift when the same rounding error is applied thousands of times (as in SSM
state updates).

**Before this commit**: SR in quack required the `cvt.rs` PTX instruction, which is only available
on SM100a and SM103a (Blackwell B200 / B100). Running SR on an H100 (SM90) or A100 (SM80) would
either crash or silently fall back to RTN.

**After this commit**: Software emulations of SR for BF16, FP16, FP8 e4m3, and FP8 e5m2 are
provided that work on any SM80+ GPU and produce **bit-exact results** for a given seed.

---

## The Three Algorithms

### BF16 Stochastic Rounding (`cvt_f32x2_bf16x2_rs_sw`)

BF16 has 7 mantissa bits; FP32 has 23. When converting FP32 → BF16, 16 trailing mantissa bits
are discarded. SR adds a random value in [0, 2^16) to those trailing bits *before* truncation,
so the carry-out into the kept bits determines the round direction probabilistically.

```
bits = f32_as_uint32(x)
trailing = bits & 0x0000FFFF          # the 16 bits that get dropped
carry = (trailing + rand_u16) >> 16   # 1 if we round up, 0 if we round down
result = (bits >> 16) + carry         # truncate + apply carry
```

This is a predicated integer add — very cheap, and exact because BF16 is just the top 16 bits
of FP32 with no exponent rebias.

### FP16 Stochastic Rounding (`cvt_f32x2_f16x2_rs_sw`)

FP16 has a different exponent bias and explicit mantissa, making the integer-add trick less
clean (subnormals straddle a power-of-two boundary where the ULP changes size). The solution:
use an FMA-rz (fused multiply-add with round-to-zero) to add scaled random noise directly in
floating-point:

```
scale = 2^(exponent(x) - 13)     # 1 ULP of x in FP16 grid
noise = random_float * scale      # uniform on [0, 1 ULP)
y = fma_rz(random_float, scale, x)  # x + noise, truncated toward zero
result = truncate_to_fp16(y)
```

`fma_rz` is critical: it prevents rounding in the addition itself. The subnormal case is
handled correctly because `scale` is derived from the biased exponent of `x` and the FMA
propagates carries across the subnormal boundary.

### FP8 Stochastic Rounding (`cvt_f32x2_e4m3x2_rs_sw`, `cvt_f32x2_e5m2x2_rs_sw`)

Same FMA-rz pattern as FP16, but the target grid has much coarser spacing (3 mantissa bits
for e4m3, 2 for e5m2). The "intermediate grid" trick routes through FP16 to benefit from its
cleaner subnormal handling, then truncates to the FP8 format:

```
prescale x into FP16 range → add scaled noise via fma_rz → truncate to FP8
```

For e4m3fn specifically: max representable value is 448, smallest normal is 2^-6, smallest
subnormal is 2^-9. Saturating cast is used (clamp to ±448) because without saturation,
values ≥ 480 map to NaN in e4m3fn — a silent poison bug.

---

## Changes to `copy_utils.py`

`sr_cvt_copy()` is quack's public API for doing SR conversion inside a GEMM epilogue
(e.g., storing FP32 accumulator to FP16/BF16 output). Before this commit it only supported
BF16 output. After:

- Added import of `convert_f32_to_f16_sr` (new function)
- Added runtime check that dst dtype is BF16 *or* FP16
- Dispatches to the right SR path based on dst dtype

This is the call site that will be used by FlashInfer's ucache kernel when we integrate.

---

## Changes to `tests/test_linear.py`

- **Removed** `test_gemm_sr_requires_sm100()`: this test asserted that SR raises on non-SM100.
  No longer true with software emulation.
- **Removed SM100 hardware gate** from the main SR test: no longer needed to skip on H100/A100.
- **Added `torch.float16`** to the dtype parameterization: SR is now tested for both BF16 and
  FP16 on all supported architectures.

---

## Why This Matters for the SSM Precision Study

The SSM precision study needs to run SR experiments on this node's B200 GPUs. While B200 does
have hardware `cvt.rs`, the software path matters for two reasons:

1. **Portability**: future study nodes may be H100 or A100. Software SR ensures the same
   numerical semantics everywhere, independent of hardware SR availability.

2. **Bit-exact reproducibility**: the software path is seeded and deterministic. The hardware
   `cvt.rs` path uses per-warp PRNG state that varies with scheduling — harder to reproduce.

The `StateQuantizer.from_str("fp16_sr")` mode in `ssm_precision/state_quantizer.py` uses
the same FMA-rz-based algorithm (adapted for PyTorch) described above.

---

## Architecture Detection (`_use_hw_cvt`)

A new compile-time function queries CUTLASS arch info to select hardware vs. software path:

```python
def _use_hw_cvt() -> bool:
    return arch in (SM100a, SM103a)
```

The public `convert_f32_to_f16_sr()` dispatches accordingly. Software path is the default
everywhere else.
