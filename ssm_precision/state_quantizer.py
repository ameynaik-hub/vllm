# SPDX-License-Identifier: Apache-2.0
"""Emulated SSM state quantization for precision studies.

Adapted from Johnny's fp8_sr_replayssm.py (manually verified).
All modes operate in fp32: fp32-in → fp32-out, round-tripped through the
target format. Nothing is stored in low precision; only the numerics move.

Usage:
    q = StateQuantizer.from_str("fp16_rtn")   # or fp16_sr, fp8_sr, fp32
    state = q(state)   # applied after kernel writes the state cache

Correctness invariant (rule a): always compute the output y from the
*new* state S_new before quantizing. Quantizing then reading would inject
an extra read-path error (~1e-3 relative at fp16) on every step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _Format:
    name: str
    dtype: torch.dtype
    mantissa_bits: int
    min_exp: int
    max_value: float
    saturate: bool

    @property
    def subnormal_ulp(self) -> float:
        return 2.0 ** (self.min_exp - self.mantissa_bits)


_FP8_E4M3 = _Format("fp8e4m3", torch.float8_e4m3fn, 3, -6, 448.0, saturate=True)
_FP16 = _Format("fp16", torch.float16, 10, -14, 65504.0, saturate=False)


class StateQuantizer:
    """Emulated quantizer: fp32 state -> fp32, round-tripped through target format.

    Create via StateQuantizer.from_str(env_string) where env_string is one of:
        fp32       → identity (no-op; from_str returns None)
        fp16_rtn   → FP16 round-to-nearest (plain .half().float())
        fp16_sr    → FP16 stochastic rounding
        fp8_sr     → FP8 e4m3 block-128 scaled + SR
    """

    def __init__(
        self,
        precision: str,
        rounding: str,
        *,
        seed: int = 1234,
        layer: int = 0,
        block: int = 128,
        axis: int = -1,
    ):
        self.precision = precision
        self.rounding = rounding
        self.block = block
        self.axis = axis
        self._gen: torch.Generator | None = None
        self._seed = seed * 1000 + layer
        # Pre-allocated noise buffer — reused in-place to avoid per-step allocation.
        # Resized only when the state shape grows (rare); otherwise uniform_() refills it.
        self._noise: torch.Tensor | None = None

    @classmethod
    def from_str(cls, s: str, **kw) -> "StateQuantizer | None":
        """Parse env-var string. Returns None for fp32 (skip hook entirely)."""
        if s == "fp32":
            return None
        if s == "fp16_rtn":
            return cls("fp16", "rtn", **kw)
        if s == "fp16_sr":
            return cls("fp16", "sr", **kw)
        if s == "fp8_rtn":
            return cls("fp8", "rtn", **kw)
        if s == "fp8_sr":
            return cls("fp8", "sr", **kw)
        raise ValueError(f"Unknown SSM_PRECISION_DTYPE: {s!r}. "
                         f"Expected one of: fp32, fp16_rtn, fp16_sr, fp8_sr")

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        """Apply emulated quantization. Input and output are fp32."""
        if self.precision == "fp16":
            return self._round(state, _FP16)
        if self.precision == "fp8":
            return self._fp8_blockwise(state)
        raise ValueError(self.precision)

    # -- rounding ------------------------------------------------------------------

    def _round(self, x: torch.Tensor, fmt: _Format) -> torch.Tensor:
        return self._sr(x, fmt) if self.rounding == "sr" else self._rtn(x, fmt)

    def _rtn(self, x: torch.Tensor, fmt: _Format) -> torch.Tensor:
        """Round-to-nearest (plain cast). Equivalent to what Triton does with
        `tl.store(ptrs, val.to(ptrs.dtype.element_ty))`."""
        return self._cast(x, fmt)

    def _sr(self, x: torch.Tensor, fmt: _Format) -> torch.Tensor:
        """Stochastic rounding matching PTX cvt.rs semantics.

        q = sign(x) * floor(|x| / ulp(x) + u) * ulp(x),   u ~ U[0,1)

        Unbiased, E[q] = x, *including across binade boundaries*.
        See quack commit 709b829 for the GPU-side implementation of the same algorithm.
        """
        ulp = self._ulp(x, fmt)
        q = torch.sign(x) * torch.floor(x.abs() / ulp + self._uniform(x)) * ulp
        return self._cast(q, fmt)

    def _ulp(self, x: torch.Tensor, fmt: _Format) -> torch.Tensor:
        a = x.abs()
        e = torch.floor(
            torch.log2(a.clamp_min(2.0 ** fmt.min_exp))
        ).clamp_min(fmt.min_exp)
        return torch.where(
            a < 2.0 ** fmt.min_exp,
            torch.full_like(a, fmt.subnormal_ulp),
            torch.exp2(e - fmt.mantissa_bits),
        )

    @staticmethod
    def _cast(x: torch.Tensor, fmt: _Format) -> torch.Tensor:
        if fmt.saturate:
            x = x.clamp(-fmt.max_value, fmt.max_value)
        return x.to(fmt.dtype).to(torch.float32)

    def _uniform(self, x: torch.Tensor) -> torch.Tensor:
        if self._gen is None:
            self._gen = torch.Generator(device=x.device)
            self._gen.manual_seed(self._seed)
        # Reuse pre-allocated buffer; only reallocate when shape grows.
        # uniform_() fills in-place — avoids a new tensor allocation every decode step.
        if self._noise is None or self._noise.numel() < x.numel():
            self._noise = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        noise = self._noise.view(-1)[: x.numel()].view(x.shape)
        noise.uniform_(generator=self._gen)
        return noise

    # -- fp8 block scaling ---------------------------------------------------------

    def _fp8_blockwise(self, x: torch.Tensor) -> torch.Tensor:
        """Block-scaled e4m3 round-trip: scale = amax_block / 448.

        block contiguous elements along axis share one scale. For the GDN state
        [slots, HV, V, K], axis=-1 (K-axis) with block=128 puts one scale per
        (HV, V-row) across K — the axis the readout contracts over.
        """
        xm = x.movedim(self.axis, -1)
        lead, n = xm.shape[:-1], xm.shape[-1]
        if n % self.block != 0:
            raise ValueError(
                f"FP8 block size {self.block} does not divide axis of size {n}")
        xr = xm.reshape(*lead, n // self.block, self.block)
        scale = xr.abs().amax(dim=-1, keepdim=True) / _FP8_E4M3.max_value
        # Guard: all-zero block or amax below FP32 subnormal threshold gives 0.
        # Dividing by 0 → NaN for zero elements. Unit scale is correct fallback:
        # such values quantize to zero anyway.
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = self._round(xr / scale, _FP8_E4M3)
        return (q * scale).reshape(*lead, n).movedim(-1, self.axis).contiguous()
