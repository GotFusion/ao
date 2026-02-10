from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional, Tuple

import torch


@lru_cache(None)
def _load_torch_npu() -> Tuple[Optional[Any], Optional[type]]:
    """
    Best-effort import of torch_npu and its HiFloat8 tensor type.

    We intentionally catch broad exceptions because importing torch_npu can
    raise RuntimeError in environments without Ascend/NPU support or when
    another accelerator is active.
    """
    try:
        import torch_npu  # type: ignore
        from torch_npu.utils.hif8_tensor import _HiFloat8Tensor  # type: ignore
    except Exception:
        return None, None
    return torch_npu, _HiFloat8Tensor


def is_hifloat8_dtype(dtype: Any) -> bool:
    torch_npu, _ = _load_torch_npu()
    if torch_npu is None:
        return False
    try:
        return dtype == torch_npu.hifloat8
    except Exception:
        return False


def is_hifloat8_tensor(tensor: Any) -> bool:
    _, hif8_cls = _load_torch_npu()
    return hif8_cls is not None and isinstance(tensor, hif8_cls)


def to_hifloat8(tensor: torch.Tensor) -> torch.Tensor:
    torch_npu, hif8_cls = _load_torch_npu()
    if torch_npu is None or hif8_cls is None:
        raise RuntimeError(
            "HiFloat8 support requires torch_npu with Ascend/NPU runtime available."
        )
    return hif8_cls.to_hifloat8(tensor)


def _decode_hifloat8_byte(v: int) -> float:
    # This conversion follows the reference implementation used in
    # torch_npu tests for HiFloat8.
    if v == 0 or v == 128:
        return 0.0
    if v == 239:
        return -32768.0
    if v == 111:
        return 32768.0
    if v >= 128:
        sign = -1.0
    else:
        sign = 1.0
    dot_4_bits = v & 120
    dot_4_value = dot_4_bits >> 3
    if dot_4_value >= 12:
        exponent = v & 30
        exponent_int = exponent >> 1
        if exponent_int >= 8:
            exponent_value = -exponent_int
        else:
            exponent_value = exponent_int + 8
        fra_int = v & 1
        m_value = 1.0 + fra_int * 0.5
    elif dot_4_value >= 8:
        exponent = v & 28
        exponent_int = exponent >> 2
        if exponent_int >= 4:
            exponent_value = -exponent_int
        else:
            exponent_value = exponent_int + 4
        fra_int = v & 3
        m_value = 1.0 + fra_int * 0.25
    elif dot_4_value >= 4:
        exponent = v & 24
        exponent_int = exponent >> 3
        if exponent_int >= 2:
            exponent_value = -exponent_int
        else:
            exponent_value = exponent_int + 2
        fra_int = v & 7
        m_value = 1.0 + fra_int * 0.125
    elif dot_4_value >= 2:
        exponent = v & 8
        exponent_sign = exponent >> 3
        if exponent_sign >= 1:
            exponent_value = -1
        else:
            exponent_value = 1
        fra_int = v & 7
        m_value = 1.0 + fra_int * 0.125
    elif dot_4_value == 1:
        exponent_value = 0
        fra_int = v & 7
        m_value = 1.0 + fra_int * 0.125
    elif dot_4_value == 0:
        m_value = 1.0
        exponent_value = (v & 7) - 23
    else:
        return 0.0
    return sign + pow(2.0, exponent_value) * m_value


@lru_cache(None)
def hifloat8_max_abs() -> float:
    return max(abs(_decode_hifloat8_byte(v)) for v in range(256))


@lru_cache(None)
def hifloat8_min_max() -> Tuple[float, float]:
    vals = [_decode_hifloat8_byte(v) for v in range(256)]
    return min(vals), max(vals)


def hifloat8_raw_int8(tensor: torch.Tensor) -> torch.Tensor:
    """
    Return raw HiFloat8 payload as int8 tensor suitable for npu_quant_matmul.
    """
    if is_hifloat8_tensor(tensor):
        raw = tensor._data  # type: ignore[attr-defined]
    else:
        raw = tensor
    return raw.contiguous().to(torch.int8)


def hifloat8_scales_to_npu(
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_features: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Map Float8TrainingTensor scales to npu_quant_matmul scale/pertoken_scale.
    """
    pertoken_scale = None
    if a_scale is not None:
        pertoken_scale = a_scale.reciprocal()
        if pertoken_scale.numel() == 1:
            pertoken_scale = pertoken_scale.reshape(1)
        elif pertoken_scale.dim() == 2 and pertoken_scale.shape[-1] == 1:
            pertoken_scale = pertoken_scale.reshape(-1)

    if b_scale is not None:
        scale = b_scale.reciprocal()
        if scale.numel() == 1:
            scale = scale.reshape(1)
        elif scale.dim() == 2 and scale.shape[0] == 1:
            scale = scale.reshape(-1)
    else:
        scale = torch.ones((out_features,), device=a_scale.device, dtype=torch.float32)

    return scale, pertoken_scale
