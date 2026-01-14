# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
import types
from dataclasses import dataclass
from functools import partial
from typing import Union

import torch
import torch.nn as nn
from fbgemm_gpu.experimental.gen_ai.quantize import int4_row_quantize_zp, pack_int4

from torchao.core.config import AOBaseConfig
from torchao.quantization import Int4Tensor, Int8Tensor
from torchao.quantization.granularity import PerRow
from torchao.quantization.quant_api import (
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig,
    _module_extra_repr,
)
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.quantization.utils import get_block_size

from .observer import GPTQObserverTensor

CONFIG_TO_TORCHAO_BASE_TENSOR = {
    Int4WeightOnlyConfig: Int4Tensor,
    Int8WeightOnlyConfig: Int8Tensor,
}


@dataclass
class GPTQConfig(AOBaseConfig):
    """Unified config for GPTQ quantization with explicit step control.

    step="observe": wraps weights as GPTQObserverTensor for observation.
    step="convert": applies GPTQ quantization to observed tensors.

    Args:
        step: Either "observe" or "convert"
        base_config: Base quantization configuration that determines the target dtype.
            Use Int4WeightOnlyConfig() for int4 or Int8WeightOnlyConfig() for int8.
        percdamp: Damping factor for Hessian
        gptq_quantize_block_size: Block size for GPTQ algorithm
    """

    step: str = "observe"  # "observe" or "convert"
    base_config: Union[Int4WeightOnlyConfig, Int8WeightOnlyConfig] = None
    percdamp: float = 0.01
    gptq_quantize_block_size: int = 128

    def __post_init__(self):
        if self.base_config is None:
            self.base_config = Int4WeightOnlyConfig(group_size=128)


@register_quantize_module_handler(GPTQConfig)
def _gptq_config_transform(
    module: torch.nn.Module, config: GPTQConfig, *, parameter_name="weight"
) -> torch.nn.Module:
    """Unified transform handler that uses explicit step control."""
    tensor = getattr(module, parameter_name)

    if config.step == "observe":
        # Observation phase: wrap as GPTQObserverTensor
        new_tensor = GPTQObserverTensor.from_hp(tensor)
        setattr(module, parameter_name, nn.Parameter(new_tensor, requires_grad=False))
        module.extra_repr = types.MethodType(
            partial(
                _module_extra_repr,
                original_extra_repr=module.extra_repr,
                parameter_name=parameter_name,
            ),
            module,
        )
        return module
    elif config.step == "convert":
        # Quantization phase: tensor should be an GPTQObserverTensor
        if not isinstance(tensor, GPTQObserverTensor):
            raise ValueError(
                f"Expected {parameter_name} to be GPTQObserverTensor in 'convert' step, "
                f"but got {type(tensor)}. Did you run the 'observe' step first?"
            )

        # Validate that observations were recorded
        if tensor.hessian is None:
            raise ValueError(
                f"No observations recorded for {parameter_name}. "
                f"Hessian is None. Did you run forward passes during the observe step?"
            )

        # Use pre-computed Hessian directly
        hessian = tensor.hessian
        new_tensor = gptq_quantize(hessian, tensor.hp_data, config)
        new_quantized_tensor = nn.Parameter(new_tensor, requires_grad=False)
        setattr(module, parameter_name, new_quantized_tensor)
        return module
    else:
        raise ValueError(
            f"Invalid step '{config.step}'. Must be 'observe' or 'convert'."
        )


def _int4_row_quantize_zp_precomputed_qparams(
    x: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Quantize tensor using precomputed scales and zero points."""
    n_bit = 4
    to_quant = torch.split(x.to(torch.float), group_size, dim=-1)

    scales_row = scales.t().contiguous()
    zeros_row = zeros.t().contiguous()
    scales_list = torch.split(scales_row, 1, dim=-1)
    zeros_list = torch.split(zeros_row, 1, dim=-1)

    min_val = [
        zero_chunk - scale_chunk * (2 ** (n_bit - 1))
        for zero_chunk, scale_chunk in zip(zeros_list, scales_list)
    ]
    max_int = 2**n_bit - 1
    min_int = 0

    out = [
        chunk.sub(min_chunk).div(scale_chunk).round().clamp_(min_int, max_int)
        for chunk, min_chunk, scale_chunk in zip(to_quant, min_val, scales_list)
    ]
    out = [(chunk - 2 ** (n_bit - 1)).to(dtype=torch.int8) for chunk in out]
    out = torch.cat(out, dim=-1)
    return out


def _int4_row_dequantize_zp(
    x: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize int4 row-quantized tensor with zero point."""
    n_bit = 4

    scales = scales.t().contiguous()
    zeros = zeros.t().contiguous()

    x_chunks = torch.split(x, group_size, dim=-1)
    scales_list = torch.split(scales, 1, dim=-1)
    zeros_list = torch.split(zeros, 1, dim=-1)

    dequant_chunks = []
    for chunk, scale_chunk, zero_chunk in zip(x_chunks, scales_list, zeros_list):
        chunk_float = chunk.to(torch.float32) + 2 ** (n_bit - 1)
        min_val = zero_chunk - scale_chunk * (2 ** (n_bit - 1))
        dequant = chunk_float * scale_chunk + min_val
        dequant_chunks.append(dequant)

    return torch.cat(dequant_chunks, dim=-1)


def gptq_quantize(H: torch.Tensor, W: torch.Tensor, config: GPTQConfig) -> Int4Tensor:
    """
    This function implements the GPTQ algorithm described in this paper: https://arxiv.org/abs/2210.17323
    It is currently hardcoded to only support int4 per-group quantization.

    Example:
        2x2 GPTQ walkthrough (quantizing to 2-bit integers)

        Initial weight matrix:
            W = [[1.2, 0.8],
                 [0.5, 1.5]]

            X = [[1.0],
                 [1.0]]

        Step 1: Compute Hessian
            H = 2/n * (X^T @ X)
            H = [[2.0, 2.0],
                 [2.0, 2.0]]

        Step 2: Apply damping and compute inverse
            damp = 0.01 * mean(diag(H)) = 0.02

            H_damped = [[2.02, 2.00],
                        [2.00, 2.02]]

            H_inv = [[0.505, -0.495],
                     [-0.495, 0.505]]

        Step 3: Quantize column 0, propagate errors to column 1

            Original W:
            W = [[1.2, 0.8],
                 [0.5, 1.5]]

            Quantize column 0: [1.2, 0.5] → [1, 0] (after scale/zero point)
            Dequantized: [1.166, 0.5]

            Compute errors:
            err = (original - dequantized) / H_inv[col, col]
            err = [[0.034 / 0.505],  = [[0.067],
                   [0.0 / 0.505]]       [0.0]]

            Update remaining columns:
            W[:, 1:] = W[:, 1:] - err @ H_inv[0, 1:]

            W[:, 1:] = [[0.8],   - [[0.067],  @ [[-0.495]]
                        [1.5]]      [0.0]]

            W[:, 1:] = [[0.8 - (0.067 * -0.495)],  = [[0.833],
                        [1.5 - (0.0 * -0.495)]]      [1.5]]

            After Step 3:
            W = [[1.166, 0.833],    (column 0 dequantized, column 1 updated)
                 [0.5, 1.5]]

        Step 4: Quantize column 1 (now uses updated values)

            Quantize column 1: [0.833, 1.5] → [1, 2]
            Dequantized: [0.833, 1.499]

            Final quantized weight (after dequantization):
            W_gptq = [[1.166, 0.833],
                      [0.5, 1.499]]

        Result comparison:
            Original output: W @ X = [[2.0], [2.0]]

            Naive quantization: [[1, 1], [0, 2]] → output [[2], [2]] (error = 0)
            GPTQ quantization:  [[1, 1], [0, 2]] → output [[1.999], [1.999]] (error = 0.001)

        Key insight: By updating column 1 based on column 0's quantization error,
        GPTQ ensures the overall output stays closer to the original.

    Args:
        H: Hessian matrix approximation
        W: Weight matrix to quantize
        config: GPTQ configuration

    Returns:
        Int4Tensor: Quantized weight matrix
    """
    gptq_quantize_block_size = config.gptq_quantize_block_size
    percdamp = config.percdamp
    base_config = config.base_config
    base_tensor = CONFIG_TO_TORCHAO_BASE_TENSOR[base_config.__class__]

    if isinstance(base_config, Int4WeightOnlyConfig):
        group_size = config.base_config.group_size
        block_size = [1, group_size]
    elif isinstance(base_config, Int8WeightOnlyConfig):
        assert isinstance(base_config.granularity, PerRow), (
            "GPTQ only supports per-row quantization"
        )
        block_size = get_block_size(W.shape, base_config.granularity)
        block_size = list(block_size)
        group_size = block_size[-1]

    assert W.dim() == 2
    assert group_size > 0

    W = W.view(-1, W.shape[-1]).detach()
    columns = W.shape[1]
    device = W.device

    gptq_quantize_block_size = (
        math.ceil(gptq_quantize_block_size / group_size) * group_size
    )

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    all_qparams = []

    # The general ideal of GPTQ is simple.
    # If we have some linear layer W, we want to find a quantized version of W that minimizes some error metric, like MSE
    # argmin W' ||WX − W'X ||

    # We do this iteratively, row by row, which also allows us to update the remaining rows to better account for the quantization error
    # This is the general idea of GPTQ, which seeks to do this update much faster on GPUs.

    # We iterate through the weight matrix in blocks
    for W_quantize_block, block_start in zip(
        torch.split(W, gptq_quantize_block_size, dim=1),
        range(0, columns, gptq_quantize_block_size),
    ):
        block_end = min(block_start + gptq_quantize_block_size, columns)

        Err1 = torch.zeros_like(W_quantize_block, dtype=H.dtype)
        Hinv_quantize_block = Hinv[block_start:block_end, block_start:block_end]

        # If we are doing per-row quantization, the group_size is equal to the number of columns
        # Otherwise, if we do per-group quantization, we need to iterate through the block one group at a time.
        for W_group, group_start in zip(
            torch.split(W_quantize_block, group_size, dim=1),
            range(block_start, block_end, group_size),
        ):
            group_end = min(group_start + group_size, columns)

            # calculate qparams for the group only once
            if group_start % group_size == 0:
                if isinstance(base_config, Int4WeightOnlyConfig):
                    _, scale, zero = int4_row_quantize_zp(W_group, group_size)
                    all_qparams.append((scale, zero))
                elif isinstance(base_config, Int8WeightOnlyConfig):
                    quantized_tensor = base_tensor.from_hp(
                        W_group, base_config.granularity
                    )
                    scale = quantized_tensor.scale
                    all_qparams.append(scale)

            # now we go row by row, updating the remaining rows to better account for the quantization error
            for i in range(group_start - block_start, group_end - block_start):
                w = W_quantize_block[:, i].unsqueeze(1)

                if isinstance(base_config, Int4WeightOnlyConfig):
                    q = _int4_row_quantize_zp_precomputed_qparams(
                        w, scale, zero, group_size
                    )
                    # Dequantize for error calculation
                    dq = _int4_row_dequantize_zp(q, scale, zero, group_size)
                elif isinstance(base_config, Int8WeightOnlyConfig):
                    q = Int8Tensor.from_hp(
                        w, granularity=base_config.granularity, scale=scale
                    )
                    dq = q.dequantize(output_dtype=torch.float)

                err1 = (w - dq) / Hinv_quantize_block[i, i]
                W_quantize_block[:, i:] -= err1.matmul(
                    Hinv_quantize_block[i, i:].unsqueeze(0)
                )
                Err1[:, i] = err1.flatten()

        # Update the rest of the remaining rows outside of the block
        W[:, block_end:] -= Err1.matmul(Hinv[block_start:block_end, block_end:])

    if "cuda" in device.type:
        torch.cuda.synchronize()

    if isinstance(base_config, Int4WeightOnlyConfig):
        # Create final Int4Tensor using standard from_hp method
        final_qparams = [torch.cat(x, dim=0) for x in zip(*all_qparams)]

        # Quantize using precomputed qparams
        wq = _int4_row_quantize_zp_precomputed_qparams(
            W, final_qparams[0], final_qparams[1], group_size
        )
        wq_packed = pack_int4(wq)

        res = Int4Tensor(
            qdata=wq_packed,
            scale=final_qparams[0].to(W.dtype),
            zero_point=final_qparams[1].to(W.dtype),
            block_size=block_size,
            shape=W.shape,
            act_pre_scale=None,
        )

    elif isinstance(base_config, Int8WeightOnlyConfig):
        final_scale = all_qparams[0]
        res = Int8Tensor.from_hp(
            W, granularity=base_config.granularity, scale=final_scale
        )

    return res


def _calculate_hessian(inputs, device=None):
    """Calculate Hessian matrix from input activations for GPTQ.

    DEPRECATED: This function is kept for backward compatibility in tests only.
    GPTQObserverTensor now computes Hessian incrementally during observation.
    Use GPTQObserverTensor.hessian instead for production code.
    """
    H = 0
    total_batches = 0

    for inp in inputs:
        # Setup x (activation tensor)
        x = inp.float()
        if device:
            x = x.to(device)
        shape = x.shape
        n = 1 if len(shape) == 2 else shape[0]
        x = x.reshape(-1, shape[-1])

        # Update Hessian with running average
        H *= total_batches / (total_batches + n)
        total_batches += n

        x = ((2 / total_batches) ** (1 / 2)) * x.t()
        H += x.matmul(x.t())

    return H


__all__ = [
    "GPTQConfig",
    "gptq_quantize",
]
