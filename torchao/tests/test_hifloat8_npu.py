import unittest
from unittest.mock import patch

import torch

from torchao.float8.hifloat8_utils import (
    hifloat8_max_abs,
    hifloat8_min_max,
    hifloat8_scales_to_npu,
    is_hifloat8_tensor,
)


def _npu_available() -> bool:
    try:
        import torch_npu  # type: ignore

        return torch_npu.npu.is_available()
    except Exception:
        return False


class TestHiFloat8Utils(unittest.TestCase):
    def test_hifloat8_bounds(self):
        mn, mx = hifloat8_min_max()
        self.assertAlmostEqual(mn, -32768.0)
        self.assertAlmostEqual(mx, 32769.0)
        self.assertAlmostEqual(hifloat8_max_abs(), 32769.0)

    def test_hifloat8_scale_mapping_shapes(self):
        a_scale = torch.ones((4, 1), dtype=torch.float32)
        b_scale = torch.ones((1, 6), dtype=torch.float32)
        scale, pertoken_scale = hifloat8_scales_to_npu(a_scale, b_scale, out_features=6)
        self.assertEqual(scale.shape, (6,))
        self.assertEqual(pertoken_scale.shape, (4,))

        a_scale = torch.tensor(2.0, dtype=torch.float32)
        b_scale = torch.tensor(4.0, dtype=torch.float32)
        scale, pertoken_scale = hifloat8_scales_to_npu(a_scale, b_scale, out_features=1)
        self.assertEqual(scale.shape, (1,))
        self.assertEqual(pertoken_scale.shape, (1,))


@unittest.skipUnless(_npu_available(), "torch_npu/npu not available")
class TestHiFloat8NPUIntegration(unittest.TestCase):
    def test_float8tensor_from_hp_hifloat8(self):
        import torch_npu  # type: ignore
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            Float8Tensor,
        )
        from torchao.quantization.granularity import PerRow

        x = torch.randn(4, 8, device="npu", dtype=torch.bfloat16)
        qt = Float8Tensor.from_hp(
            x,
            float8_dtype=torch_npu.hifloat8,
            granularity=PerRow(),
        )
        self.assertTrue(is_hifloat8_tensor(qt.qdata))
        dq = qt.dequantize()
        self.assertEqual(dq.shape, x.shape)

    def test_float8trainingtensor_mm_uses_npu_quant_matmul(self):
        import torch_npu  # type: ignore
        from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
        from torchao.float8.float8_training_tensor import (
            GemmInputRole,
            LinearMMConfig,
        )
        from torchao.float8.config import ScalingGranularity

        x = torch.randn(4, 8, device="npu", dtype=torch.float16)
        w = torch.randn(8, 16, device="npu", dtype=torch.float16)
        mm_cfg = LinearMMConfig()

        a = hp_tensor_to_float8_dynamic(
            x,
            torch_npu.hifloat8,
            mm_cfg,
            gemm_input_role=GemmInputRole.INPUT,
            scaling_granularity=ScalingGranularity.TENSORWISE,
        )
        b = hp_tensor_to_float8_dynamic(
            w,
            torch_npu.hifloat8,
            mm_cfg,
            gemm_input_role=GemmInputRole.WEIGHT,
            scaling_granularity=ScalingGranularity.TENSORWISE,
        )
        self.assertTrue(is_hifloat8_tensor(a._data))
        self.assertTrue(is_hifloat8_tensor(b._data))

        def _fake_npu_quant_matmul(x1, x2, scale, **kwargs):
            out_dtype = kwargs.get("output_dtype", torch.float16)
            return torch.empty((x1.shape[0], x2.shape[1]), device=x1.device, dtype=out_dtype)

        with patch("torch_npu.npu_quant_matmul", side_effect=_fake_npu_quant_matmul) as m:
            out = torch.mm(a, b)
            self.assertTrue(m.called)
            self.assertEqual(out.shape, (4, 16))
            self.assertEqual(out.dtype, x.dtype)


if __name__ == "__main__":
    unittest.main()
