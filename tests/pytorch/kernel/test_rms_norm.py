import pytest
import torch

from lmdeploy.utils import is_bf16_supported


def _bf16_mark():
    return pytest.mark.skipif(not is_bf16_supported(), reason='bf16 not supported.')


class TestRMSNorm:

    @pytest.fixture(autouse=True, scope='class')
    def initialize(self):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        yield

    @pytest.fixture(scope='class')
    def dtype(self, request):
        yield request.param

    @pytest.fixture(scope='class')
    def input_shape(self, request):
        yield request.param

    @pytest.fixture(scope='class')
    def hidden_size(self, input_shape):
        yield input_shape[-1]

    @pytest.fixture(scope='class')
    def input(self, dtype, input_shape):
        yield torch.randn(input_shape, dtype=dtype, device='cuda')

    @pytest.fixture(scope='class')
    def weight(self, dtype, hidden_size):
        yield torch.randn(hidden_size, dtype=dtype, device='cuda')

    @pytest.fixture(scope='class')
    def eps(self):
        yield 1e-6

    @pytest.fixture(scope='class')
    def gt(self, input, weight, eps):
        input_dtype = input.dtype
        input = input.to(torch.float32)
        variance = (input * input).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + eps)
        return weight * input.to(input_dtype)

    @pytest.mark.parametrize('input_shape', [(2, 4, 4096), (4, 4096), (4096, )], indirect=True)
    @pytest.mark.parametrize('dtype', [pytest.param(torch.bfloat16, marks=_bf16_mark()), torch.float16], indirect=True)
    def test_rms_norm(self, input, weight, eps, gt):
        from lmdeploy.pytorch.kernels.cuda import rms_norm

        out = rms_norm(input, weight, eps)
        torch.testing.assert_close(out, gt)

    @pytest.fixture(scope='class')
    def residual(self, dtype, input_shape):
        yield torch.randn(input_shape, dtype=dtype, device='cuda')

    @pytest.fixture(scope='class')
    def gt_residual(self, input, residual, weight, eps):

        input = input + residual
        out_res = input
        input_dtype = input.dtype
        input = input.to(torch.float32)
        variance = (input * input).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + eps)
        return weight * input.to(input_dtype), out_res

    @pytest.mark.parametrize('input_shape', [(2, 4, 4096), (4, 4096), (4096, )], indirect=True)
    @pytest.mark.parametrize('dtype', [pytest.param(torch.bfloat16, marks=_bf16_mark()), torch.float16], indirect=True)
    def test_rms_norm_residual(self, input, residual, weight, eps, gt_residual):
        from lmdeploy.pytorch.kernels.cuda import rms_norm

        out, out_res = rms_norm(input, weight, eps, residual=residual)
        gt, gt_res = gt_residual
        torch.testing.assert_close(out, gt)
        torch.testing.assert_close(out_res, gt_res)

    @pytest.mark.parametrize('input_shape', [
        (7168, ),
        (1, 7168),
        (33, 7168),
        (2, 3, 7168),
    ])
    def test_fp32_input_bf16_residual_matches_explicit_boundary_cast(
            self, input_shape):
        from lmdeploy.pytorch.kernels.cuda import rms_norm

        generator = torch.Generator(device='cuda').manual_seed(20260805)
        reduced_fp32 = torch.randn(
            input_shape,
            dtype=torch.float32,
            device='cuda',
            generator=generator,
        )
        residual = torch.randn(
            input_shape,
            dtype=torch.bfloat16,
            device='cuda',
            generator=generator,
        )
        weight = torch.randn(
            input_shape[-1],
            dtype=torch.bfloat16,
            device='cuda',
            generator=generator,
        )
        midpoint = torch.tensor(
            [1.00390625, -1.00390625, 0.501953125, -0.501953125],
            dtype=torch.float32,
            device='cuda',
        )
        toward_zero = torch.nextafter(midpoint, torch.zeros_like(midpoint))
        away_from_zero = torch.nextafter(
            midpoint,
            torch.where(midpoint > 0,
                        torch.full_like(midpoint, torch.inf),
                        torch.full_like(midpoint, -torch.inf)),
        )
        boundary_values = torch.cat(
            [midpoint, toward_zero, away_from_zero])
        reduced_fp32.reshape(-1)[:boundary_values.numel()].copy_(
            boundary_values)

        reference, reference_residual = rms_norm(
            reduced_fp32.to(torch.bfloat16),
            weight,
            1e-6,
            residual=residual,
        )
        result, result_residual = rms_norm(
            reduced_fp32,
            weight,
            1e-6,
            residual=residual,
            cast_input_to_bf16=True,
        )

        assert result.dtype == torch.bfloat16
        assert result_residual.dtype == torch.bfloat16
        assert torch.equal(result, reference)
        assert torch.equal(result_residual, reference_residual)

    def test_mixed_dtype_requires_explicit_boundary_contract(self):
        from lmdeploy.pytorch.kernels.cuda import rms_norm

        reduced_fp32 = torch.randn(
            (2, 7168), dtype=torch.float32, device='cuda')
        residual = torch.randn(
            (2, 7168), dtype=torch.bfloat16, device='cuda')
        weight = torch.randn(7168, dtype=torch.bfloat16, device='cuda')

        result, result_residual = rms_norm(
            reduced_fp32, weight, residual=residual)

        assert result.dtype == torch.float32
        assert result_residual.dtype == torch.bfloat16

    def test_fp32_input_bf16_residual_cuda_graph_replay_is_exact(self):
        from lmdeploy.pytorch.kernels.cuda import rms_norm

        generator = torch.Generator(device='cuda').manual_seed(20260806)
        shape = (32, 7168)
        reduced_fp32 = torch.randn(
            shape,
            dtype=torch.float32,
            device='cuda',
            generator=generator,
        )
        residual = torch.randn(
            shape,
            dtype=torch.bfloat16,
            device='cuda',
            generator=generator,
        )
        weight = torch.randn(
            shape[-1],
            dtype=torch.bfloat16,
            device='cuda',
            generator=generator,
        )
        graph_out = torch.empty_like(reduced_fp32, dtype=torch.bfloat16)
        graph_residual = torch.empty_like(residual)

        rms_norm(
            reduced_fp32,
            weight,
            residual=residual,
            out=graph_out,
            out_residual=graph_residual,
            cast_input_to_bf16=True,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result, result_residual = rms_norm(
                reduced_fp32,
                weight,
                residual=residual,
                out=graph_out,
                out_residual=graph_residual,
                cast_input_to_bf16=True,
            )
        output_ptr = result.data_ptr()
        residual_ptr = result_residual.data_ptr()

        for _ in range(2):
            reduced_fp32.normal_(generator=generator)
            residual.normal_(generator=generator)
            reference, reference_residual = rms_norm(
                reduced_fp32.to(torch.bfloat16),
                weight,
                residual=residual,
            )
            graph.replay()
            torch.cuda.synchronize()

            assert result.dtype == torch.bfloat16
            assert result_residual.dtype == torch.bfloat16
            assert result.data_ptr() == output_ptr
            assert result_residual.data_ptr() == residual_ptr
            assert torch.equal(result, reference)
            assert torch.equal(result_residual, reference_residual)
