# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = "xpu"
pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="XPU is required")
FP4_VALUES = torch.tensor([
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
])


def _make_nvfp4(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, shape, dtype=torch.int64,
                          generator=generator)
    packed = (codes[..., 0::2] |
              (codes[..., 1::2] << 4)).to(torch.uint8).to(DEVICE)
    scale_values = torch.rand((*shape[:-1], shape[-1] // 16),
                              generator=generator) + 0.5
    scales = scale_values.to(DEVICE).to(torch.float8_e4m3fn)
    decoded = FP4_VALUES[codes].to(DEVICE)
    decoded = decoded * scales.float().repeat_interleave(16, dim=-1)
    return packed.contiguous(), scales.contiguous(), decoded


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16,
                                    torch.float32])
@torch.inference_mode()
def test_nvfp4_gemm_matches_dequantized_reference(dtype):
    torch.manual_seed(0)
    m, n, k = 3, 40, 64
    global_scale = 0.125
    x = (torch.randn(m, k, device=DEVICE) * 0.1).to(dtype)
    weight, scales, decoded = _make_nvfp4((n, k), seed=1)

    output = torch.ops._xpu_C.nvfp4_gemm(
        x, weight, scales, global_scale)
    reference = x.float() @ (decoded * global_scale).T

    tolerance = 1e-4 if dtype == torch.float32 else 4e-2
    torch.testing.assert_close(output.float(), reference, atol=tolerance,
                               rtol=2e-2)


@pytest.mark.parametrize("m", [0, 1, 8, 9])
@torch.inference_mode()
def test_nvfp4_gemm_batch_chunk_boundaries(m):
    torch.manual_seed(1)
    n, k = 40, 64
    global_scale = 0.125
    x = (torch.randn(m, k, device=DEVICE) * 0.1).to(torch.bfloat16)
    weight, scales, decoded = _make_nvfp4((n, k), seed=11)

    output = torch.ops._xpu_C.nvfp4_gemm(
        x, weight, scales, global_scale)
    reference = x.float() @ (decoded * global_scale).T

    assert output.shape == (m, n)
    torch.testing.assert_close(output.float(), reference, atol=4e-2,
                               rtol=2e-2)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn,
                                       torch.float8_e5m2])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("m", [1, 2, 4, 8])
@torch.inference_mode()
def test_fp8_gemm_matches_dequantized_reference(fp8_dtype, per_channel, m):
    torch.manual_seed(2)
    n, k = 40, 64
    x = (torch.randn(m, k, device=DEVICE) * 0.1).to(torch.bfloat16)
    weight = (torch.randn(n, k, device=DEVICE) * 0.2).to(fp8_dtype)
    scale_shape = (n, ) if per_channel else (1, )
    scale = torch.rand(*scale_shape, device=DEVICE,
                       dtype=torch.float32) + 0.5

    output = torch.ops._xpu_C.fp8_gemm_w8a16(
        x, weight.transpose(0, 1), scale, None)
    weight_scale = scale[:, None] if per_channel else scale
    reference = x.float() @ (weight.float() * weight_scale).T

    torch.testing.assert_close(output.float(), reference, atol=8e-2,
                               rtol=4e-2)


@pytest.mark.parametrize("multiply_router_weight", [False, True])
@torch.inference_mode()
def test_nvfp4_moe_matches_reference(multiply_router_weight):
    torch.manual_seed(3)
    m, experts, topk, hidden_size, intermediate_size = 5, 3, 2, 32, 32
    hidden = (torch.randn(m, hidden_size, device=DEVICE) * 0.1).to(
        torch.bfloat16)
    topk_ids = torch.tensor([
        [0, experts],
        [1, -1],
        [2, 0],
        [0, 1],
        [2, 1],
    ], dtype=torch.int32, device=DEVICE)
    topk_weights = torch.tensor([
        [0.7, 0.3],
        [0.6, 0.4],
        [0.8, 0.2],
        [0.55, 0.45],
        [0.65, 0.35],
    ], dtype=torch.float32, device=DEVICE)

    w13, w13_scale, w13_decoded = _make_nvfp4(
        (experts, 2 * intermediate_size, hidden_size), seed=4)
    w2, w2_scale, w2_decoded = _make_nvfp4(
        (experts, hidden_size, intermediate_size), seed=5)
    w13_global = torch.tensor([0.025, 0.02, 0.03], device=DEVICE)
    w2_global = torch.tensor([0.025, 0.03, 0.02], device=DEVICE)

    reference = torch.zeros(m, hidden_size, dtype=torch.float32, device=DEVICE)
    for row in range(m):
        for route in range(topk):
            expert = int(topk_ids[row, route])
            if expert < 0 or expert >= experts:
                continue
            gate_up = hidden[row].float() @ (
                w13_decoded[expert] * w13_global[expert]).T
            activated = F.silu(gate_up[:intermediate_size]) * gate_up[
                intermediate_size:]
            expert_output = activated @ (
                w2_decoded[expert] * w2_global[expert]).T
            route_weight = topk_weights[row, route] \
                if multiply_router_weight else 1.0
            reference[row] += route_weight * expert_output

    args = (hidden, topk_ids, topk_weights, w13, w13_scale, w13_global,
            w2, w2_scale, w2_global, multiply_router_weight)
    output = torch.ops._xpu_C.nvfp4_moe(*args)

    torch.testing.assert_close(output, reference, atol=3e-3, rtol=2e-2)


@torch.inference_mode()
def test_nvfp4_moe_automatic_split_fallback_matches_reference():
    torch.manual_seed(5)
    hidden_size = 32
    local_mem_size = torch.xpu.get_device_properties().local_mem_size
    intermediate_size = ((local_mem_size // 12 + 1 + 31) // 32) * 32
    assert intermediate_size * 4 <= local_mem_size
    assert intermediate_size * 12 > local_mem_size

    hidden = (torch.randn(1, hidden_size, device=DEVICE) * 0.1).to(
        torch.bfloat16)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32, device=DEVICE)
    topk_weights = torch.tensor([[0.75]], device=DEVICE)
    w13, w13_scale, w13_decoded = _make_nvfp4(
        (1, 2 * intermediate_size, hidden_size), seed=12)
    w2, w2_scale, w2_decoded = _make_nvfp4(
        (1, hidden_size, intermediate_size), seed=13)
    w13_global = torch.tensor([0.01], device=DEVICE)
    w2_global = torch.tensor([0.01], device=DEVICE)

    gate_up = hidden.float() @ (w13_decoded[0] * w13_global[0]).T
    activated = F.silu(gate_up[:, :intermediate_size]) * gate_up[
        :, intermediate_size:]
    reference = activated @ (w2_decoded[0] * w2_global[0]).T
    reference *= topk_weights[0, 0]

    output = torch.ops._xpu_C.nvfp4_moe(
        hidden, topk_ids, topk_weights, w13, w13_scale, w13_global, w2,
        w2_scale, w2_global, True)

    torch.testing.assert_close(output, reference, atol=3e-3, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16,
                                    torch.float32])
@torch.inference_mode()
def test_rms_norm_and_fused_add_match_reference(dtype):
    torch.manual_seed(6)
    epsilon = 1e-6
    x = torch.randn(3, 128, device=DEVICE, dtype=dtype)
    weight = (torch.randn(128, device=DEVICE) * 0.1 + 1).to(dtype)

    output = torch.empty_like(x)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(dim=-1, keepdim=True) + epsilon)
    reference = normalized.to(x.dtype).mul(weight)
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)

    residual = torch.randn_like(x)
    residual_reference = (x.float() + residual.float()).to(x.dtype)
    summed = x.float() + residual.float()
    normalized = summed * torch.rsqrt(
        summed.square().mean(dim=-1, keepdim=True) + epsilon)
    reference = normalized.to(x.dtype).mul(weight)
    fused_input = x.clone()
    torch.ops._C.fused_add_rms_norm(fused_input, residual, weight, epsilon)

    torch.testing.assert_close(fused_input, reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(residual, residual_reference, atol=0, rtol=0)


def _reference_gdn_decode(projected_qkvz, projected_ba, conv_state,
                          ssm_state, conv_weight, conv_bias, a_log, dt_bias,
                          state_indices):
    batch = projected_qkvz.shape[0]
    num_k_heads, num_v_heads, head_dim = 16, 32, 128
    q_dim = num_k_heads * head_dim
    k_dim = num_k_heads * head_dim
    v_dim = num_v_heads * head_dim
    conv_dim = q_dim + k_dim + v_dim
    slots = state_indices.long()
    conv_state_sd = conv_state if conv_state.shape[1] == 3 else \
        conv_state.transpose(1, 2)

    x = projected_qkvz[:, :conv_dim]
    z = projected_qkvz[:, conv_dim:].reshape(batch, num_v_heads,
                                             head_dim).clone()
    history = conv_state_sd[slots].clone()
    x4 = torch.cat([history, x.unsqueeze(1)], dim=1).float()
    mixed = F.silu(
        (x4 * conv_weight.T.unsqueeze(0).float()).sum(dim=1) +
        conv_bias.float()).to(projected_qkvz.dtype)

    conv_state_sd[slots, 0] = history[:, 1]
    conv_state_sd[slots, 1] = history[:, 2]
    conv_state_sd[slots, 2] = x

    q = mixed[:, :q_dim].reshape(batch, num_k_heads, head_dim).float()
    k = mixed[:, q_dim:q_dim + k_dim].reshape(
        batch, num_k_heads, head_dim).float()
    v = mixed[:, q_dim + k_dim:].reshape(
        batch, num_v_heads, head_dim).float()
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    q = q * (head_dim**-0.5)

    beta = torch.sigmoid(projected_ba[:, :num_v_heads].float())
    decay = torch.exp(-torch.exp(a_log.float()) * F.softplus(
        projected_ba[:, num_v_heads:].float() + dt_bias.float()))
    state = ssm_state[slots].float()
    output = torch.empty(batch, num_v_heads, head_dim, device=DEVICE)
    heads_per_k = num_v_heads // num_k_heads
    for v_head in range(num_v_heads):
        k_head = v_head // heads_per_k
        current = state[:, v_head] * decay[:, v_head, None, None]
        prediction = (current * k[:, k_head, None]).sum(dim=-1)
        delta = (v[:, v_head] - prediction) * beta[:, v_head, None]
        current = current + delta[..., None] * k[:, k_head, None]
        output[:, v_head] = (current * q[:, k_head, None]).sum(dim=-1)
        state[:, v_head] = current
    ssm_state[slots] = state.to(ssm_state.dtype)
    return output.to(projected_qkvz.dtype), z


@pytest.mark.parametrize("conv_dim_first", [False, True])
@torch.inference_mode()
def test_qwen_gdn_decode_matches_reference(conv_dim_first):
    torch.manual_seed(7)
    batch, slots = 2, 4
    dtype = torch.bfloat16
    projected_qkvz = (torch.randn(batch, 12288, device=DEVICE) * 0.03).to(
        dtype)
    projected_ba = (torch.randn(batch, 64, device=DEVICE) * 0.03).to(dtype)
    conv_state = (torch.randn(slots, 3, 8192, device=DEVICE) * 0.03).to(dtype)
    if conv_dim_first:
        conv_state = conv_state.transpose(1, 2).contiguous()
    ssm_state = torch.randn(slots, 32, 128, 128, device=DEVICE) * 0.03
    conv_weight = (torch.randn(8192, 4, device=DEVICE) * 0.03).to(dtype)
    conv_bias = (torch.randn(8192, device=DEVICE) * 0.01).to(dtype)
    a_log = (torch.randn(32, device=DEVICE) * 0.02).float()
    dt_bias = (torch.randn(32, device=DEVICE) * 0.02).to(dtype)
    state_indices = torch.tensor([0, 3], dtype=torch.int32, device=DEVICE)

    conv_reference = conv_state.clone()
    ssm_reference = ssm_state.clone()
    output_reference, z_reference = _reference_gdn_decode(
        projected_qkvz, projected_ba, conv_reference, ssm_reference,
        conv_weight, conv_bias, a_log, dt_bias, state_indices)

    output, z = torch.ops._xpu_C.qwen_gdn_decode(
        projected_qkvz, projected_ba, conv_state, ssm_state, conv_weight,
        conv_bias, a_log, dt_bias, state_indices)

    torch.testing.assert_close(output, output_reference, atol=2e-5, rtol=1e-2)
    torch.testing.assert_close(z, z_reference, atol=0, rtol=0)
    torch.testing.assert_close(conv_state, conv_reference, atol=0, rtol=0)
    torch.testing.assert_close(ssm_state, ssm_reference, atol=2e-7, rtol=1e-2)


@torch.inference_mode()
def test_qwen_gdn_decode_rejects_mismatched_conv_state_dtype():
    batch, slots = 1, 1
    projected_qkvz = torch.empty(batch, 12288, device=DEVICE,
                                 dtype=torch.bfloat16)
    projected_ba = torch.empty(batch, 64, device=DEVICE,
                               dtype=torch.bfloat16)
    conv_state = torch.empty(slots, 3, 8192, device=DEVICE,
                             dtype=torch.float16)
    ssm_state = torch.empty(slots, 32, 128, 128, device=DEVICE)
    conv_weight = torch.empty(8192, 4, device=DEVICE, dtype=torch.bfloat16)
    conv_bias = torch.empty(8192, device=DEVICE, dtype=torch.bfloat16)
    a_log = torch.empty(32, device=DEVICE)
    dt_bias = torch.empty(32, device=DEVICE, dtype=torch.bfloat16)
    state_indices = torch.zeros(batch, dtype=torch.int32, device=DEVICE)

    with pytest.raises(RuntimeError, match="conv_state.*must match"):
        torch.ops._xpu_C.qwen_gdn_decode(
            projected_qkvz, projected_ba, conv_state, ssm_state,
            conv_weight, conv_bias, a_log, dt_bias, state_indices)


@torch.inference_mode()
def test_qwen_gdn_decode_handles_invalid_state_indices():
    torch.manual_seed(8)
    batch, slots = 2, 2
    dtype = torch.bfloat16
    projected_qkvz = torch.randn(batch, 12288, device=DEVICE, dtype=dtype)
    projected_ba = torch.randn(batch, 64, device=DEVICE, dtype=dtype)
    conv_state = torch.randn(slots, 3, 8192, device=DEVICE, dtype=dtype)
    ssm_state = torch.randn(slots, 32, 128, 128, device=DEVICE)
    conv_weight = torch.randn(8192, 4, device=DEVICE, dtype=dtype)
    conv_bias = torch.randn(8192, device=DEVICE, dtype=dtype)
    a_log = torch.randn(32, device=DEVICE)
    dt_bias = torch.randn(32, device=DEVICE, dtype=dtype)
    state_indices = torch.tensor([-1, slots], dtype=torch.int32,
                                 device=DEVICE)
    conv_reference = conv_state.clone()
    ssm_reference = ssm_state.clone()

    output, z = torch.ops._xpu_C.qwen_gdn_decode(
        projected_qkvz, projected_ba, conv_state, ssm_state, conv_weight,
        conv_bias, a_log, dt_bias, state_indices)

    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(
        z, projected_qkvz[:, 8192:].reshape(batch, 32, 128))
    torch.testing.assert_close(conv_state, conv_reference)
    torch.testing.assert_close(ssm_state, ssm_reference)


@torch.inference_mode()
def test_decode_kernels_handle_empty_batches():
    dtype = torch.bfloat16

    fp8_x = torch.empty((0, 64), device=DEVICE, dtype=dtype)
    fp8_weight = torch.empty(
        (40, 64), device=DEVICE, dtype=torch.float8_e4m3fn)
    fp8_scale = torch.ones(40, device=DEVICE)
    fp8_output = torch.ops._xpu_C.fp8_gemm_w8a16(
        fp8_x, fp8_weight.transpose(0, 1), fp8_scale, None)
    assert fp8_output.shape == (0, 40)

    empty_fp8_weight = torch.empty(
        (0, 64), device=DEVICE, dtype=torch.float8_e4m3fn)
    empty_fp8_scale = torch.empty(0, device=DEVICE)
    empty_fp8_output = torch.ops._xpu_C.fp8_gemm_w8a16(
        torch.empty((2, 64), device=DEVICE, dtype=dtype),
        empty_fp8_weight.transpose(0, 1), empty_fp8_scale, None)
    assert empty_fp8_output.shape == (2, 0)

    empty_nvfp4_weight = torch.empty(
        (0, 32), device=DEVICE, dtype=torch.uint8)
    empty_nvfp4_scale = torch.empty(
        (0, 4), device=DEVICE, dtype=torch.float8_e4m3fn)
    empty_nvfp4_output = torch.ops._xpu_C.nvfp4_gemm(
        torch.empty((2, 64), device=DEVICE, dtype=dtype),
        empty_nvfp4_weight, empty_nvfp4_scale, 1.0)
    assert empty_nvfp4_output.shape == (2, 0)

    hidden = torch.empty((0, 32), device=DEVICE, dtype=dtype)
    topk_ids = torch.empty((0, 1), device=DEVICE, dtype=torch.int32)
    topk_weights = torch.empty((0, 1), device=DEVICE)
    w13, w13_scale, _ = _make_nvfp4((1, 64, 32), seed=14)
    w2, w2_scale, _ = _make_nvfp4((1, 32, 32), seed=15)
    global_scale = torch.ones(1, device=DEVICE)
    moe_output = torch.ops._xpu_C.nvfp4_moe(
        hidden, topk_ids, topk_weights, w13, w13_scale, global_scale, w2,
        w2_scale, global_scale, True)
    assert moe_output.shape == (0, 32)

    rms_input = torch.empty((0, 128), device=DEVICE, dtype=dtype)
    rms_output = torch.empty_like(rms_input)
    rms_weight = torch.ones(128, device=DEVICE, dtype=dtype)
    torch.ops._C.rms_norm(rms_output, rms_input, rms_weight, 1e-6)
    residual = torch.empty_like(rms_input)
    torch.ops._C.fused_add_rms_norm(
        rms_input, residual, rms_weight, 1e-6)
    assert rms_output.numel() == rms_input.numel() == residual.numel() == 0

    projected_qkvz = torch.empty((0, 12288), device=DEVICE, dtype=dtype)
    projected_ba = torch.empty((0, 64), device=DEVICE, dtype=dtype)
    conv_state = torch.empty((0, 3, 8192), device=DEVICE, dtype=dtype)
    ssm_state = torch.empty((0, 32, 128, 128), device=DEVICE)
    conv_weight = torch.empty((8192, 4), device=DEVICE, dtype=dtype)
    conv_bias = torch.empty(8192, device=DEVICE, dtype=dtype)
    a_log = torch.empty(32, device=DEVICE)
    dt_bias = torch.empty(32, device=DEVICE, dtype=dtype)
    state_indices = torch.empty(0, device=DEVICE, dtype=torch.int32)
    gdn_output, z = torch.ops._xpu_C.qwen_gdn_decode(
        projected_qkvz, projected_ba, conv_state, ssm_state, conv_weight,
        conv_bias, a_log, dt_bias, state_indices)
    assert gdn_output.shape == z.shape == (0, 32, 128)
